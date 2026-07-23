from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from scatrope import (
    DeformationStabilityRegularizer,
    ScatteringRotaryKernel,
    StandardRoPEKernel,
    sample_smooth_monotone_warp,
)


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str
    regularized: bool = False
    demodulation: float = 0.75


PRIMARY_VARIANTS = (
    Variant("RoPE", "rope"),
    Variant("RoPE+Reg", "rope", True),
    Variant("ScatRoPE", "scat"),
    Variant("ScatRoPE+Reg", "scat", True),
)
OFFSET_VARIANTS = (
    Variant("RoPE", "rope"),
    Variant("ScatRoPE-localized", "scat", demodulation=0.0),
    Variant("ScatRoPE-demod", "scat", demodulation=0.75),
)


def make_kernel(variant, head_dim, reference_length, num_bands=4, min_scale=4.0):
    if variant.kind == "rope":
        return StandardRoPEKernel(head_dim)
    return ScatteringRotaryKernel(
        head_dim,
        num_bands=num_bands,
        min_scale=min_scale,
        max_scale=float(reference_length),
        demodulation=variant.demodulation,
        learnable_scales=False,
        learnable_demodulation=False,
        learnable_band_gains=False,
    )


def mean(values):
    values = list(values)
    return sum(values) / max(1, len(values))


def std(values):
    values = list(values)
    mu = mean(values)
    return math.sqrt(mean((value - mu) ** 2 for value in values))


def query_row_scores(kernel, q, k, query_position, key_positions):
    """Score one query against all keys without materializing an LxL tensor."""
    q_rot = kernel.rotate(q, query_position)
    k_rot = kernel.rotate(k, key_positions)
    if isinstance(kernel, StandardRoPEKernel):
        return (q_rot @ k_rot.transpose(-1, -2))[:, 0, 0] / math.sqrt(kernel.head_dim)

    delta = key_positions[None] - query_position.reshape(1, 1)
    scales = kernel.scales.to(q)
    demod = kernel.demodulation.to(q)
    gains = kernel.band_gains.to(q)
    scores = q.new_zeros(q.shape[0], key_positions.numel())
    for band in range(kernel.num_bands):
        start = band * kernel.band_dim
        stop = start + kernel.band_dim
        raw = (q[..., start:stop] @ k[..., start:stop].transpose(-1, -2))[:, 0, 0]
        rotary = (
            q_rot[..., start:stop] @ k_rot[..., start:stop].transpose(-1, -2)
        )[:, 0, 0]
        rho = kernel._envelope(delta, scales[band])
        scores += gains[band] * (rho * rotary + demod[band] * (1.0 - rho) * raw)
    return scores / math.sqrt(kernel.head_dim)


def exact_offset_probe(config, device):
    """Hostile probe: identical keys, with one exact phase-matched displacement."""
    reference = int(config["train_length"])
    trials = int(config["trials"])
    dim = int(config["head_dim"])
    bands = int(config.get("num_bands", 4))
    min_scale = float(config.get("min_scale", 4.0))
    rows = []
    for variant_index, variant in enumerate(OFFSET_VARIANTS):
        kernel = make_kernel(variant, dim, reference, bands, min_scale).to(device).eval()
        generator = torch.Generator(device=device).manual_seed(
            int(config.get("seed", 1234)) + 1009 * variant_index
        )
        for length in map(int, config["eval_lengths"]):
            positions = torch.arange(length, device=device, dtype=torch.float32)
            for offset in map(int, config["offsets"]):
                if not 0 < offset < length:
                    continue
                base = F.normalize(
                    torch.randn(trials, 1, 1, dim, generator=generator, device=device),
                    dim=-1,
                )
                query = kernel.rotate(base, torch.tensor([-float(offset)], device=device))
                keys = base.expand(-1, -1, length, -1).contiguous()
                with torch.no_grad():
                    scores = query_row_scores(kernel, query, keys, positions[-1:], positions)
                target = length - 1 - offset
                target_score = scores[:, target]
                distractors = scores.clone()
                distractors[:, target] = -torch.inf
                rank = 1 + (scores > target_score[:, None]).sum(-1)
                rows.append(
                    {
                        "probe": "exact_offset",
                        "model": variant.name,
                        "reference_length": reference,
                        "eval_length": length,
                        "offset": offset,
                        "offset_ratio": offset / length,
                        "top1_accuracy": float((scores.argmax(-1) == target).float().mean()),
                        "mean_reciprocal_rank": float((1.0 / rank.float()).mean()),
                        "target_margin": float((target_score - distractors.max(-1).values).mean()),
                    }
                )
    return rows


class KernelAttention(nn.Module):
    def __init__(self, dim, heads, variant, reference, bands, min_scale, dropout):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.kernel = make_kernel(variant, self.head_dim, reference, bands, min_scale)

    def forward(self, x, positions, return_aux=False):
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).reshape(
            batch, length, 3, self.heads, self.head_dim
        ).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = self.kernel(q, k, positions)
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        weights = F.softmax(scores.masked_fill(mask, torch.finfo(scores.dtype).min), -1)
        weights = F.dropout(weights, self.dropout, self.training)
        out = (weights @ v).transpose(1, 2).reshape(batch, length, self.dim)
        result = self.output(out)
        return (result, (self.kernel, q, k)) if return_aux else result


class Block(nn.Module):
    def __init__(self, dim, heads, variant, reference, bands, min_scale, dropout, ratio):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = KernelAttention(dim, heads, variant, reference, bands, min_scale, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * ratio), nn.GELU(), nn.Linear(dim * ratio, dim)
        )

    def forward(self, x, positions, return_aux=False):
        if return_aux:
            y, aux = self.attn(self.norm1(x), positions, True)
            x = x + y
            return x + self.mlp(self.norm2(x)), aux
        x = x + self.attn(self.norm1(x), positions)
        return x + self.mlp(self.norm2(x))


class TinyCausalLM(nn.Module):
    def __init__(
        self, vocab, dim, layers, heads, variant, *, reference_length,
        num_bands, min_scale, dropout, mlp_ratio
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim, heads, variant, reference_length, num_bands,
                    min_scale, dropout, mlp_ratio
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab, bias=False)
        self.output.weight = self.embedding.weight
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, positions=None, return_aux=False):
        positions = positions if positions is not None else torch.arange(
            tokens.shape[1], device=tokens.device, dtype=torch.float32
        )
        x, aux_list = self.embedding(tokens), []
        for block in self.blocks:
            if return_aux:
                x, aux = block(x, positions, True)
                aux_list.append(aux)
            else:
                x = block(x, positions)
        logits = self.output(self.norm(x))
        return (logits, aux_list) if return_aux else logits


def synthetic_corpus(size, vocab, seed):
    generator = torch.Generator().manual_seed(seed)
    data = torch.empty(size, dtype=torch.long)
    data[:17] = torch.randint(0, vocab, (17,), generator=generator)
    for index in range(17, size):
        data[index] = (
            data[index - 1] + 3 * data[index - 5] + data[index - 17] + (index // 29) % 7
        ) % vocab
    return data


def load_corpus(path, size, vocab, seed):
    if path:
        raw = Path(path).read_bytes()
        if len(raw) < 100:
            raise ValueError(f"corpus too small: {path}")
        return torch.tensor(list(raw)), 256, "byte_corpus"
    return synthetic_corpus(size, vocab, seed), vocab, "synthetic_smoke"


def sample_batch(data, batch, length, device, generator):
    starts = torch.randint(0, data.numel() - length - 1, (batch,), generator=generator)
    x = torch.stack([data[start : start + length] for start in starts.tolist()]).to(device)
    y = torch.stack([data[start + 1 : start + length + 1] for start in starts.tolist()]).to(device)
    return x, y


def _kernel_regularization(aux_list, positions, max_slope, regularizer):
    warped = sample_smooth_monotone_warp(positions, max_slope=max_slope, num_modes=4)
    return torch.stack(
        [
            regularizer(kernel, q, k, positions, warped_positions=warped).loss
            for kernel, q, k in aux_list
        ]
    ).mean()


@torch.no_grad()
def evaluate_lm(model, data, length, batches, batch, device, seed, slope):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    positions = torch.arange(length, device=device, dtype=torch.float32)
    warp_generator = torch.Generator(device=device).manual_seed(seed + 91337)
    warped = sample_smooth_monotone_warp(
        positions, max_slope=slope, num_modes=4, generator=warp_generator
    )
    clean, changed, agreements = [], [], []
    for _ in range(batches):
        x, y = sample_batch(data, batch, length, device, generator)
        a, b = model(x, positions), model(x, warped)
        clean.append(float(F.cross_entropy(a.flatten(0, 1), y.flatten())))
        changed.append(float(F.cross_entropy(b.flatten(0, 1), y.flatten())))
        agreements.append(float((a.argmax(-1) == b.argmax(-1)).float().mean()))
    clean_loss, warped_loss = mean(clean), mean(changed)
    return {
        "clean_loss": clean_loss,
        "warped_loss": warped_loss,
        "clean_perplexity": math.exp(min(clean_loss, 20)),
        "warped_perplexity": math.exp(min(warped_loss, 20)),
        "prediction_agreement": mean(agreements),
    }


def language_model_probe(config, device, train_file=None, valid_file=None):
    train, vocab_a, source_a = load_corpus(
        train_file, int(config["synthetic_train_size"]),
        int(config["synthetic_vocab_size"]), 711
    )
    valid, vocab_b, source_b = load_corpus(
        valid_file, int(config["synthetic_valid_size"]),
        int(config["synthetic_vocab_size"]), 977
    )
    if vocab_a != vocab_b:
        raise ValueError("training and validation vocabularies differ")
    source = source_a if source_a == source_b else f"{source_a}+{source_b}"
    length, rows = int(config["train_length"]), []
    for seed in map(int, config["seeds"]):
        for variant in PRIMARY_VARIANTS:
            torch.manual_seed(seed)
            model = TinyCausalLM(
                vocab_a, int(config["dim"]), int(config["layers"]), int(config["heads"]),
                variant, reference_length=length, num_bands=int(config.get("num_bands", 4)),
                min_scale=float(config.get("min_scale", 4)),
                dropout=float(config.get("dropout", 0)), mlp_ratio=int(config.get("mlp_ratio", 4))
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(config["learning_rate"]),
                weight_decay=float(config.get("weight_decay", 0.01))
            )
            regularizer = DeformationStabilityRegularizer()
            generator = torch.Generator().manual_seed(seed + 313)
            positions = torch.arange(length, device=device, dtype=torch.float32)
            started = time.perf_counter()
            model.train()
            task_value, reg_value = math.nan, 0.0
            for _ in range(int(config["steps"])):
                x, y = sample_batch(train, int(config["batch_size"]), length, device, generator)
                logits, aux = model(x, positions, True)
                task = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                reg = _kernel_regularization(
                    aux, positions, float(config["regularization_slope"]), regularizer
                ) if variant.regularized else task.new_zeros(())
                loss = task + float(config["regularization_weight"]) * reg
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                task_value, reg_value = float(task.detach()), float(reg.detach())
            tokens = int(config["steps"]) * int(config["batch_size"]) * length
            throughput = tokens / max(time.perf_counter() - started, 1e-9)
            for eval_length in map(int, config["eval_lengths"]):
                metrics = evaluate_lm(
                    model, valid, eval_length, int(config["eval_batches"]),
                    int(config["eval_batch_size"]), device, seed + 10000 + eval_length,
                    float(config["evaluation_slope"])
                )
                rows.append(
                    {
                        "probe": "language_model", "model": variant.name, "seed": seed,
                        "corpus": source, "train_length": length, "eval_length": eval_length,
                        "length_ratio": eval_length / length, "regularized": variant.regularized,
                        "train_tokens": tokens, "tokens_per_second": throughput,
                        "final_train_task_loss": task_value,
                        "final_train_regularization_loss": reg_value, **metrics,
                    }
                )
    return rows, source


def summarize_lm(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["model"], row["eval_length"]), []).append(row)
    fields = (
        "clean_loss", "warped_loss", "clean_perplexity", "warped_perplexity",
        "prediction_agreement", "tokens_per_second"
    )
    output = []
    for (model, length), selected in sorted(groups.items()):
        item = {"model": model, "eval_length": length, "n_seeds": len(selected)}
        for field in fields:
            values = [float(row[field]) for row in selected]
            item[field + "_mean"], item[field + "_std"] = mean(values), std(values)
        output.append(item)
    return output


def evaluate_gates(offset_rows, lm_summary, config, source):
    gates, limits = [], config["gates"]
    reference = int(config["offset_probe"]["train_length"])

    def offset_score(model, extrapolated):
        rows = [
            row for row in offset_rows
            if row["model"] == model
            and row["offset"] / reference >= float(limits["far_offset_ratio"])
            and ((row["eval_length"] > reference) == extrapolated)
        ]
        return mean(row["top1_accuracy"] for row in rows)

    for name, extrapolated, tolerance in (
        ("exact_long_range_retention", False, float(limits["exact_accuracy_tolerance"])),
        ("unseen_length_extrapolation", True, float(limits["extrapolation_accuracy_tolerance"])),
    ):
        rope, scat = offset_score("RoPE", extrapolated), offset_score("ScatRoPE-demod", extrapolated)
        gates.append({"gate": name, "status": "pass" if scat + tolerance >= rope else "fail",
                      "scatrope": scat, "rope": rope, "tolerance": tolerance})
    if not lm_summary:
        return gates + [{"gate": "clean_language_model_perplexity", "status": "not_run"},
                        {"gate": "matched_regularized_benefit", "status": "not_run"}]

    def value(model, length, field):
        return next(row[field] for row in lm_summary if row["model"] == model and row["eval_length"] == length)

    train_length = int(config["language_model"]["train_length"])
    max_length = max(map(int, config["language_model"]["eval_lengths"]))
    tolerance = float(limits["clean_perplexity_relative_tolerance"])
    rope, scat = value("RoPE", train_length, "clean_perplexity_mean"), value("ScatRoPE", train_length, "clean_perplexity_mean")
    gates.append({"gate": "clean_language_model_perplexity",
                  "status": "pass" if scat <= rope * (1 + tolerance) else "fail",
                  "scatrope": scat, "rope": rope, "tolerance": tolerance,
                  "evidence_class": "smoke_only" if source == "synthetic_smoke" else "scientific"})
    gain = float(limits["regularized_warped_perplexity_gain"])
    rope = value("RoPE+Reg", max_length, "warped_perplexity_mean")
    scat = value("ScatRoPE+Reg", max_length, "warped_perplexity_mean")
    gates.append({"gate": "matched_regularized_benefit",
                  "status": "pass" if scat <= rope * (1 - gain) else "fail",
                  "scatrope_reg": scat, "rope_reg": rope, "required_gain": gain,
                  "evidence_class": "smoke_only" if source == "synthetic_smoke" else "scientific"})
    return gates


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(gates, summary, source):
    lines = ["# ScatRoPE falsification report", "", f"Language-model evidence source: `{source}`.",
             "", "## Decision gates", "", "| gate | status |", "|---|---|"]
    lines += [f"| {gate['gate']} | **{gate['status']}** |" for gate in gates]
    if source == "synthetic_smoke":
        lines += ["", "> Synthetic perplexity validates the harness only; it cannot support a language-model claim."]
    if summary:
        lines += ["", "## Language-model summary", "",
                  "| model | eval length | clean PPL | warped PPL | agreement | tokens/s |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in summary:
            lines.append(
                f"| {row['model']} | {row['eval_length']} | {row['clean_perplexity_mean']:.4f} | "
                f"{row['warped_perplexity_mean']:.4f} | {row['prediction_agreement_mean']:.4f} | "
                f"{row['tokens_per_second_mean']:.1f} |"
            )
    return "\n".join(lines) + "\n"


def run(config, args):
    torch.set_num_threads(int(config.get("threads", 1)))
    device, output = torch.device(args.device), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    offsets = exact_offset_probe(config["offset_probe"], device)
    write_csv(output / "exact_offset.csv", offsets)
    lm_rows, source = ([], "not_run") if args.skip_lm else language_model_probe(
        config["language_model"], device, args.train_file, args.valid_file
    )
    write_csv(output / "language_model_runs.csv", lm_rows)
    summary = summarize_lm(lm_rows)
    gates = evaluate_gates(offsets, summary, config, source)
    payload = {"config": config, "corpus_source": source, "offset_rows": offsets,
               "language_model_summary": summary, "gates": gates}
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output / "REPORT.md").write_text(report(gates, summary, source))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/falsification_quick.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/falsification"))
    parser.add_argument("--train-file")
    parser.add_argument("--valid-file")
    parser.add_argument("--skip-lm", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    payload = run(json.loads(args.config.read_text()), args)
    print(json.dumps({"gates": payload["gates"]}, indent=2))


if __name__ == "__main__":
    main()
