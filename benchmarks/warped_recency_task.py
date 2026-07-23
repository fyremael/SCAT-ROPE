from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from scatrope import ScatteringRotaryKernel, StandardRoPEKernel, sample_smooth_monotone_warp


@dataclass
class Metrics:
    clean_accuracy: float
    warped_accuracy: float
    prediction_agreement: float
    clean_loss: float
    warped_loss: float


class RecencyRetriever(nn.Module):
    def __init__(self, input_dim: int, heads: int, head_dim: int, kernel: nn.Module) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(input_dim, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, heads * head_dim, bias=False)
        self.kernel = kernel
        self.head_mix = nn.Parameter(torch.zeros(heads))
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits = self.kernel(q, k, positions)
        query_logits = logits[:, :, -1, :-1]
        weights = F.softmax(self.head_mix, dim=0)
        mixed = torch.einsum("h,bhl->bl", weights, query_logits)
        return mixed * self.logit_scale.clamp(max=5.0).exp()


def make_batch(
    batch: int,
    length: int,
    content_dim: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Last input coordinate marks query tokens.
    x = torch.randn(batch, length, content_dim + 1, device=device, generator=generator)
    x[..., -1] = 0.0
    target = torch.empty(batch, dtype=torch.long, device=device)
    for b in range(batch):
        recent = int(torch.randint(length // 2, length - 4, (1,), generator=generator, device=device))
        older = int(torch.randint(1, max(2, recent - 5), (1,), generator=generator, device=device))
        motif = torch.randn(content_dim, device=device, generator=generator)
        x[b, older, :content_dim] = motif
        x[b, recent, :content_dim] = motif
        x[b, -1, :content_dim] = motif + 0.03 * torch.randn(content_dim, device=device, generator=generator)
        x[b, -1, -1] = 1.0
        target[b] = recent
    return x, target


def make_kernel(name: str, head_dim: int, length: int) -> nn.Module:
    if name == "RoPE":
        return StandardRoPEKernel(head_dim)
    if name == "ScatRoPE-localized":
        return ScatteringRotaryKernel(
            head_dim,
            num_bands=4,
            min_scale=4.0,
            max_scale=float(length),
            demodulation=0.0,
            learnable_scales=False,
            learnable_demodulation=False,
            learnable_band_gains=False,
        )
    if name == "ScatRoPE-demod":
        return ScatteringRotaryKernel(
            head_dim,
            num_bands=4,
            min_scale=4.0,
            max_scale=float(length),
            demodulation=0.75,
            learnable_scales=False,
            learnable_demodulation=False,
            learnable_band_gains=False,
        )
    raise ValueError(name)


@torch.no_grad()
def evaluate(
    model: RecencyRetriever,
    *,
    batches: int,
    batch_size: int,
    length: int,
    content_dim: int,
    slope: float,
    device: torch.device,
    seed: int,
) -> Metrics:
    model.eval()
    positions = torch.arange(length, device=device, dtype=torch.float32)
    gen = torch.Generator(device=device).manual_seed(seed)
    warp_gen = torch.Generator(device=device).manual_seed(seed + 999)
    warped = sample_smooth_monotone_warp(positions, max_slope=slope, num_modes=4, generator=warp_gen)
    clean_correct = warped_correct = agreement = total = 0
    clean_loss = warped_loss = 0.0
    for _ in range(batches):
        x, target = make_batch(batch_size, length, content_dim, device=device, generator=gen)
        clean = model(x, positions)
        changed = model(x, warped)
        clean_pred = clean.argmax(dim=-1)
        warped_pred = changed.argmax(dim=-1)
        clean_correct += int((clean_pred == target).sum())
        warped_correct += int((warped_pred == target).sum())
        agreement += int((clean_pred == warped_pred).sum())
        total += batch_size
        clean_loss += float(F.cross_entropy(clean, target))
        warped_loss += float(F.cross_entropy(changed, target))
    return Metrics(
        clean_accuracy=clean_correct / total,
        warped_accuracy=warped_correct / total,
        prediction_agreement=agreement / total,
        clean_loss=clean_loss / batches,
        warped_loss=warped_loss / batches,
    )


def train_one(name: str, args: argparse.Namespace, seed: int) -> Metrics:
    torch.manual_seed(seed)
    device = torch.device(args.device)
    kernel = make_kernel(name, args.head_dim, args.length).to(device)
    model = RecencyRetriever(args.content_dim + 1, args.heads, args.head_dim, kernel).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    positions = torch.arange(args.length, device=device, dtype=torch.float32)
    gen = torch.Generator(device=device).manual_seed(seed + 17)

    model.train()
    for _ in range(args.steps):
        x, target = make_batch(
            args.batch_size, args.length, args.content_dim, device=device, generator=gen
        )
        logits = model(x, positions)
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return evaluate(
        model,
        batches=args.eval_batches,
        batch_size=args.eval_batch_size,
        length=args.length,
        content_dim=args.content_dim,
        slope=args.test_slope,
        device=device,
        seed=seed + 10_000,
    )


def run(args: argparse.Namespace) -> list[dict]:
    torch.set_num_threads(args.threads)
    rows = []
    names = ["RoPE", "ScatRoPE-localized", "ScatRoPE-demod"]
    for seed in args.seeds:
        for name in names:
            metrics = train_one(name, args, seed)
            rows.append({"model": name, "seed": seed, **asdict(metrics)})
            print(rows[-1], flush=True)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    models = sorted({row["model"] for row in rows})
    fields = ["clean_accuracy", "warped_accuracy", "prediction_agreement", "clean_loss", "warped_loss"]
    output = []
    for model in models:
        selected = [row for row in rows if row["model"] == model]
        summary = {"model": model, "n_seeds": len(selected)}
        for field in fields:
            values = torch.tensor([row[field] for row in selected], dtype=torch.float64)
            summary[field + "_mean"] = float(values.mean())
            summary[field + "_std"] = float(values.std(unbiased=False))
        output.append(summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=48)
    parser.add_argument("--content-dim", type=int, default=24)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--test-slope", type=float, default=0.20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("results/warped_recency"))
    args = parser.parse_args()

    rows = run(args)
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Warped recency retrieval benchmark",
        "",
        "Two identical content keys occur before a query; the correct key is the most recent occurrence. Models train on uniform positions and are evaluated on clean and smoothly warped coordinates.",
        "",
        "| model | clean accuracy | warped accuracy | prediction agreement | clean loss | warped loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['clean_accuracy_mean']:.4f} ± {row['clean_accuracy_std']:.4f} | "
            f"{row['warped_accuracy_mean']:.4f} ± {row['warped_accuracy_std']:.4f} | "
            f"{row['prediction_agreement_mean']:.4f} ± {row['prediction_agreement_std']:.4f} | "
            f"{row['clean_loss_mean']:.4f} | {row['warped_loss_mean']:.4f} |"
        )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
