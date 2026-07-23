from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn.functional as F

from scatrope import (
    ScatteringRotaryKernel,
    StandardRoPEKernel,
    discrete_deformation_size,
    sample_smooth_monotone_warp,
)


def center_scale(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    return x / x.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)


def jsd(a: torch.Tensor, b: torch.Tensor) -> float:
    pa = F.softmax(a.float(), dim=-1)
    pb = F.softmax(b.float(), dim=-1)
    m = 0.5 * (pa + pb)
    value = 0.5 * (
        (pa * (pa.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(dim=-1)
        + (pb * (pb.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(dim=-1)
    )
    return float(value.mean())


def relative_drift(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    delta = center_scale(a) - center_scale(b)
    if mask is not None:
        mask4 = mask[None, None, :, :]
        delta = delta.masked_select(mask4)
        base = center_scale(a).masked_select(mask4)
    else:
        base = center_scale(a)
    return float(delta.square().mean().sqrt() / base.square().mean().sqrt().clamp_min(1e-8))


def distance_masks(length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    idx = torch.arange(length, device=device)
    d = (idx[:, None] - idx[None, :]).abs()
    return {
        "near": d <= max(4, length // 16),
        "mid": (d > max(4, length // 16)) & (d <= length // 3),
        "far": d > length // 3,
    }


def timed_call(kernel, q, k, positions, repeats: int = 5) -> float:
    with torch.no_grad():
        for _ in range(2):
            kernel(q, k, positions)
        t0 = time.perf_counter()
        for _ in range(repeats):
            kernel(q, k, positions)
        return (time.perf_counter() - t0) / repeats


def make_models(head_dim: int, length: int):
    return {
        "RoPE": StandardRoPEKernel(head_dim),
        "ScatRoPE-localized": ScatteringRotaryKernel(
            head_dim,
            num_bands=4,
            min_scale=8.0,
            max_scale=float(length),
            demodulation=0.0,
            learnable_scales=False,
            learnable_demodulation=False,
            learnable_band_gains=False,
        ),
        "ScatRoPE-demod": ScatteringRotaryKernel(
            head_dim,
            num_bands=4,
            min_scale=8.0,
            max_scale=float(length),
            demodulation=0.75,
            learnable_scales=False,
            learnable_demodulation=False,
            learnable_band_gains=False,
        ),
    }


def run(args: argparse.Namespace) -> list[dict]:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float32
    q = torch.randn(args.batch, args.heads, args.length, args.head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    positions = torch.arange(args.length, device=device, dtype=dtype)
    masks = distance_masks(args.length, device)
    models = {name: model.to(device) for name, model in make_models(args.head_dim, args.length).items()}

    rows: list[dict] = []
    clean = {name: model(q, k, positions).detach() for name, model in models.items()}
    runtimes = {name: timed_call(model, q, k, positions, args.runtime_repeats) for name, model in models.items()}

    for slope in args.slopes:
        accum: Dict[str, Dict[str, float]] = {
            name: {"overall": 0.0, "near": 0.0, "mid": 0.0, "far": 0.0, "jsd": 0.0, "size": 0.0}
            for name in models
        }
        for trial in range(args.trials):
            gen = torch.Generator(device=device).manual_seed(args.seed + 1000 * trial + int(10_000 * slope))
            warped = sample_smooth_monotone_warp(
                positions, max_slope=slope, num_modes=args.num_modes, generator=gen
            )
            size = float(discrete_deformation_size(positions, warped))
            for name, model in models.items():
                warped_logits = model(q, k, warped).detach()
                accum[name]["overall"] += relative_drift(clean[name], warped_logits)
                for bin_name, mask in masks.items():
                    accum[name][bin_name] += relative_drift(clean[name], warped_logits, mask)
                accum[name]["jsd"] += jsd(clean[name], warped_logits)
                accum[name]["size"] += size

        for name in models:
            vals = {key: value / args.trials for key, value in accum[name].items()}
            rows.append(
                {
                    "model": name,
                    "max_slope": slope,
                    "deformation_size": vals["size"],
                    "relative_logit_drift": vals["overall"],
                    "near_drift": vals["near"],
                    "mid_drift": vals["mid"],
                    "far_drift": vals["far"],
                    "attention_jsd": vals["jsd"],
                    "seconds_per_forward": runtimes[name],
                    "runtime_vs_rope": runtimes[name] / runtimes["RoPE"],
                }
            )
    return rows


def write_outputs(rows: list[dict], output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "deformation_benchmark.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "rows": rows,
    }
    (output_dir / "deformation_benchmark.json").write_text(json.dumps(summary, indent=2))

    # Compact Markdown report.
    by_slope = {}
    for row in rows:
        by_slope.setdefault(row["max_slope"], []).append(row)
    lines = [
        "# ScatRoPE synthetic deformation benchmark",
        "",
        "Metric: normalized attention-logit drift under smooth monotone coordinate warps. Lower is better.",
        "",
        "| max slope | model | logit drift | near | mid | far | attention JSD | runtime vs RoPE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for slope in sorted(by_slope):
        for row in by_slope[slope]:
            lines.append(
                f"| {slope:.3f} | {row['model']} | {row['relative_logit_drift']:.5f} | "
                f"{row['near_drift']:.5f} | {row['mid_drift']:.5f} | {row['far_drift']:.5f} | "
                f"{row['attention_jsd']:.7f} | {row['runtime_vs_rope']:.2f}x |"
            )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--slopes", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--runtime-repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    result_rows = run(cfg)
    write_outputs(result_rows, cfg.output_dir, cfg)
    print(json.dumps(result_rows, indent=2))
