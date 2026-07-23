import torch

from benchmarks.falsification_suite import (
    PRIMARY_VARIANTS,
    TinyCausalLM,
    _kernel_regularization,
    exact_offset_probe,
)
from scatrope import DeformationStabilityRegularizer


def test_exact_offset_probe_returns_finite_metrics():
    rows = exact_offset_probe(
        {
            "train_length": 16,
            "eval_lengths": [16, 24],
            "offsets": [2, 4, 8],
            "trials": 4,
            "head_dim": 16,
            "num_bands": 4,
            "min_scale": 2.0,
            "seed": 1,
        },
        torch.device("cpu"),
    )
    assert rows
    assert {row["model"] for row in rows} == {
        "RoPE",
        "ScatRoPE-localized",
        "ScatRoPE-demod",
    }
    for row in rows:
        assert 0.0 <= row["top1_accuracy"] <= 1.0
        assert 0.0 < row["mean_reciprocal_rank"] <= 1.0
        assert torch.isfinite(torch.tensor(row["target_margin"]))


def test_all_lm_variants_support_longer_contexts():
    tokens = torch.randint(0, 32, (2, 24))
    for variant in PRIMARY_VARIANTS:
        model = TinyCausalLM(
            32,
            16,
            1,
            2,
            variant,
            reference_length=12,
            num_bands=4,
            min_scale=2.0,
            dropout=0.0,
            mlp_ratio=2,
        )
        logits = model(tokens)
        assert logits.shape == (2, 24, 32)
        assert torch.isfinite(logits).all()


def test_matched_regularizer_backpropagates_for_rope_and_scatrope():
    tokens = torch.randint(0, 32, (2, 12))
    positions = torch.arange(12, dtype=torch.float32)
    regularizer = DeformationStabilityRegularizer()
    for variant in (PRIMARY_VARIANTS[1], PRIMARY_VARIANTS[3]):
        model = TinyCausalLM(
            32,
            16,
            1,
            2,
            variant,
            reference_length=12,
            num_bands=4,
            min_scale=2.0,
            dropout=0.0,
            mlp_ratio=2,
        )
        _, aux = model(tokens, positions, return_aux=True)
        loss = _kernel_regularization(
            aux,
            positions,
            max_slope=0.05,
            regularizer=regularizer,
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert any(parameter.grad is not None for parameter in model.parameters())
