"""Correctness tests for the PyTorch hybrid attention-mask reference."""

import pytest
import torch

from reference import create_hybrid_attention_mask


NEG_INF = float("-inf")


def _mask(**overrides: object) -> torch.Tensor:
    arguments: dict[str, object] = {
        "batch_size": 1,
        "num_heads": 1,
        "seq_len": 3,
        "past_len": 0,
        "sliding_window": 2,
        "full_attention_heads": [True],
    }
    arguments.update(overrides)
    return create_hybrid_attention_mask(**arguments)


def test_full_causal_attention_without_past_context() -> None:
    mask = _mask()
    expected = torch.tensor(
        [
            [0.0, NEG_INF, NEG_INF],
            [0.0, 0.0, NEG_INF],
            [0.0, 0.0, 0.0],
        ]
    )

    torch.testing.assert_close(mask[0, 0], expected)


def test_sliding_window_has_correct_lower_and_upper_bounds() -> None:
    mask = _mask(
        seq_len=4,
        sliding_window=2,
        full_attention_heads=[False],
    )
    expected = torch.tensor(
        [
            [0.0, NEG_INF, NEG_INF, NEG_INF],
            [0.0, 0.0, NEG_INF, NEG_INF],
            [NEG_INF, 0.0, 0.0, NEG_INF],
            [NEG_INF, NEG_INF, 0.0, 0.0],
        ]
    )

    torch.testing.assert_close(mask[0, 0], expected)


def test_hybrid_heads_are_configured_independently() -> None:
    mask = _mask(
        num_heads=3,
        seq_len=3,
        full_attention_heads=torch.tensor([True, False, True]),
    )
    expected_full = torch.tensor(
        [
            [0.0, NEG_INF, NEG_INF],
            [0.0, 0.0, NEG_INF],
            [0.0, 0.0, 0.0],
        ]
    )
    expected_sliding = torch.tensor(
        [
            [0.0, NEG_INF, NEG_INF],
            [0.0, 0.0, NEG_INF],
            [NEG_INF, 0.0, 0.0],
        ]
    )

    torch.testing.assert_close(mask[0, 0], expected_full)
    torch.testing.assert_close(mask[0, 1], expected_sliding)
    torch.testing.assert_close(mask[0, 2], expected_full)


def test_mask_is_replicated_across_batch_elements() -> None:
    mask = _mask(
        batch_size=3,
        num_heads=2,
        full_attention_heads=[True, False],
    )

    assert mask.shape == (3, 2, 3, 3)
    torch.testing.assert_close(mask[1], mask[0])
    torch.testing.assert_close(mask[2], mask[0])


def test_nonzero_past_length_uses_absolute_query_positions() -> None:
    mask = _mask(
        num_heads=2,
        seq_len=2,
        past_len=2,
        full_attention_heads=[True, False],
    )
    expected_full = torch.tensor(
        [
            [0.0, 0.0, 0.0, NEG_INF],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    expected_sliding = torch.tensor(
        [
            [NEG_INF, 0.0, 0.0, NEG_INF],
            [NEG_INF, NEG_INF, 0.0, 0.0],
        ]
    )

    torch.testing.assert_close(mask[0, 0], expected_full)
    torch.testing.assert_close(mask[0, 1], expected_sliding)


def test_sliding_window_of_one_allows_only_current_absolute_position() -> None:
    mask = _mask(
        seq_len=2,
        past_len=3,
        sliding_window=1,
        full_attention_heads=[False],
    )
    expected = torch.tensor(
        [
            [NEG_INF, NEG_INF, NEG_INF, 0.0, NEG_INF],
            [NEG_INF, NEG_INF, NEG_INF, NEG_INF, 0.0],
        ]
    )

    torch.testing.assert_close(mask[0, 0], expected)


def test_window_larger_than_available_context_matches_full_causal() -> None:
    mask = _mask(
        num_heads=2,
        seq_len=3,
        past_len=1,
        sliding_window=20,
        full_attention_heads=[True, False],
    )

    torch.testing.assert_close(mask[0, 1], mask[0, 0])


def test_single_query_token_with_past_context() -> None:
    mask = _mask(
        seq_len=1,
        past_len=2,
        sliding_window=2,
        full_attention_heads=[False],
    )
    expected = torch.tensor([[NEG_INF, 0.0, 0.0]])

    assert mask.shape == (1, 1, 1, 3)
    torch.testing.assert_close(mask[0, 0], expected)


def test_output_shape_dtype_device_and_values() -> None:
    mask = _mask(
        batch_size=2,
        num_heads=2,
        seq_len=2,
        past_len=1,
        full_attention_heads=[True, False],
        dtype=torch.float64,
        device="cpu",
    )

    assert mask.shape == (2, 2, 2, 3)
    assert mask.dtype == torch.float64
    assert mask.device.type == "cpu"
    assert torch.all((mask == 0) | torch.isneginf(mask))
    assert torch.any(mask == 0)
    assert torch.any(torch.isneginf(mask))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_device_is_supported_when_available() -> None:
    mask = _mask(device="cuda")

    assert mask.device.type == "cuda"
    assert torch.all((mask == 0) | torch.isneginf(mask))


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("batch_size", 0),
        ("batch_size", -1),
        ("num_heads", 0),
        ("seq_len", 0),
        ("sliding_window", 0),
        ("sliding_window", -1),
    ],
)
def test_nonpositive_size_arguments_are_rejected(
    argument: str, value: int
) -> None:
    with pytest.raises(ValueError):
        _mask(**{argument: value})


def test_negative_past_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="past_len"):
        _mask(past_len=-1)


@pytest.mark.parametrize("argument", ["batch_size", "past_len"])
def test_noninteger_size_arguments_are_rejected(argument: str) -> None:
    with pytest.raises(TypeError, match=argument):
        _mask(**{argument: 1.5})


@pytest.mark.parametrize(
    "configuration",
    [
        [True],
        [[True, False]],
        torch.tensor([[True, False]]),
    ],
)
def test_incorrect_head_configuration_shape_is_rejected(
    configuration: object,
) -> None:
    with pytest.raises(ValueError, match="shape"):
        _mask(num_heads=2, full_attention_heads=configuration)


@pytest.mark.parametrize(
    "configuration",
    [
        [1],
        torch.tensor([1], dtype=torch.int64),
    ],
)
def test_invalid_head_configuration_dtype_is_rejected(
    configuration: object,
) -> None:
    with pytest.raises(TypeError, match="bool"):
        _mask(full_attention_heads=configuration)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool])
def test_nonfloating_output_dtype_is_rejected(dtype: torch.dtype) -> None:
    with pytest.raises(TypeError, match="floating point"):
        _mask(dtype=dtype)


def test_non_torch_dtype_is_rejected() -> None:
    with pytest.raises(TypeError, match="torch.dtype"):
        _mask(dtype="float32")
