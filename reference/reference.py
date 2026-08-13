"""Ground-truth PyTorch implementation of hybrid causal attention masks."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _validate_positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _validate_non_negative_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _head_configuration(
    full_attention_heads: Sequence[bool] | torch.Tensor,
    num_heads: int,
    device: torch.device,
) -> torch.Tensor:
    """Validate and place the per-head full-attention flags."""
    if isinstance(full_attention_heads, torch.Tensor):
        if full_attention_heads.dtype != torch.bool:
            raise TypeError("full_attention_heads tensor must have dtype torch.bool")
        if full_attention_heads.shape != (num_heads,):
            raise ValueError(
                f"full_attention_heads must have shape ({num_heads},), "
                f"got {tuple(full_attention_heads.shape)}"
            )
        return full_attention_heads.to(device=device)

    try:
        values = tuple(full_attention_heads)
    except TypeError as exc:
        raise TypeError(
            "full_attention_heads must be a one-dimensional sequence of bools "
            "or a torch.bool tensor"
        ) from exc

    if len(values) != num_heads:
        raise ValueError(
            f"full_attention_heads must have shape ({num_heads},), "
            f"got ({len(values)},)"
        )
    if any(type(value) is not bool for value in values):
        raise TypeError("full_attention_heads sequence must contain only bool values")

    return torch.tensor(values, dtype=torch.bool, device=device)


def create_hybrid_attention_mask(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    past_len: int,
    sliding_window: int,
    full_attention_heads: Sequence[bool] | torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create an additive causal mask with independently configured heads.

    ``full_attention_heads[h]`` selects full causal attention for head ``h``
    when true and sliding-window causal attention when false. Query positions
    are relative to the current sequence, while key positions cover both the
    cached prefix and current sequence.

    Returns:
        A tensor shaped ``[batch_size, num_heads, seq_len,
        past_len + seq_len]``. Visible positions contain zero and masked
        positions contain negative infinity.
    """
    _validate_positive_integer("batch_size", batch_size)
    _validate_positive_integer("num_heads", num_heads)
    _validate_positive_integer("seq_len", seq_len)
    _validate_non_negative_integer("past_len", past_len)
    _validate_positive_integer("sliding_window", sliding_window)

    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be floating point for an additive mask")

    try:
        resolved_device = torch.device("cpu" if device is None else device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid device: {device!r}") from exc

    head_is_full = _head_configuration(
        full_attention_heads, num_heads, resolved_device
    )

    key_positions = torch.arange(
        past_len + seq_len, device=resolved_device
    ).unsqueeze(0)
    absolute_query_positions = (
        past_len + torch.arange(seq_len, device=resolved_device)
    ).unsqueeze(1)

    causal = key_positions <= absolute_query_positions
    within_window = key_positions >= (
        absolute_query_positions - sliding_window + 1
    )
    sliding_causal = causal & within_window

    allowed_by_head = torch.where(
        head_is_full[:, None, None],
        causal[None, :, :],
        sliding_causal[None, :, :],
    )
    allowed = allowed_by_head[None, :, :, :].expand(
        batch_size, -1, -1, -1
    )

    zero = torch.zeros((), dtype=dtype, device=resolved_device)
    negative_infinity = torch.full(
        (), float("-inf"), dtype=dtype, device=resolved_device
    )
    return torch.where(allowed, zero, negative_infinity)
