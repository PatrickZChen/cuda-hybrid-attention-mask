"""End-to-end comparisons between the CUDA runner and PyTorch oracle."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import NamedTuple

import pytest
import torch

from reference import create_hybrid_attention_mask


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CUDA_UNAVAILABLE_EXIT_CODE = 77


class MaskCase(NamedTuple):
    batch_size: int
    num_heads: int
    seq_len: int
    past_len: int
    sliding_window: int
    full_attention_heads: tuple[bool, ...]


CASES = [
    pytest.param(MaskCase(1, 1, 4, 0, 3, (True,)), id="full-causal"),
    pytest.param(MaskCase(1, 1, 5, 0, 2, (False,)), id="sliding-window"),
    pytest.param(
        MaskCase(1, 3, 4, 0, 2, (True, False, True)), id="mixed-heads"
    ),
    pytest.param(
        MaskCase(2, 2, 3, 0, 2, (True, False)), id="multiple-batches"
    ),
    pytest.param(
        MaskCase(1, 2, 3, 4, 3, (True, False)), id="cached-prefix"
    ),
    pytest.param(MaskCase(1, 1, 3, 2, 1, (False,)), id="window-one"),
    pytest.param(MaskCase(1, 1, 3, 1, 16, (False,)), id="large-window"),
    pytest.param(
        MaskCase(2, 2, 1, 3, 2, (True, False)), id="single-query"
    ),
    pytest.param(
        MaskCase(2, 4, 32, 0, 8, (True,) * 4), id="stress-all-full"
    ),
    pytest.param(
        MaskCase(2, 8, 64, 32, 16, (False,) * 8),
        id="stress-all-sliding-with-past",
    ),
    pytest.param(
        MaskCase(1, 16, 128, 128, 32, (True, False) * 8),
        id="stress-alternating-with-past",
    ),
    pytest.param(
        MaskCase(4, 8, 64, 64, 1, (False,) * 8),
        id="stress-window-one",
    ),
    pytest.param(
        MaskCase(2, 4, 32, 0, 128, (False,) * 4),
        id="stress-window-larger-than-context",
    ),
]


def _conventional_runner_paths() -> list[Path]:
    executable_name = (
        "cuda_mask_runner.exe" if os.name == "nt" else "cuda_mask_runner"
    )
    paths: list[Path] = []
    for build_directory in (
        "build",
        "build-local",
        "cmake-build-debug",
        "cmake-build-release",
    ):
        base = REPOSITORY_ROOT / build_directory
        paths.append(base / executable_name)
        for configuration in ("Debug", "Release", "RelWithDebInfo"):
            paths.append(base / configuration / executable_name)
    return paths


def _find_cuda_runner() -> Path:
    configured_path = os.environ.get("HYBRID_MASK_CUDA_RUNNER")
    if configured_path:
        runner = Path(configured_path).expanduser().resolve()
        if not runner.is_file():
            pytest.fail(
                "HYBRID_MASK_CUDA_RUNNER does not point to a file: "
                f"{runner}"
            )
        return runner

    for candidate in _conventional_runner_paths():
        if candidate.is_file():
            return candidate

    pytest.skip(
        "CUDA runner not found; set HYBRID_MASK_CUDA_RUNNER or build "
        "cuda_mask_runner with CMake"
    )


@pytest.fixture(scope="session")
def cuda_runner() -> Path:
    runner = _find_cuda_runner()
    probe = subprocess.run(
        [str(runner), "--probe"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode == CUDA_UNAVAILABLE_EXIT_CODE:
        reason = probe.stderr.strip() or "CUDA device is unavailable"
        pytest.skip(reason)
    if probe.returncode != 0:
        pytest.fail(
            f"CUDA runner probe failed with exit code {probe.returncode}:\n"
            f"{probe.stderr.strip()}"
        )
    return runner


@pytest.mark.parametrize("case", CASES)
def test_cuda_output_matches_pytorch_reference(
    cuda_runner: Path, case: MaskCase
) -> None:
    expected = create_hybrid_attention_mask(
        batch_size=case.batch_size,
        num_heads=case.num_heads,
        seq_len=case.seq_len,
        past_len=case.past_len,
        sliding_window=case.sliding_window,
        full_attention_heads=case.full_attention_heads,
        dtype=torch.float32,
        device="cpu",
    )

    command = [
        str(cuda_runner),
        str(case.batch_size),
        str(case.num_heads),
        str(case.seq_len),
        str(case.past_len),
        str(case.sliding_window),
        *("1" if is_full else "0" for is_full in case.full_attention_heads),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"CUDA runner failed with exit code {completed.returncode}:\n"
            f"{completed.stderr.strip()}"
        )

    tokens = completed.stdout.split()
    unexpected_tokens = set(tokens) - {"0", "-inf"}
    if unexpected_tokens:
        pytest.fail(f"CUDA runner emitted unexpected values: {unexpected_tokens}")
    if len(tokens) != expected.numel():
        pytest.fail(
            f"CUDA runner emitted {len(tokens)} values; expected "
            f"{expected.numel()}"
        )

    actual = torch.tensor(
        [float(token) for token in tokens], dtype=torch.float32
    ).reshape(expected.shape)
    assert torch.equal(actual, expected)
