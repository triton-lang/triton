# isort: off
# fmt: off
import types
from typing import Callable
import pytest

torch = pytest.importorskip("torch")

import triton_kernels.matmul_ogs_details.opt_flags as opt_flags


class _DummyPrecisionConfig:
    def __init__(self):
        self.weight_scale = None
        self.max_num_imprecise_acc = None
        self.act_scale = None
        self.out_scale = None
        self.enforce_bitwise_invariance = False


def _stub_cuda_props(*_args, **_kwargs):
    return types.SimpleNamespace(multi_processor_count=16)


def setup_amd(monkeypatch):
    monkeypatch.setattr(opt_flags, "get_cdna_version", lambda: 3)
    monkeypatch.setattr(opt_flags.torch.cuda, "get_device_properties", _stub_cuda_props)
    monkeypatch.setattr(
        opt_flags.opt_flags_amd,
        "compute_block_nk",
        lambda *args, **kwargs: (64, 32),
    )


def setup_nvidia(monkeypatch):
    monkeypatch.setattr(opt_flags.torch.cuda, "get_device_properties", _stub_cuda_props)
    monkeypatch.setattr(opt_flags.torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(
        opt_flags.opt_flags_nvidia,
        "compute_block_n",
        lambda n, arch, precision_config: (64, 32),
    )
    monkeypatch.setattr(
        opt_flags.opt_flags_nvidia,
        "compute_grid_size",
        lambda routing_data, batch_size, m, n, block_m, block_n: 4,
    )
    monkeypatch.setattr(
        opt_flags.opt_flags_nvidia,
        "compute_block_k",
        lambda m, k, is_persistent, lhs_dtype, rhs_dtype, precision_config, has_y_acc_in: 32,
    )
    monkeypatch.setattr(
        opt_flags.opt_flags_nvidia,
        "compute_split_k",
        lambda block_k, k, estimated_actual_grid_size: 1,
    )
    monkeypatch.setattr(
        opt_flags.opt_flags_nvidia,
        "compute_num_stages",
        lambda *args, **kwargs: 2,
    )
    monkeypatch.setattr(
        opt_flags.opt_flags_nvidia,
        "compute_num_warps",
        lambda block_m, block_n, is_persistent, precision_config: 4,
    )


def make_split_k_limiter(
    max_size_bytes: float,
    max_split_k: int,
) -> Callable[[int, int, int, int, torch.dtype], int]:
    """Create a ki_split_k callback that respects a memory ceiling and max_split_k.

    Args:
        max_size_bytes: Maximum intermediate size in bytes.
        max_split_k: Maximum allowable split_k value.

    Returns:
        A callable that computes the maximum split_k that keeps the
        intermediate matrix ``split_k * b * m * n`` of the provided dtype under the
        size limit. The value is clamped between 1 and ``max_split_k`` for positive shapes and
        raises ``ValueError`` for non-positive arguments or invalid dtypes.
    """

    if max_size_bytes <= 0:
        raise ValueError("max_size_bytes must be positive")
    if max_split_k < 1:
        raise ValueError("max_split_k must be at least 1")

    def _limit_split_k(b: int, m: int, n: int, k: int, dtype: torch.dtype) -> int:
        del k  # unused but kept for signature compatibility
        elem_size = torch.empty((), dtype=dtype).element_size()
        bytes_per_split = b * m * n * elem_size

        if bytes_per_split <= 0:
            raise ValueError(
                "Invalid arguments: "
                f"{bytes_per_split=} = {b=} * {m=} * {n=} * size(dtype)={elem_size}"
            )

        max_split = int(max_size_bytes // bytes_per_split)
        return min(max_split_k, max(1, max_split))

    return _limit_split_k


def test_make_default_opt_flags_amd_split_k_callable(monkeypatch):
    setup_amd(monkeypatch)

    captured_args = {}

    def split_k_callable(batch_size, m, n, k, out_dtype):
        captured_args["value"] = (batch_size, m, n, k, out_dtype)
        return 5

    precision_config = _DummyPrecisionConfig()
    flags = opt_flags.make_default_opt_flags_amd(
        torch.float16,
        torch.float16,
        torch.float16,
        precision_config,
        2,
        128,
        64,
        32,
        None,
        False,
        False,
        False,
        0,
        False,
        False,
        {"split_k": split_k_callable},
    )

    assert flags.split_k == 5
    assert captured_args["value"] == (2, 128, 64, 32, torch.float16)


def test_make_default_opt_flags_nvidia_split_k_callable(monkeypatch):
    setup_nvidia(monkeypatch)

    captured_args = {}

    def split_k_callable(batch_size, m, n, k, out_dtype):
        captured_args["value"] = (batch_size, m, n, k, out_dtype)
        return 3

    precision_config = _DummyPrecisionConfig()
    flags = opt_flags.make_default_opt_flags_nvidia(
        torch.float16,
        torch.float16,
        torch.float16,
        precision_config,
        4,
        256,
        128,
        64,
        None,
        False,
        False,
        False,
        0,
        False,
        False,
        {"split_k": split_k_callable},
    )

    assert flags.split_k == 3
    assert captured_args["value"] == (4, 256, 128, 64, torch.float16)


def test_split_k_callable_with_max_size_callable(monkeypatch):
    setup_nvidia(monkeypatch)

    batch_size, m, n, k = 4, 256, 128, 64
    bytes_float16 = 2
    intermediate_size = batch_size * m * n * bytes_float16

    def get_flags(_split_k_callable):

        return opt_flags.make_default_opt_flags_nvidia(
            torch.float16,
            torch.float16,
            torch.float16,
            _DummyPrecisionConfig(),
            batch_size,
            m,
            n,
            k,
            None,
            False,
            False,
            False,
            0,
            False,
            False,
            { "split_k": _split_k_callable},
        )

    # Test with a very small allowance that only allows split_k=allowance
    allowance = 2
    max_allowable_split_k = 4
    split_k_callable = make_split_k_limiter(allowance * intermediate_size, max_allowable_split_k)
    flags = get_flags(split_k_callable)

    assert flags.split_k == allowance

    # With a larger allowance, we should bump against the max allowable split_k
    allowance = 8
    max_allowable_split_k = 4
    split_k_callable = make_split_k_limiter(allowance * intermediate_size, max_allowable_split_k)
    flags = get_flags(split_k_callable)

    assert flags.split_k == max_allowable_split_k

    # If we bump up the max_allowable_split_k, we should get the allowance
    allowance = 8
    max_allowable_split_k = 8
    split_k_callable = make_split_k_limiter(allowance * intermediate_size, max_allowable_split_k)
    flags = get_flags(split_k_callable)

    assert flags.split_k == max_allowable_split_k

