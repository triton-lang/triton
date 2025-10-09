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


def test_make_default_opt_flags_amd_split_k_constraint(monkeypatch):
    setup_amd(monkeypatch)

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
        {"split_k": 5},
    )

    assert flags.split_k == 5


def test_make_default_opt_flags_nvidia_split_k_constraint(monkeypatch):
    setup_nvidia(monkeypatch)

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
        {"split_k": 3},
    )

    assert flags.split_k == 3


def test_dynamic_split_k(monkeypatch):
    setup_nvidia(monkeypatch)

    batch_size, m, n = 4, 256, 128
    k = (2**6) * 3

    bytes_float16 = 2
    intermediate_size = batch_size * m * n * bytes_float16

    def get_flags(split_k, dynamic_split_k_max_size_bytes, dynamic_split_k_max_split_k):
        dynamic_split_k = dynamic_split_k_max_size_bytes is not None
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
            {
                "split_k": split_k,
                "dynamic_split_k": dynamic_split_k,
                "dynamic_split_k_max_size_bytes": dynamic_split_k_max_size_bytes,
                "dynamic_split_k_max_split_k": dynamic_split_k_max_split_k,
            },
        )

    # If `dynamic_split_k` is not specified, we get the specified split_k
    for split_k in [1, 2, 4, 8]:
        flags = get_flags(split_k, None, None)
        assert flags.split_k == split_k

    # If `dynamic_split_k` is specified, then it is computed, and the specified split_k is ignored.
    possible_splits = 6
    # So 6 splits are possible
    assert k % possible_splits == 0
    allowance = possible_splits * intermediate_size
    given_split_k = 3
    flags = get_flags(given_split_k, allowance, None)
    assert flags.split_k == possible_splits

    # If we specify a max split size in the above scenario, it is respected, even though more splits are possible.
    max_split_k = 4
    flags = get_flags(given_split_k, allowance, max_split_k)
    assert flags.split_k == max_split_k

    # When the allowance is low enough, no splits are possible.
    allowance = intermediate_size
    flags = get_flags(given_split_k, allowance, max_split_k)
    assert flags.split_k == 1

    # Extreme case, split_k = k
    allowance = k * intermediate_size
    flags = get_flags(given_split_k, allowance, None)
    assert flags.split_k == k

    # Split k doesn't need to be a divisor of k
    non_divisor_k = 5
    assert k % non_divisor_k != 0
    allowance = non_divisor_k * intermediate_size
    flags = get_flags(None, allowance, None)
    assert flags.split_k == non_divisor_k

