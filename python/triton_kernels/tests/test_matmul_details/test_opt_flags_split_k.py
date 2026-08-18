# isort: off
# fmt: off
import pytest
import types

import torch

import triton_kernels.matmul_details.opt_flags as opt_flags
from triton_kernels.matmul import FusedActivation, PrecisionConfig, init_allocation, _resolve_tma_override
from triton_kernels.tensor_details.dtype import BF16, FP16, FP32

class _DummyPrecisionConfig:
    def __init__(self):
        self.b_mx_scale = None
        self.max_num_imprecise_acc = None
        self.a_mx_scale = None
        self.c_mx_scale = None
        self.enforce_bitwise_invariance = False


def _stub_cuda_props(*_args, **_kwargs):
    return types.SimpleNamespace(multi_processor_count=16)


def setup_amd(monkeypatch):
    monkeypatch.setattr(opt_flags, "get_cdna_version", lambda: 3)
    monkeypatch.setattr(opt_flags, "get_rdna_version", lambda: -1)
    monkeypatch.setattr(opt_flags.torch.cuda, "get_device_properties", _stub_cuda_props)
    monkeypatch.setattr(
        opt_flags.opt_flags_amd,
        "compute_block_nk",
        lambda *args, **kwargs: (64, 32),
    )

    fake_target = types.SimpleNamespace(backend="hip", arch=0)
    monkeypatch.setattr(
        "triton.runtime.driver.active.get_current_target",
        lambda: fake_target,
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
        lambda block_m, block_n, is_persistent, precision_config, constraints: 4,
    )

    fake_target = types.SimpleNamespace(backend="cuda", arch=100)
    monkeypatch.setattr(
        "triton.runtime.driver.active.get_current_target",
        lambda: fake_target,
    )


def test_make_default_opt_flags_amd_split_k_constraint(monkeypatch):
    setup_amd(monkeypatch)

    precision_config = _DummyPrecisionConfig()
    flags = opt_flags.make_default_opt_flags_amd(
        FP16,
        FP16,
        FP16,
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
        torch.float32,
    )

    assert flags.split_k == 3


@pytest.mark.parametrize("swap_xw", [None, False, True])
def test_make_default_opt_flags_nvidia_swap_xw_constraint(monkeypatch, swap_xw):
    setup_nvidia(monkeypatch)
    seen = {}

    def capture_num_stages(*args, **kwargs):
        seen["swap_xw"] = kwargs["swap_xw"]
        return 2

    monkeypatch.setattr(opt_flags.opt_flags_nvidia, "compute_num_stages", capture_num_stages)
    flags = opt_flags.make_default_opt_flags_nvidia(
        torch.float16,
        torch.float16,
        torch.float16,
        _DummyPrecisionConfig(),
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
        {"swap_xw": swap_xw},
        torch.float32,
    )

    assert flags.swap_xw is swap_xw
    assert seen["swap_xw"] is swap_xw


@pytest.mark.parametrize("group_m", [2, 4, 8, 16, 32, 64])
def test_make_default_opt_flags_nvidia_group_m_constraint(monkeypatch, group_m):
    setup_nvidia(monkeypatch)
    flags = opt_flags.make_default_opt_flags_nvidia(
        torch.float16,
        torch.float16,
        torch.float16,
        _DummyPrecisionConfig(),
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
        {"group_m": group_m},
        torch.float32,
    )

    assert flags.group_m == group_m


def test_make_default_opt_flags_nvidia_execution_constraints(monkeypatch):
    setup_nvidia(monkeypatch)
    constraints = {
        "use_output_tma": False,
        "occupancy_target": 2,
        "flatten_loops": False,
        "maxnreg": 192,
    }
    flags = opt_flags.make_default_opt_flags_nvidia(
        torch.float16,
        torch.float16,
        torch.float16,
        _DummyPrecisionConfig(),
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
        constraints,
        torch.float32,
    )

    assert flags.use_output_tma is False
    assert flags.occupancy_target == 2
    assert flags.flatten_loops is False
    assert flags.maxnreg == 192
    assert flags.target_kernel_kwargs["FLATTEN_LOOPS"] is False
    assert flags.target_kernel_kwargs["maxnreg"] == 192


def test_resolve_tma_override():
    assert _resolve_tma_override("use_output_tma", True, None) is True
    assert _resolve_tma_override("use_output_tma", True, False) is False
    assert _resolve_tma_override("use_output_tma", True, True) is True
    with pytest.raises(opt_flags.InapplicableConstraint):
        _resolve_tma_override("use_output_tma", False, True)


def test_split_k_uses_intermediate_out_dtype(monkeypatch):
    setup_nvidia(monkeypatch)

    x = torch.empty((7, 11), dtype=torch.bfloat16)
    w = torch.empty((11, 13), dtype=torch.bfloat16)
    seen = {}

    def capture_num_stages(*args, **kwargs):
        seen["out_dtype"] = args[5]
        return 2

    monkeypatch.setattr(opt_flags.opt_flags_nvidia, "compute_num_stages", capture_num_stages)

    cases = [
        (PrecisionConfig(), torch.float32, FP32),
        (
            PrecisionConfig(intermediate_out_dtype=torch.bfloat16),
            torch.bfloat16,
            BF16,
        ),
    ]
    for precision_config, scratch_dtype, opt_dtype in cases:
        allocation = init_allocation(
            x, w, precision_config, FusedActivation(), None, None, 1, 1,
            types.SimpleNamespace(split_k=3), scratch_dtype,
        )
        assert allocation.scratchpads["matmul"] == ((3, 1, 7, 13), scratch_dtype)

        opt_flags.make_default_opt_flags_nvidia(
            torch.float16, torch.float16, torch.float16, _DummyPrecisionConfig(),
            4, 256, 128, 64, None, False, False, False, 0, False, False,
            {"split_k": 3, "epilogue_subtile": 1},
            scratch_dtype,
        )
        assert seen["out_dtype"] == opt_dtype


def test_max_allowable_mn_and_split_k_constraints(monkeypatch):
    setup_nvidia(monkeypatch)

    opt_flags.reset_opt_flags()
    opt_flags.reset_opt_flags_constraints()
    with opt_flags.scoped_opt_flags_constraints({"max_allowable_mn": 256}):
        # Without split_k, this should raise an error.
        with pytest.raises(opt_flags.InapplicableConstraint):
            opt_flags.make_opt_flags(
                        torch.float16,
                        torch.float16,
                        torch.float16,
                        _DummyPrecisionConfig(),
                        1,
                        256,
                        256,
                        256,
                        None,
                        False,
                        False,
                        False,
                        0,
                        False,
                        None,
                        torch.float32,
                    )

def test_max_allowable_mn(monkeypatch):
    setup_nvidia(monkeypatch)

    batch_size, m, n, k = 1, 256, 256, 256

    def get_flags(split_k, max_mn):
        opt_flags.reset_opt_flags()
        opt_flags.reset_opt_flags_constraints()
        with opt_flags.scoped_opt_flags_constraints(
            {
                "split_k": split_k,
                "max_allowable_mn": max_mn,
            }
        ):
            return opt_flags.make_opt_flags(
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
                True,
                False,
                0,
                False,
                None,
                torch.float32,
            )

    split_k = 6
    # Allowable mn is less than actual mn, so split_k should be set to 1
    max_mn = (m * n) // 2
    flags = get_flags(split_k, max_mn)
    assert flags.split_k == 1

    split_k = 6
    # Allowable mn is more than actual mn, so split_k should be unchanged
    max_mn = (m * n) * 2
    flags = get_flags(split_k, max_mn)
    assert flags.split_k == split_k
