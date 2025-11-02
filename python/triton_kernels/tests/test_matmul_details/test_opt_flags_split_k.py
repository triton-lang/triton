# isort: off
# fmt: off
import pytest
import types

import torch

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

    fake_target = types.SimpleNamespace(backend="hip")
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
        lambda block_m, block_n, is_persistent, precision_config: 4,
    )

    fake_target = types.SimpleNamespace(backend="cuda")
    monkeypatch.setattr(
        "triton.runtime.driver.active.get_current_target",
        lambda: fake_target,
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

def test_max_allowable_mn_and_split_k_constraints(monkeypatch):
    setup_nvidia(monkeypatch)

    opt_flags._opt_flags = None
    opt_flags.reset_opt_flags_constraints()
    opt_flags.update_opt_flags_constraints(
        {
            "max_allowable_mn": 256,
            # Without split_k, this should raise an error
        }
    )

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
                )

def test_max_allowable_mn(monkeypatch):
    setup_nvidia(monkeypatch)

    batch_size, m, n, k = 1, 256, 256, 256

    def get_flags(split_k, max_mn):
        opt_flags._opt_flags = None
        opt_flags.reset_opt_flags_constraints()
        opt_flags.update_opt_flags_constraints(
            {
                "split_k": split_k,
                "max_allowable_mn": max_mn,
            }
        )
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
