# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import shutil
import struct
import subprocess

import pytest

import triton.backends.gaudi.driver as driver_module
from triton._C.libtriton import ir
from triton.backends.gaudi.artifact import ArtifactError, GaudiKernelArtifactV1
from triton.backends.gaudi.compiler import GaudiConfig, GaudiOptions
from triton.backends.gaudi.lowering import GaudiLoweringError, emit_tpc_c, lower_ttir


def _parse_elementwise_add():
    context = ir.context()
    ir.load_dialects(context)
    path = Path(__file__).parents[2] / "test" / "elementwise_add.mlir"
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    return module


def _parse_elementwise_add_bf16():
    context = ir.context()
    ir.load_dialects(context)
    path = Path(__file__).parents[2] / "test" / "elementwise_add_bf16.mlir"
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    return module


def _parse_fused_add_rms_norm_bf16():
    context = ir.context()
    ir.load_dialects(context)
    path = Path(__file__).parents[2] / "test" / "fused_add_rms_norm_bf16.mlir"
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    return module


def _parse_dynamic_quant_bf16_fp8():
    context = ir.context()
    ir.load_dialects(context)
    path = Path(__file__).parents[2] / "test" / "dynamic_quant_bf16_fp8.mlir"
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    return module


def _parse_silu_and_mul_bf16():
    context = ir.context()
    ir.load_dialects(context)
    path = Path(__file__).parents[2] / "test" / "silu_and_mul_bf16.mlir"
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    return module


def test_elementwise_add_lowers_to_native_index_space():
    program = lower_ttir(_parse_elementwise_add())

    assert program.name == "gaudi_add_f32"
    assert program.block_size == 256
    assert program.vector_lanes == 64
    assert program.input_args == (0, 1)
    assert program.output_arg == 2
    assert program.bound_arg == 3
    assert program.expression == {
        "op": "add",
        "lhs": {"op": "load", "arg": 0},
        "rhs": {"op": "load", "arg": 1},
    }
    gtir = str(program)
    assert "gaudi.tpc.region" in gtir
    assert "vector_bits = 2048" in gtir


def test_elementwise_add_emits_tpc_c():
    source = str(emit_tpc_c(lower_ttir(_parse_elementwise_add())))

    assert "get_index_space_offset" in source
    assert "get_index_space_size" in source
    assert "for (int program_id = index_space_start[0]" in source
    assert "v_f32_add_b" in source
    assert "v_f32_ld_tnsr_partial_b" in source
    assert "v_f32_st_tnsr_partial" in source
    assert "if (remaining <= 0)\n            break;" in source
    assert "threadIdx" not in source


def test_bf16_uses_full_2048_bit_tpc_vectors():
    program = lower_ttir(_parse_elementwise_add_bf16())
    source = str(emit_tpc_c(program))

    assert program.tensor_dtype == "bf16"
    assert program.vector_lanes == 128
    assert "bfloat128" in source
    assert "v_bf16_add_b" in source
    assert "v_bf16_ld_tnsr_partial_b" in source


def test_fused_add_rms_norm_lowers_to_multi_output_tpc_program():
    program = lower_ttir(_parse_fused_add_rms_norm_bf16())

    assert program.kind == "fused_add_rms_norm"
    assert program.input_args == (0, 1, 2)
    assert program.output_args == (3, 4)
    assert program.bound_arg is None
    assert program.block_size == 1024
    assert program.parameters == {
        "n_cols": 769,
        "epsilon_arg": 5,
        "max_cols": 1024,
        "vlm_bytes": 2048,
    }
    assert program.manifest()["output_args"] == [3, 4]
    assert "gaudi.execute_fused_add_rms_norm" in str(program)


def test_fused_add_rms_norm_emits_native_reduction_tpc_c():
    source = str(emit_tpc_c(lower_ttir(_parse_fused_add_rms_norm_bf16())))

    assert "__local__ bfloat128 triton_gaudi_summed" in source
    assert "v_bf16_mac_acc32_b" in source
    assert "v_f32_reduce_add" in source
    assert "v_f32_shuffle_b" in source
    assert "v_rsqrt_f32" in source
    assert "for (int row = index_space_start[0]" in source
    assert "coords, arg4, summed, tail_columns - 1, 0" in source
    assert "triton_gaudi_summed[lane / TRITON_GAUDI_VECTOR_LANES]" in source
    assert "output_coords, arg3, result, tail_columns - 1, 0" in source
    assert "threadIdx" not in source


def test_dynamic_quant_lowers_to_mixed_dtype_tpc_program():
    program = lower_ttir(_parse_dynamic_quant_bf16_fp8())

    assert program.kind == "dynamic_quant"
    assert program.input_args == (0,)
    assert program.output_args == (1, 2)
    assert program.tensor_dtype == "fp8e4nv"
    assert program.block_size == 1024
    assert program.vector_lanes == 256
    assert program.parameters == {
        "n_cols": 769,
        "fp8_max": 240.0,
        "scale_epsilon": 1.0e-8,
        "vlm_bytes": 0,
    }
    assert program.access_patterns[0]["mapping"][0]["a"] == 769
    assert program.access_patterns[1]["mapping"][0]["a"] == 769
    assert program.access_patterns[2]["mapping"][0]["a"] == 1
    assert program.manifest()["output_args"] == [1, 2]
    assert "gaudi.execute_dynamic_quant" in str(program)


def test_dynamic_quant_emits_native_fp8_reduction_tpc_c():
    source = str(emit_tpc_c(lower_ttir(_parse_dynamic_quant_bf16_fp8())))

    assert "v_f32_reduce_max" in source
    assert "v_f32_st_tnsr_partial(scale_coords, arg2, scale, 0, 0)" in source
    assert "scaled, SW_RHNE | SW_FP8_BIAS7 | SW_LINEAR" in source
    assert "v_f8_st_tnsr_partial" in source
    assert "1.0e-8f" in source
    assert "/ 240.0f" in source
    assert "threadIdx" not in source


def test_silu_and_mul_lowers_to_asymmetric_row_access_pattern():
    program = lower_ttir(_parse_silu_and_mul_bf16())

    assert program.kind == "silu_and_mul"
    assert program.input_args == (0,)
    assert program.output_args == (1,)
    assert program.block_size == 128
    assert program.index_space_rank == 2
    assert program.program_id_axes == (0, 1)
    assert program.parameters == {
        "n_cols": 3584,
        "input_row_stride": 7168,
        "chunk_size": 128,
        "chunks_per_row": 28,
        "max_cols": 3584,
        "vlm_bytes": 0,
    }
    assert program.access_patterns[0]["mapping"][0]["a"] == 128
    assert program.access_patterns[0]["mapping"][0]["end_b"] == 3711
    assert program.access_patterns[0]["mapping"][1]["index_space_dim"] == 1
    assert program.access_patterns[1]["mapping"][0]["a"] == 128
    assert program.access_patterns[1]["mapping"][0]["end_b"] == 127
    assert program.manifest()["index_space"]["rank"] == 2
    assert "gaudi.execute_silu_and_mul" in str(program)


def test_silu_and_mul_emits_native_f32_sigmoid_tpc_c():
    source = str(emit_tpc_c(lower_ttir(_parse_silu_and_mul_bf16())))

    assert "v_sigmoid_f32(gate_f32.v1)" in source
    assert "gate_f32.v1 * v_sigmoid_f32(gate_f32.v1) * up_f32.v1" in source
    assert "v_bf16_ld_tnsr_partial_b" in source
    assert "v_bf16_st_tnsr_partial" in source
    assert "int5 up_coords = {n_cols + column, row, 0, 0, 0}" in source
    assert "chunk * TRITON_GAUDI_CHUNK_SIZE" in source
    assert "row = index_space_start[1]" in source
    assert "threadIdx" not in source


@pytest.mark.parametrize(
    "parser",
    [
        _parse_elementwise_add,
        _parse_elementwise_add_bf16,
        _parse_fused_add_rms_norm_bf16,
        _parse_dynamic_quant_bf16_fp8,
        _parse_silu_and_mul_bf16,
    ],
)
def test_generated_source_compiles_with_tpc_clang(parser, tmp_path):
    compiler = shutil.which("tpc-clang")
    if compiler is None:
        pytest.skip("tpc-clang is not installed")
    source = tmp_path / "kernel.c"
    output = tmp_path / "kernel.o"
    source.write_text(str(emit_tpc_c(lower_ttir(parser()))))

    subprocess.run(
        [compiler, "-Wall", "-Werror", "-march=gaudi2", "-O2", "-c", source, "-o", output],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.read_bytes().startswith(b"\x7fELF")


def test_unsupported_ttir_operation_fails_closed(tmp_path):
    original = (Path(__file__).parents[2] / "test" / "elementwise_add.mlir").read_text()
    unsupported = original.replace("arith.addf %lhs, %rhs", "arith.maximumf %lhs, %rhs")
    path = tmp_path / "unsupported.mlir"
    path.write_text(unsupported)
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context

    with pytest.raises(GaudiLoweringError, match="strict mode: arith.maximumf"):
        lower_ttir(module)


def test_fused_add_rms_norm_mismatched_divisor_fails_closed(tmp_path):
    original = (Path(__file__).parents[2] / "test" / "fused_add_rms_norm_bf16.mlir").read_text()
    path = tmp_path / "wrong_rms_divisor.mlir"
    path.write_text(original.replace("7.690000e+02 : f32", "7.680000e+02 : f32"))
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context

    with pytest.raises(GaudiLoweringError, match="divide an axis-0 sum by n_cols"):
        lower_ttir(module)


def test_dynamic_quant_wrong_fp8_max_fails_closed(tmp_path):
    original = (Path(__file__).parents[2] / "test" / "dynamic_quant_bf16_fp8.mlir").read_text()
    path = tmp_path / "wrong_dynamic_quant_max.mlir"
    path.write_text(original.replace("2.400000e+02 : f32", "4.480000e+02 : f32"))
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context

    with pytest.raises(GaudiLoweringError, match="scale=\\(max_abs\\+1e-8\\)/240"):
        lower_ttir(module)


def test_dynamic_quant_reversed_scale_division_fails_closed(tmp_path):
    original = (Path(__file__).parents[2] / "test" / "dynamic_quant_bf16_fp8.mlir").read_text()
    path = tmp_path / "reversed_dynamic_quant_division.mlir"
    path.write_text(original.replace(
        "%quantized = arith.divf %input_f32, %scale_splat",
        "%quantized = arith.divf %scale_splat, %input_f32",
    ))
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context

    with pytest.raises(GaudiLoweringError, match="divide the loaded BF16 values by scale"):
        lower_ttir(module)


def test_silu_and_mul_wrong_up_offset_fails_closed(tmp_path):
    original = (Path(__file__).parents[2] / "test" / "silu_and_mul_bf16.mlir").read_text()
    path = tmp_path / "wrong_silu_up_offset.mlir"
    path.write_text(original.replace("%up_offsets = arith.addi %gate_offsets, %n_cols_tensor",
                                     "%up_offsets = arith.addi %gate_offsets, %columns"))
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context

    with pytest.raises(GaudiLoweringError, match="up projection must begin exactly n_cols"):
        lower_ttir(module)


def test_kernel_artifact_round_trip_and_tamper_detection():
    # Artifact validation only requires the ELF magic for this serialization
    # unit test; tpc-clang integration is covered by the compile smoke test.
    elf = b"\x7fELF" + bytes(range(32))
    artifact = GaudiKernelArtifactV1.create({
        "target": "gaudi2",
        "engine": "tpc",
        "kernel_name": "test",
    }, elf)

    decoded = GaudiKernelArtifactV1.from_bytes(artifact.to_bytes())
    assert decoded.artifact_hash == artifact.artifact_hash
    assert decoded.elf == elf

    corrupted = bytearray(artifact.to_bytes())
    corrupted[-1] ^= 1
    with pytest.raises(ArtifactError, match="digest mismatch"):
        GaudiKernelArtifactV1.from_bytes(bytes(corrupted))


def test_runtime_keeps_deduplicated_artifact_handle_until_last_release(monkeypatch):
    class FakeBridge:
        def __init__(self):
            self.unregister_calls = []
            self.launch_calls = []

        @staticmethod
        def _triton_gaudi_register_artifact(artifact_hash, elf, manifest, device):
            return 7

        @staticmethod
        def _triton_gaudi_launch_abi():
            return {
                "major": 1,
                "minor": 8,
                "target": "gaudi2",
                "kernel_guid": "triton_gaudi2_v1",
                "graph_op": True,
            }

        def _triton_gaudi_unregister_artifact(self, handle):
            self.unregister_calls.append(handle)

        def _triton_gaudi_launch(self, handle, grid, stream, tensors, scalars):
            self.launch_calls.append((handle, grid, stream, tensors, scalars))

    bridge = FakeBridge()
    monkeypatch.setattr(driver_module, "_bridge_module", lambda: bridge)
    utils = driver_module.GaudiUtils()
    payload = GaudiKernelArtifactV1.create(
        {
            "target": "gaudi2",
            "engine": "tpc",
            "kernel_name": "deduplicated",
            "arguments": [],
        },
        b"\x7fELF" + bytes(range(32)),
    ).to_bytes()

    first = utils.load_binary("deduplicated", payload, 0, 0)[0]
    second = utils.load_binary("deduplicated", payload, 0, 0)[0]
    assert first == second == 7

    utils.unload_module(first)
    utils.launch(second, (1, 1, 1), 11, ())
    assert bridge.launch_calls == [(7, [1, 1, 1], 11, [], [])]

    utils.unload_module(second)
    assert bridge.unregister_calls == [7, 7]
    with pytest.raises(RuntimeError, match="unknown or released"):
        utils.launch(second, (1, 1, 1), 11, ())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"engine": "cuda"}, "engine must"),
        ({"mode": "fallback"}, "mode must"),
        ({"codegen": "ptx"}, "codegen must"),
    ],
)
def test_gaudi_config_rejects_unknown_backend_semantics(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GaudiConfig(**kwargs)


def test_mme_request_fails_closed_until_graph_partitioning_exists():
    with pytest.raises(ValueError, match="MME graph partitioning is not enabled"):
        GaudiOptions(engine="mme")


def test_tpc_clang_o3_fails_closed_on_gaudi2():
    with pytest.raises(ValueError, match="tpc-clang -O3 is disabled"):
        GaudiOptions(optimization_level=3)


def test_bridge_launch_abi_mismatch_fails_closed():
    class IncompatibleBridge:
        _triton_gaudi_register_artifact = object()
        _triton_gaudi_unregister_artifact = object()
        _triton_gaudi_launch = object()

        @staticmethod
        def _triton_gaudi_launch_abi():
            return {
                "major": 2,
                "minor": 0,
                "target": "gaudi3",
                "kernel_guid": "different",
                "graph_op": False,
            }

    with pytest.raises(RuntimeError, match="incompatible Gaudi Triton launch ABI"):
        driver_module.validate_bridge_launch_abi(IncompatibleBridge())


def test_bridge_launch_abi_v2_accepts_explicit_v1_compatibility_surface():
    class CompatibleV2Bridge:
        _triton_gaudi_register_artifact = object()
        _triton_gaudi_unregister_artifact = object()
        _triton_gaudi_launch = object()

        @staticmethod
        def _triton_gaudi_launch_abi():
            return {
                "major": 2,
                "minor": 0,
                "target": "gaudi2",
                "kernel_guid": "triton_gaudi2_v2",
                "graph_op": True,
                "artifact_abi": 2,
                "typed_scalars": True,
            }

    assert driver_module.validate_bridge_launch_abi(CompatibleV2Bridge())["major"] == 2


def test_bridge_launch_abi_v2_without_v1_compatibility_fails_closed():
    class IncompleteV2Bridge:
        @staticmethod
        def _triton_gaudi_launch_abi():
            return {
                "major": 2,
                "minor": 0,
                "target": "gaudi2",
                "kernel_guid": "triton_gaudi2_v2",
                "graph_op": True,
                "artifact_abi": 2,
                "typed_scalars": True,
            }

    with pytest.raises(RuntimeError, match="missing the Triton launch ABI"):
        driver_module.validate_bridge_launch_abi(IncompleteV2Bridge())


def test_runtime_packs_f32_scalar_parameters_as_ieee_bits(monkeypatch):
    class FakeBridge:
        launch_calls = []

        @staticmethod
        def _triton_gaudi_register_artifact(artifact_hash, elf, manifest, device):
            return 9

        @staticmethod
        def _triton_gaudi_unregister_artifact(handle):
            return None

        @staticmethod
        def _triton_gaudi_launch_abi():
            return {
                "major": 1,
                "minor": 8,
                "target": "gaudi2",
                "kernel_guid": "triton_gaudi2_v1",
                "graph_op": True,
            }

        def _triton_gaudi_launch(self, handle, grid, stream, tensors, scalars):
            self.launch_calls.append((handle, grid, stream, tensors, scalars))

    bridge = FakeBridge()
    monkeypatch.setattr(driver_module, "_bridge_module", lambda: bridge)
    utils = driver_module.GaudiUtils()
    manifest = lower_ttir(_parse_fused_add_rms_norm_bf16()).manifest()
    payload = GaudiKernelArtifactV1.create(manifest, b"\x7fELF" + bytes(range(32))).to_bytes()
    handle = utils.load_binary("fused", payload, 0, 0)[0]
    tensors = [object() for _ in range(5)]

    utils.launch(handle, (7, 1, 1), 11, (*tensors, 1.0e-6))

    epsilon_bits = struct.unpack("<I", struct.pack("<f", 1.0e-6))[0]
    assert bridge.launch_calls == [(9, [7, 1, 1], 11, tensors, [epsilon_bits])]
