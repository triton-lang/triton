# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any


class GaudiLoweringError(RuntimeError):
    """Fail-closed diagnostic for Triton constructs not yet native on Gaudi."""


@dataclass(frozen=True)
class Argument:
    index: int
    kind: str
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "kind": self.kind, "dtype": self.dtype}


@dataclass(frozen=True)
class Operation:
    name: str
    results: tuple[int, ...]
    operands: tuple[int, ...]
    attributes: dict[str, Any]


@dataclass(frozen=True)
class GaudiProgram:
    name: str
    source_hash: str
    arguments: tuple[Argument, ...]
    operations: tuple[Operation, ...]
    block_size: int
    vector_lanes: int
    output_arg: int
    input_args: tuple[int, ...]
    bound_arg: int | None
    expression: dict[str, Any]
    access_patterns: tuple[dict[str, Any], ...]
    index_space_rank: int = 1
    program_id_axes: tuple[int, ...] = (0,)
    additional_output_args: tuple[int, ...] = ()
    kind: str = "elementwise"
    parameters: dict[str, Any] | None = None

    @property
    def engine(self) -> str:
        return "tpc"

    @property
    def tensor_dtype(self) -> str:
        return self.arguments[self.output_arg].dtype

    @property
    def output_args(self) -> tuple[int, ...]:
        return (self.output_arg, *self.additional_output_args)

    def manifest(self) -> dict[str, Any]:
        return {
            "target": "gaudi2",
            "engine": self.engine,
            "kernel_name": self.name,
            "source_sha256": self.source_hash,
            "tensor_dtype": self.tensor_dtype,
            "arguments": [arg.to_dict() for arg in self.arguments],
            "index_space": {
                "rank": self.index_space_rank,
                "program_id_axes": list(self.program_id_axes),
                "block_size": self.block_size,
                "vector_lanes": self.vector_lanes,
            },
            "access_patterns": list(self.access_patterns),
            "output_arg": self.output_arg,
            "output_args": list(self.output_args),
            "input_args": list(self.input_args),
            "bound_arg": self.bound_arg,
            "kind": self.kind,
            "parameters": self.parameters or {},
            "expression": self.expression,
        }

    def __str__(self) -> str:
        args = ", ".join(
            f"%arg{arg.index}: !gaudi.{arg.kind}<{arg.dtype}>" if arg.kind == "tensor" else
            f"%arg{arg.index}: {arg.dtype}" for arg in self.arguments)
        manifest = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"))
        execution = {
            "elementwise": "      gaudi.execute_elementwise\n",
            "fused_add_rms_norm": "      gaudi.execute_fused_add_rms_norm\n",
            "dynamic_quant": "      gaudi.execute_dynamic_quant\n",
            "silu_and_mul": "      gaudi.execute_silu_and_mul\n",
            "gdn_decode_packed": "      gaudi.execute_gdn_decode_packed\n",
            "gdn_decode_conv_packed": "      gaudi.execute_gdn_decode_conv_packed\n",
            "gdn_qk_conv_packed": "      gaudi.execute_gdn_qk_conv_packed\n",
            "gdn_decode_value_conv_packed":
                "      gaudi.execute_gdn_decode_value_conv_packed\n",
        }.get(self.kind)
        if execution is None:
            raise GaudiLoweringError(f"cannot serialize unsupported Gaudi program kind {self.kind}")
        return (
            "module attributes {gaudi.abi = 1 : i32, gaudi.target = \"gaudi2\"} {\n"
            f"  gaudi.kernel @{self.name}({args}) "
            f"attributes {{gaudi.plan = {json.dumps(manifest)}}} {{\n"
            "    gaudi.tpc.region {\n"
            f"      gaudi.index_space rank = {self.index_space_rank}, "
            f"block = [{self.block_size}], vector_bits = 2048\n"
            f"{execution}"
            "    }\n"
            "    gaudi.return\n"
            "  }\n"
            "}\n"
        )


@dataclass(frozen=True)
class TpcCSource:
    program: GaudiProgram
    source: str

    def __str__(self) -> str:
        return self.source


_IGNORED_OPS = {"builtin.module", "tt.func", "tt.return"}
_STRUCTURAL_OPS = {
    "arith.addi",
    "arith.cmpi",
    "arith.constant",
    "arith.muli",
    "tt.addptr",
    "tt.get_program_id",
    "tt.load",
    "tt.make_range",
    "tt.splat",
    "tt.store",
}
_ELEMENTWISE_OPS = {
    "arith.addf": "add",
    "arith.subf": "sub",
    "arith.mulf": "mul",
}


def _operation_attributes(op) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for name in ("axis", "start", "end", "predicate", "cache", "evict", "isVolatile"):
        try:
            value = op.get_int_attr(name)
        except Exception:
            value = None
        if value is not None:
            attrs[name] = value
    if op.get_name() == "arith.constant":
        value = op.get_constant_value()
        if value is not None:
            attrs["value"] = value
    return attrs


def _collect_operations(module) -> tuple[tuple[Operation, ...], dict[int, Operation]]:
    operations: list[Operation] = []
    producers: dict[int, Operation] = {}

    def collect(op):
        record = Operation(
            op.get_name(),
            tuple(op.get_result(index).id() for index in range(op.get_num_results())),
            tuple(op.get_operand(index).id() for index in range(op.get_num_operands())),
            _operation_attributes(op),
        )
        operations.append(record)
        for result in record.results:
            producers[result] = record

    module.walk(collect)
    return tuple(operations), producers


def _parse_arguments(module) -> tuple[tuple[Argument, ...], dict[int, int]]:
    function = module.get_function(module.get_entry_func_name())
    signature = module.get_function_signature(function)
    arguments: list[Argument] = []
    values: dict[int, int] = {}
    for index, dtype in enumerate(signature):
        if dtype.startswith("*"):
            tensor_dtype = {
                "f8E4M3FN": "fp8e4nv",
                "f8E5M2": "fp8e5",
            }.get(dtype[1:], dtype[1:])
            arguments.append(Argument(index, "tensor", tensor_dtype))
        else:
            arguments.append(Argument(index, "scalar", dtype))
        values[function.args(index).id()] = index
    return tuple(arguments), values


def _producer(value: int, producers: dict[int, Operation], expected: str | None = None) -> Operation:
    op = producers.get(value)
    if op is None or (expected is not None and op.name != expected):
        detail = expected if expected is not None else "a supported operation"
        raise GaudiLoweringError(f"expected {detail} while tracing TTIR value {value}")
    return op


def _argument_from_splat(value: int, producers: dict[int, Operation], argument_values: dict[int, int]) -> int:
    splat = _producer(value, producers, "tt.splat")
    if len(splat.operands) != 1 or splat.operands[0] not in argument_values:
        raise GaudiLoweringError("only splats of kernel arguments are supported in the initial Gaudi fast path")
    return argument_values[splat.operands[0]]


def _constant(value: int, producers: dict[int, Operation]) -> int:
    op = _producer(value, producers, "arith.constant")
    if "value" not in op.attributes or not isinstance(op.attributes["value"], int):
        raise GaudiLoweringError("expected an integer TTIR constant")
    return op.attributes["value"]


def _is_commutative_pair(operation: Operation, lhs: int, rhs: int) -> bool:
    return operation.operands == (lhs, rhs) or operation.operands == (rhs, lhs)


def _expect_unary(value: int, producers: dict[int, Operation], name: str) -> int:
    operation = _producer(value, producers, name)
    if len(operation.operands) != 1:
        raise GaudiLoweringError(f"malformed {name} while matching fused add+RMSNorm")
    return operation.operands[0]


def _match_fused_load(
    value: int,
    expected_arg: int,
    expected_offset: int,
    expected_mask: int,
    producers: dict[int, Operation],
    argument_values: dict[int, int],
) -> None:
    load = _producer(value, producers, "tt.load")
    if len(load.operands) != 3:
        raise GaudiLoweringError("fused add+RMSNorm requires masked loads with an explicit zero value")
    tensor_arg, offset = _trace_pointer(load.operands[0], producers, argument_values)
    if tensor_arg != expected_arg or offset != expected_offset or load.operands[1] != expected_mask:
        raise GaudiLoweringError("fused add+RMSNorm load access does not match the canonical row layout")


def _match_dynamic_quant(
    name: str,
    source: str,
    arguments: tuple[Argument, ...],
    argument_values: dict[int, int],
    operations: tuple[Operation, ...],
    producers: dict[int, Operation],
) -> GaudiProgram | None:
    if "dynamic_quant" not in name:
        return None

    expected_arguments = (
        Argument(0, "tensor", "bf16"),
        Argument(1, "tensor", "fp8e4nv"),
        Argument(2, "tensor", "f32"),
    )
    if arguments != expected_arguments:
        raise GaudiLoweringError(
            "dynamic FP8 quantization ABI requires BF16 input, E4M3 output, and f32 scale tensors; "
            f"got {arguments!r}")

    expected_counts = {
        "arith.addf": 1,
        "arith.addi": 1,
        "arith.cmpi": 1,
        "arith.constant": 5,
        "arith.divf": 2,
        "arith.extf": 1,
        "arith.maxnumf": 1,
        "arith.muli": 1,
        "builtin.module": 1,
        "math.absf": 1,
        "tt.addptr": 3,
        "tt.fp_to_fp": 1,
        "tt.func": 1,
        "tt.get_program_id": 1,
        "tt.load": 1,
        "tt.make_range": 1,
        "tt.reduce": 1,
        "tt.reduce.return": 1,
        "tt.return": 1,
        "tt.splat": 4,
        "tt.store": 2,
    }
    actual_counts: dict[str, int] = {}
    for operation in operations:
        actual_counts[operation.name] = actual_counts.get(operation.name, 0) + 1
    if actual_counts != expected_counts:
        raise GaudiLoweringError(
            "dynamic FP8 quantization TTIR must match the canonical strict-mode operation set")
    if source.count("rounding = rtne") != 1:
        raise GaudiLoweringError("dynamic FP8 quantization requires deterministic round-to-nearest-even")

    program_id = next(operation for operation in operations if operation.name == "tt.get_program_id")
    if program_id.attributes.get("axis") != 0:
        raise GaudiLoweringError("dynamic FP8 quantization supports tl.program_id(0) only")
    row_value = program_id.results[0]
    columns = next(operation for operation in operations if operation.name == "tt.make_range")
    block_size = columns.attributes.get("end")
    if (columns.attributes.get("start") != 0 or not isinstance(block_size, int) or
            block_size <= 0 or block_size > 16384 or block_size & (block_size - 1)):
        raise GaudiLoweringError(
            "dynamic FP8 quantization BLOCK_SIZE must be a power of two no larger than 16384")
    columns_value = columns.results[0]

    row_multiply = next(operation for operation in operations if operation.name == "arith.muli")
    if row_multiply.operands[0] == row_value:
        n_cols_value = row_multiply.operands[1]
    elif row_multiply.operands[1] == row_value:
        n_cols_value = row_multiply.operands[0]
    else:
        raise GaudiLoweringError("dynamic FP8 quantization row stride must equal n_cols")
    n_cols = _constant(n_cols_value, producers)
    if (n_cols <= 0 or n_cols > block_size or
            (n_cols != 1 and n_cols <= block_size // 2)):
        raise GaudiLoweringError("dynamic FP8 quantization n_cols must match the constexpr Triton block")

    row_splats = [
        operation for operation in operations
        if operation.name == "tt.splat" and operation.operands == (row_multiply.results[0],)
    ]
    if len(row_splats) != 1:
        raise GaudiLoweringError("dynamic FP8 quantization requires one canonical row-offset splat")
    row_offset = next(operation for operation in operations if operation.name == "arith.addi")
    if not _is_commutative_pair(row_offset, row_splats[0].results[0], columns_value):
        raise GaudiLoweringError("dynamic FP8 quantization requires contiguous row-major offsets")
    row_offset_value = row_offset.results[0]

    compare = next(operation for operation in operations if operation.name == "arith.cmpi")
    if (compare.attributes.get("predicate") != 2 or compare.operands[0] != columns_value or
            _constant(compare.operands[1], producers) != n_cols):
        raise GaudiLoweringError("dynamic FP8 quantization requires the canonical columns < n_cols mask")
    mask_value = compare.results[0]

    load = next(operation for operation in operations if operation.name == "tt.load")
    if len(load.operands) != 3 or load.operands[1] != mask_value:
        raise GaudiLoweringError("dynamic FP8 quantization requires a zero-masked BF16 input load")
    input_arg, input_offset = _trace_pointer(load.operands[0], producers, argument_values)
    _producer(load.operands[2], producers, "arith.constant")
    if (input_arg != 0 or input_offset != row_offset_value or
            source.count("arith.constant dense<0.000000e+00>") != 1):
        raise GaudiLoweringError("dynamic FP8 quantization input must use the canonical contiguous row layout")
    values_f32 = next(
        operation.results[0] for operation in operations
        if operation.name == "arith.extf" and operation.operands == (load.results[0],)
    )

    absolute = next(operation for operation in operations if operation.name == "math.absf")
    reduction = next(operation for operation in operations if operation.name == "tt.reduce")
    if absolute.operands != (values_f32,) or reduction.operands != (absolute.results[0],) or \
            reduction.attributes.get("axis") != 0:
        raise GaudiLoweringError("dynamic FP8 scale must reduce the row-wise absolute maximum")
    reduce_return = next(operation for operation in operations if operation.name == "tt.reduce.return")
    reducer_max = _producer(reduce_return.operands[0], producers, "arith.maxnumf")
    if any(value in producers or value in argument_values for value in reducer_max.operands):
        raise GaudiLoweringError("dynamic FP8 reduction body must be a pure scalar maximum")

    scale_add = next(operation for operation in operations if operation.name == "arith.addf")
    if reduction.results[0] == scale_add.operands[0]:
        scale_epsilon_value = scale_add.operands[1]
    elif reduction.results[0] == scale_add.operands[1]:
        scale_epsilon_value = scale_add.operands[0]
    else:
        raise GaudiLoweringError("dynamic FP8 scale must add the reference epsilon after max-abs")
    _producer(scale_epsilon_value, producers, "arith.constant")

    divides = [operation for operation in operations if operation.name == "arith.divf"]
    scale_divides = [operation for operation in divides if operation.operands[0] == scale_add.results[0]]
    if len(scale_divides) != 1:
        raise GaudiLoweringError("dynamic FP8 scale must divide max-abs plus epsilon exactly once")
    scale_divide = scale_divides[0]
    _producer(scale_divide.operands[1], producers, "arith.constant")
    scalar_f32_constants = [
        float(value) for value in re.findall(
            r"arith\.constant\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*:\s*f32",
            source,
        )
    ]
    if (len(scalar_f32_constants) != 2 or
            not any(abs(value - 1.0e-8) <= 1.0e-14 for value in scalar_f32_constants) or
            240.0 not in scalar_f32_constants):
        raise GaudiLoweringError(
            "Gaudi2 E4M3 dynamic quantization must use scale=(max_abs+1e-8)/240")
    scale_value = scale_divide.results[0]

    scale_stores = []
    output_stores = []
    for store in (operation for operation in operations if operation.name == "tt.store"):
        if len(store.operands) == 2:
            scale_stores.append(store)
        elif len(store.operands) == 3:
            output_stores.append(store)
    if len(scale_stores) != 1 or len(output_stores) != 1:
        raise GaudiLoweringError("dynamic FP8 quantization requires one scale and one data output")
    scale_store = scale_stores[0]
    scale_pointer = _producer(scale_store.operands[0], producers, "tt.addptr")
    if (len(scale_pointer.operands) != 2 or scale_pointer.operands[0] not in argument_values or
            argument_values[scale_pointer.operands[0]] != 2 or scale_pointer.operands[1] != row_value or
            scale_store.operands[1] != scale_value):
        raise GaudiLoweringError("dynamic FP8 scale output must contain one f32 value per row")

    output_store = output_stores[0]
    output_arg, output_offset = _trace_pointer(output_store.operands[0], producers, argument_values)
    if output_arg != 1 or output_offset != row_offset_value or output_store.operands[2] != mask_value:
        raise GaudiLoweringError("dynamic FP8 output must use the input row layout and mask")
    cast = _producer(output_store.operands[1], producers, "tt.fp_to_fp")
    quantized = _producer(cast.operands[0], producers, "arith.divf")
    if quantized is scale_divide or quantized.operands[0] != values_f32:
        raise GaudiLoweringError("dynamic FP8 output must divide the loaded BF16 values by scale")
    scale_splat = _producer(quantized.operands[1], producers, "tt.splat")
    if scale_splat.operands != (scale_value,):
        raise GaudiLoweringError("dynamic FP8 output must divide by the emitted per-row scale")

    row_mapping = [{
        "tensor_dim": 0,
        "index_space_dim": 0,
        "a": n_cols,
        "start_b": 0,
        "end_b": n_cols - 1,
    }]
    scale_mapping = [{
        "tensor_dim": 0,
        "index_space_dim": 0,
        "a": 1,
        "start_b": 0,
        "end_b": 0,
    }]
    return GaudiProgram(
        name=name,
        source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        arguments=arguments,
        operations=operations,
        block_size=block_size,
        vector_lanes=256,
        output_arg=1,
        additional_output_args=(2,),
        input_args=(0,),
        bound_arg=None,
        expression={
            "op": "dynamic_quant",
            "input": 0,
            "output": 1,
            "scale": 2,
            "n_cols": n_cols,
        },
        access_patterns=(
            {"arg": 0, "role": "input", "mapping": row_mapping},
            {"arg": 1, "role": "output", "mapping": row_mapping},
            {"arg": 2, "role": "output", "mapping": scale_mapping},
        ),
        kind="dynamic_quant",
        parameters={
            "n_cols": n_cols,
            "fp8_max": 240.0,
            "scale_epsilon": 1.0e-8,
            "vlm_bytes": 0,
        },
    )


def _match_fused_add_rms_norm(
    name: str,
    source: str,
    arguments: tuple[Argument, ...],
    argument_values: dict[int, int],
    operations: tuple[Operation, ...],
    producers: dict[int, Operation],
) -> GaudiProgram | None:
    if not any(operation.name == "tt.reduce" for operation in operations):
        return None

    expected_arguments = (
        Argument(0, "tensor", "bf16"),
        Argument(1, "tensor", "bf16"),
        Argument(2, "tensor", "bf16"),
        Argument(3, "tensor", "bf16"),
        Argument(4, "tensor", "bf16"),
        Argument(5, "scalar", "f32"),
    )
    if arguments != expected_arguments:
        raise GaudiLoweringError(
            "fused add+RMSNorm ABI requires five BF16 tensors followed by f32 epsilon")

    expected_counts = {
        "arith.addf": 3,
        "arith.addi": 1,
        "arith.cmpi": 1,
        "arith.constant": 4,
        "arith.divf": 1,
        "arith.extf": 4,
        "arith.mulf": 3,
        "arith.muli": 1,
        "arith.truncf": 2,
        "builtin.module": 1,
        "math.rsqrt": 1,
        "tt.addptr": 5,
        "tt.func": 1,
        "tt.get_program_id": 1,
        "tt.load": 3,
        "tt.make_range": 1,
        "tt.reduce": 1,
        "tt.reduce.return": 1,
        "tt.return": 1,
        "tt.splat": 7,
        "tt.store": 2,
    }
    actual_counts: dict[str, int] = {}
    for operation in operations:
        actual_counts[operation.name] = actual_counts.get(operation.name, 0) + 1
    if actual_counts != expected_counts:
        raise GaudiLoweringError(
            "fused add+RMSNorm TTIR must match the canonical strict-mode operation set")
    if source.count("arith.constant dense<0.000000e+00>") != 1:
        raise GaudiLoweringError("fused add+RMSNorm masked loads must use exactly one zero BF16 constant")

    value_for_argument = {index: value for value, index in argument_values.items()}
    epsilon_value = value_for_argument[5]

    program_id = next(operation for operation in operations if operation.name == "tt.get_program_id")
    if program_id.attributes.get("axis") != 0:
        raise GaudiLoweringError("fused add+RMSNorm supports tl.program_id(0) only")
    columns = next(operation for operation in operations if operation.name == "tt.make_range")
    block_size = columns.attributes.get("end")
    if (columns.attributes.get("start") != 0 or not isinstance(block_size, int) or block_size <= 0 or
            block_size > 8192 or block_size & (block_size - 1)):
        raise GaudiLoweringError("fused add+RMSNorm BLOCK_SIZE must be a power of two no larger than 8192")
    columns_value = columns.results[0]

    row_multiply = next(operation for operation in operations if operation.name == "arith.muli")
    if row_multiply.operands[0] == program_id.results[0]:
        n_cols_value = row_multiply.operands[1]
    elif row_multiply.operands[1] == program_id.results[0]:
        n_cols_value = row_multiply.operands[0]
    else:
        raise GaudiLoweringError("fused add+RMSNorm row stride must equal n_cols")
    n_cols = _constant(n_cols_value, producers)
    if (n_cols <= 0 or n_cols > block_size or
            (n_cols != 1 and n_cols <= block_size // 2)):
        raise GaudiLoweringError("fused add+RMSNorm n_cols must match the constexpr Triton block")

    compare = next(operation for operation in operations if operation.name == "arith.cmpi")
    if (compare.attributes.get("predicate") != 2 or compare.operands[0] != columns_value or
            _constant(compare.operands[1], producers) != n_cols):
        raise GaudiLoweringError("fused add+RMSNorm requires the canonical columns < n_cols mask")
    mask_value = compare.results[0]
    row_splats = [
        operation for operation in operations
        if operation.name == "tt.splat" and operation.operands == (row_multiply.results[0],)
    ]
    if len(row_splats) != 1:
        raise GaudiLoweringError("fused add+RMSNorm requires one canonical row-offset splat")
    row_offset = next(operation for operation in operations if operation.name == "arith.addi")
    if not _is_commutative_pair(row_offset, row_splats[0].results[0], columns_value):
        raise GaudiLoweringError("fused add+RMSNorm requires contiguous row-major offsets")
    row_offset_value = row_offset.results[0]

    stores: dict[int, Operation] = {}
    for store in (operation for operation in operations if operation.name == "tt.store"):
        if len(store.operands) != 3 or store.operands[2] != mask_value:
            raise GaudiLoweringError("fused add+RMSNorm stores must use the canonical n_cols mask")
        output_arg, offset = _trace_pointer(store.operands[0], producers, argument_values)
        if offset != row_offset_value or output_arg in stores:
            raise GaudiLoweringError("fused add+RMSNorm outputs must use distinct contiguous row pointers")
        stores[output_arg] = store
    if set(stores) != {3, 4}:
        raise GaudiLoweringError("fused add+RMSNorm requires output and residual_output tensor arguments")

    summed_bf16 = stores[4].operands[1]
    summed_f32_add = _producer(_expect_unary(summed_bf16, producers, "arith.truncf"), producers, "arith.addf")
    if len(summed_f32_add.operands) != 2:
        raise GaudiLoweringError("malformed fused BF16 residual sum")
    loaded_inputs: list[int] = []
    for operand in summed_f32_add.operands:
        load_value = _expect_unary(operand, producers, "arith.extf")
        load = _producer(load_value, producers, "tt.load")
        tensor_arg, offset = _trace_pointer(load.operands[0], producers, argument_values)
        if tensor_arg not in (0, 1) or tensor_arg in loaded_inputs:
            raise GaudiLoweringError("fused add+RMSNorm residual sum must load hidden_states and residual once")
        _match_fused_load(load_value, tensor_arg, row_offset_value, mask_value, producers, argument_values)
        loaded_inputs.append(tensor_arg)
    if set(loaded_inputs) != {0, 1}:
        raise GaudiLoweringError("fused add+RMSNorm residual sum is missing an input")

    summed_f32_values = [
        operation.results[0] for operation in operations
        if operation.name == "arith.extf" and operation.operands == (summed_bf16,)
    ]
    if len(summed_f32_values) != 1:
        raise GaudiLoweringError("fused add+RMSNorm must widen the rounded residual exactly once")
    summed_f32 = summed_f32_values[0]

    output_value = _expect_unary(stores[3].operands[1], producers, "arith.truncf")
    weighted = _producer(output_value, producers, "arith.mulf")
    weight_ext = None
    normalized_value = None
    for candidate_weight, candidate_normalized in (
            (weighted.operands[0], weighted.operands[1]),
            (weighted.operands[1], weighted.operands[0])):
        extension = producers.get(candidate_weight)
        if extension is None or extension.name != "arith.extf":
            continue
        load = producers.get(extension.operands[0])
        if load is None or load.name != "tt.load":
            continue
        tensor_arg, offset = _trace_pointer(load.operands[0], producers, argument_values)
        if tensor_arg == 2 and offset == columns_value and load.operands[1] == mask_value:
            weight_ext = candidate_weight
            normalized_value = candidate_normalized
            _match_fused_load(extension.operands[0], 2, columns_value, mask_value, producers, argument_values)
            break
    if weight_ext is None or normalized_value is None:
        raise GaudiLoweringError("fused add+RMSNorm output must multiply by the BF16 weight tensor")

    normalized = _producer(normalized_value, producers, "arith.mulf")
    rrms_splat_value = None
    for operand in normalized.operands:
        if operand == summed_f32:
            continue
        splat = producers.get(operand)
        if splat is not None and splat.name == "tt.splat":
            rrms_splat_value = splat.operands[0]
    if rrms_splat_value is None or summed_f32 not in normalized.operands:
        raise GaudiLoweringError("fused add+RMSNorm normalization must multiply the rounded residual by rrms")

    rsqrt_input = _expect_unary(rrms_splat_value, producers, "math.rsqrt")
    epsilon_add = _producer(rsqrt_input, producers, "arith.addf")
    variance_value = None
    if epsilon_add.operands[0] == epsilon_value:
        variance_value = epsilon_add.operands[1]
    elif epsilon_add.operands[1] == epsilon_value:
        variance_value = epsilon_add.operands[0]
    if variance_value is None:
        raise GaudiLoweringError("fused add+RMSNorm rsqrt must add the runtime epsilon argument")

    variance_divide = _producer(variance_value, producers, "arith.divf")
    reduction = _producer(variance_divide.operands[0], producers, "tt.reduce")
    _producer(variance_divide.operands[1], producers, "arith.constant")
    divisor_constants = re.findall(
        r"arith\.constant\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*:\s*f32",
        source,
    )
    if (len(divisor_constants) != 1 or float(divisor_constants[0]) != n_cols or
            reduction.attributes.get("axis") != 0):
        raise GaudiLoweringError("fused add+RMSNorm variance must divide an axis-0 sum by n_cols")
    square = _producer(reduction.operands[0], producers, "arith.mulf")
    if square.operands != (summed_f32, summed_f32):
        raise GaudiLoweringError("fused add+RMSNorm reduction must sum squared rounded residual values")
    reduce_returns = [operation for operation in operations if operation.name == "tt.reduce.return"]
    reducer_add = _producer(reduce_returns[0].operands[0], producers, "arith.addf")
    if any(value in producers or value in argument_values for value in reducer_add.operands):
        raise GaudiLoweringError("fused add+RMSNorm reduction body must be a pure scalar add")

    row_mapping = [{
        "tensor_dim": 0,
        "index_space_dim": 0,
        "a": n_cols,
        "start_b": 0,
        "end_b": n_cols - 1,
    }]
    weight_mapping = [{
        "tensor_dim": 0,
        "index_space_dim": 0,
        "a": 0,
        "start_b": 0,
        "end_b": n_cols - 1,
        "all_required": True,
    }]
    return GaudiProgram(
        name=name,
        source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        arguments=arguments,
        operations=operations,
        block_size=block_size,
        vector_lanes=128,
        output_arg=3,
        additional_output_args=(4,),
        input_args=(0, 1, 2),
        bound_arg=None,
        expression={
            "op": "fused_add_rms_norm",
            "hidden_states": 0,
            "residual": 1,
            "weight": 2,
            "output": 3,
            "residual_output": 4,
            "n_cols": n_cols,
            "epsilon": 5,
        },
        access_patterns=(
            *({"arg": arg, "role": "input", "mapping": row_mapping} for arg in (0, 1)),
            {"arg": 2, "role": "input", "mapping": weight_mapping},
            *({"arg": arg, "role": "output", "mapping": row_mapping} for arg in (3, 4)),
        ),
        kind="fused_add_rms_norm",
        parameters={
            "n_cols": n_cols,
            "epsilon_arg": 5,
            "max_cols": block_size,
            "vlm_bytes": block_size * 2,
        },
    )


def _match_silu_and_mul(
    name: str,
    source: str,
    arguments: tuple[Argument, ...],
    argument_values: dict[int, int],
    operations: tuple[Operation, ...],
    producers: dict[int, Operation],
) -> GaudiProgram | None:
    if not any(operation.name == "math.exp" for operation in operations):
        return None

    expected_arguments = (
        Argument(0, "tensor", "bf16"),
        Argument(1, "tensor", "bf16"),
    )
    if arguments != expected_arguments:
        raise GaudiLoweringError("SiLU-and-mul ABI requires one BF16 input tensor and one BF16 output tensor")

    expected_counts = {
        "arith.addf": 1,
        "arith.addi": 4,
        "arith.cmpi": 1,
        "arith.constant": 6,
        "arith.divf": 1,
        "arith.extf": 2,
        "arith.mulf": 2,
        "arith.muli": 3,
        "arith.negf": 1,
        "arith.truncf": 1,
        "builtin.module": 1,
        "math.exp": 1,
        "tt.addptr": 3,
        "tt.func": 1,
        "tt.get_program_id": 2,
        "tt.load": 2,
        "tt.make_range": 1,
        "tt.return": 1,
        "tt.splat": 5,
        "tt.store": 1,
    }
    actual_counts: dict[str, int] = {}
    for operation in operations:
        actual_counts[operation.name] = actual_counts.get(operation.name, 0) + 1
    if actual_counts != expected_counts:
        raise GaudiLoweringError("SiLU-and-mul TTIR must match the canonical strict-mode operation set")
    if source.count("arith.constant dense<0.000000e+00>") != 1:
        raise GaudiLoweringError("SiLU-and-mul masked loads must share one zero BF16 constant")
    if source.count("arith.constant dense<1.000000e+00>") != 1:
        raise GaudiLoweringError("SiLU-and-mul sigmoid must use one canonical FP32 constant")

    program_ids = {
        operation.attributes.get("axis"): operation
        for operation in operations if operation.name == "tt.get_program_id"
    }
    if set(program_ids) != {0, 1}:
        raise GaudiLoweringError("SiLU-and-mul requires chunk and row program IDs on axes 0 and 1")
    chunk_id = program_ids[0]
    row_id = program_ids[1]

    lane_range = next(operation for operation in operations if operation.name == "tt.make_range")
    block_size = lane_range.attributes.get("end")
    if (lane_range.attributes.get("start") != 0 or not isinstance(block_size, int) or
            block_size < 128 or block_size > 1024 or block_size & (block_size - 1)):
        raise GaudiLoweringError("SiLU-and-mul BLOCK_SIZE must be a power of two in [128, 1024]")
    lane_range_value = lane_range.results[0]

    chunk_products = []
    for multiply in (operation for operation in operations if operation.name == "arith.muli"):
        if chunk_id.results[0] not in multiply.operands:
            continue
        other = multiply.operands[0] if multiply.operands[1] == chunk_id.results[0] else multiply.operands[1]
        if _constant(other, producers) == block_size:
            chunk_products.append(multiply)
    if len(chunk_products) != 1:
        raise GaudiLoweringError("SiLU-and-mul columns must begin at chunk_id * BLOCK_SIZE")
    chunk_splats = [
        operation for operation in operations
        if operation.name == "tt.splat" and operation.operands == (chunk_products[0].results[0],)
    ]
    if len(chunk_splats) != 1:
        raise GaudiLoweringError("SiLU-and-mul requires one canonical chunk-offset splat")
    column_offsets = [
        operation for operation in operations
        if operation.name == "arith.addi" and
        _is_commutative_pair(operation, chunk_splats[0].results[0], lane_range_value)
    ]
    if len(column_offsets) != 1:
        raise GaudiLoweringError("SiLU-and-mul requires contiguous columns within each chunk")
    columns_value = column_offsets[0].results[0]

    compare = next(operation for operation in operations if operation.name == "arith.cmpi")
    if compare.attributes.get("predicate") != 2 or compare.operands[0] != columns_value:
        raise GaudiLoweringError("SiLU-and-mul requires the canonical columns < n_cols mask")
    n_cols = _constant(compare.operands[1], producers)
    if n_cols <= 0 or n_cols > 65536:
        raise GaudiLoweringError("SiLU-and-mul n_cols must be in [1, 65536]")
    mask_value = compare.results[0]
    chunks_per_row = (n_cols + block_size - 1) // block_size

    row_products: dict[int, Operation] = {}
    for multiply in (operation for operation in operations if operation.name == "arith.muli"):
        if chunk_id.results[0] in multiply.operands:
            continue
        if multiply.operands[0] == row_id.results[0]:
            stride = _constant(multiply.operands[1], producers)
        elif multiply.operands[1] == row_id.results[0]:
            stride = _constant(multiply.operands[0], producers)
        else:
            raise GaudiLoweringError("SiLU-and-mul row offsets must be derived from program_id(1)")
        if stride in row_products:
            raise GaudiLoweringError("SiLU-and-mul has duplicate row strides")
        row_products[stride] = multiply
    if set(row_products) != {n_cols, 2 * n_cols}:
        raise GaudiLoweringError("SiLU-and-mul input and output row strides must be 2*n_cols and n_cols")

    row_offsets: dict[int, int] = {}
    for stride, multiply in row_products.items():
        splats = [
            operation for operation in operations
            if operation.name == "tt.splat" and operation.operands == (multiply.results[0],)
        ]
        if len(splats) != 1:
            raise GaudiLoweringError("SiLU-and-mul requires one row-offset splat for each tensor stride")
        offsets = [
            operation for operation in operations
            if operation.name == "arith.addi" and
            _is_commutative_pair(operation, splats[0].results[0], columns_value)
        ]
        if len(offsets) != 1:
            raise GaudiLoweringError("SiLU-and-mul requires canonical contiguous row offsets")
        row_offsets[stride] = offsets[0].results[0]
    gate_offset = row_offsets[2 * n_cols]
    output_offset = row_offsets[n_cols]

    up_offsets = [
        operation for operation in operations
        if operation.name == "arith.addi" and gate_offset in operation.operands and
        any(producers.get(operand) is not None and producers[operand].name == "arith.constant" and
            _constant(operand, producers) == n_cols for operand in operation.operands if operand != gate_offset)
    ]
    if len(up_offsets) != 1:
        raise GaudiLoweringError("SiLU-and-mul up projection must begin exactly n_cols after the gate")
    up_offset = up_offsets[0].results[0]

    stores = [operation for operation in operations if operation.name == "tt.store"]
    store = stores[0]
    if len(store.operands) != 3 or store.operands[2] != mask_value:
        raise GaudiLoweringError("SiLU-and-mul output store must use the canonical n_cols mask")
    output_arg, stored_offset = _trace_pointer(store.operands[0], producers, argument_values)
    if output_arg != 1 or stored_offset != output_offset:
        raise GaudiLoweringError("SiLU-and-mul output must use a contiguous n_cols row layout")

    output_f32 = _expect_unary(store.operands[1], producers, "arith.truncf")
    output_multiply = _producer(output_f32, producers, "arith.mulf")

    def match_load_extension(value: int, expected_offset: int) -> bool:
        extension = producers.get(value)
        if extension is None or extension.name != "arith.extf":
            return False
        load = producers.get(extension.operands[0])
        if load is None or load.name != "tt.load" or len(load.operands) != 3:
            return False
        tensor_arg, load_offset = _trace_pointer(load.operands[0], producers, argument_values)
        return tensor_arg == 0 and load_offset == expected_offset and load.operands[1] == mask_value

    up_value = None
    gated_value = None
    for candidate_up, candidate_gated in (
            (output_multiply.operands[0], output_multiply.operands[1]),
            (output_multiply.operands[1], output_multiply.operands[0])):
        if match_load_extension(candidate_up, up_offset):
            up_value = candidate_up
            gated_value = candidate_gated
            break
    if up_value is None or gated_value is None:
        raise GaudiLoweringError("SiLU-and-mul result must multiply by the up projection")

    gated_multiply = _producer(gated_value, producers, "arith.mulf")
    gate_value = None
    sigmoid_value = None
    for candidate_gate, candidate_sigmoid in (
            (gated_multiply.operands[0], gated_multiply.operands[1]),
            (gated_multiply.operands[1], gated_multiply.operands[0])):
        if match_load_extension(candidate_gate, gate_offset):
            gate_value = candidate_gate
            sigmoid_value = candidate_sigmoid
            break
    if gate_value is None or sigmoid_value is None:
        raise GaudiLoweringError("SiLU-and-mul result must multiply the gate by its sigmoid")

    sigmoid_divide = _producer(sigmoid_value, producers, "arith.divf")
    one_value = sigmoid_divide.operands[0]
    denominator = _producer(sigmoid_divide.operands[1], producers, "arith.addf")
    if one_value not in denominator.operands:
        raise GaudiLoweringError("SiLU-and-mul sigmoid must compute 1 / (1 + exp(-gate))")
    exponential_value = denominator.operands[0] if denominator.operands[1] == one_value else denominator.operands[1]
    exponential = _producer(exponential_value, producers, "math.exp")
    negated_gate = _expect_unary(exponential.operands[0], producers, "arith.negf")
    if negated_gate != gate_value:
        raise GaudiLoweringError("SiLU-and-mul sigmoid exponent must be the negative gate")
    _producer(one_value, producers, "arith.constant")

    input_mapping = [{
        "tensor_dim": 0,
        "index_space_dim": 0,
        "a": block_size,
        "start_b": 0,
        "end_b": n_cols + block_size - 1,
    }, {
        "tensor_dim": 1,
        "index_space_dim": 1,
        "a": 1,
        "start_b": 0,
        "end_b": 0,
    }]
    output_mapping = [{
        "tensor_dim": 0,
        "index_space_dim": 0,
        "a": block_size,
        "start_b": 0,
        "end_b": block_size - 1,
    }, {
        "tensor_dim": 1,
        "index_space_dim": 1,
        "a": 1,
        "start_b": 0,
        "end_b": 0,
    }]
    return GaudiProgram(
        name=name,
        source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        arguments=arguments,
        operations=operations,
        block_size=block_size,
        vector_lanes=128,
        output_arg=1,
        input_args=(0,),
        bound_arg=None,
        expression={
            "op": "silu_and_mul",
            "input": 0,
            "output": 1,
            "n_cols": n_cols,
        },
        access_patterns=(
            {"arg": 0, "role": "input", "mapping": input_mapping},
            {"arg": 1, "role": "output", "mapping": output_mapping},
        ),
        index_space_rank=2,
        program_id_axes=(0, 1),
        kind="silu_and_mul",
        parameters={
            "n_cols": n_cols,
            "input_row_stride": 2 * n_cols,
            "chunk_size": block_size,
            "chunks_per_row": chunks_per_row,
            "max_cols": chunks_per_row * block_size,
            "vlm_bytes": 0,
        },
    )


def _specialized_pointer_argument(
    pointer: int,
    producers: dict[int, Operation],
    argument_values: dict[int, int],
    kernel_name: str,
) -> int:
    addptr = _producer(pointer, producers, "tt.addptr")
    base = addptr.operands[0]
    while base not in argument_values:
        wrapper = producers.get(base)
        if wrapper is None or wrapper.name not in (
                "tt.splat", "tt.broadcast", "tt.expand_dims", "tt.addptr"):
            raise GaudiLoweringError(
                f"{kernel_name} pointers must derive directly from tensor arguments; "
                f"encountered {wrapper.name if wrapper is not None else 'block argument'}")
        base = wrapper.operands[0]
    return argument_values[base]


def _match_gdn_qk_conv_packed(
    name: str,
    source: str,
    arguments: tuple[Argument, ...],
    argument_values: dict[int, int],
    operations: tuple[Operation, ...],
    producers: dict[int, Operation],
) -> GaudiProgram | None:
    if "gdn_qk_conv_packed" not in name:
        return None
    expected_arguments = (
        Argument(0, "tensor", "bf16"),
        Argument(1, "tensor", "bf16"),
        Argument(2, "tensor", "i32"),
        Argument(3, "tensor", "bf16"),
        Argument(4, "tensor", "bf16"),
        Argument(5, "scalar", "i32"),
    )
    if arguments != expected_arguments:
        raise GaudiLoweringError(
            f"packed Q/K convolution has an incompatible ABI: {arguments!r}")
    program_ids = {
        operation.attributes.get("axis")
        for operation in operations if operation.name == "tt.get_program_id"
    }
    if program_ids != {0, 1}:
        raise GaudiLoweringError(
            "packed Q/K convolution requires channel-block and batch program IDs")
    ranges = [operation for operation in operations if operation.name == "tt.make_range"]
    if len(ranges) != 1 or ranges[0].attributes.get("start") != 0:
        raise GaudiLoweringError(
            "packed Q/K convolution requires one zero-based channel range")
    qk_tile = int(ranges[0].attributes.get("end", 0))
    if qk_tile not in (128, 256, 512):
        raise GaudiLoweringError(
            "packed Q/K convolution channel tile must be 128, 256, or 512")
    operation_names = {operation.name for operation in operations}
    required = {
        "arith.addf",
        "arith.addi",
        "arith.extf",
        "arith.mulf",
        "arith.muli",
        "arith.remsi",
        "arith.truncf",
        "math.exp",
        "tt.addptr",
        "tt.load",
        "tt.store",
    }
    missing = sorted(required - operation_names)
    if missing:
        raise GaudiLoweringError(
            "packed Q/K convolution is missing canonical operations: " +
            ", ".join(missing))
    loaded = {
        _specialized_pointer_argument(
            operation.operands[0], producers, argument_values,
            "packed Q/K convolution")
        for operation in operations if operation.name == "tt.load"
    }
    stores = [
        _specialized_pointer_argument(
            operation.operands[0], producers, argument_values,
            "packed Q/K convolution")
        for operation in operations if operation.name == "tt.store"
    ]
    if loaded != {0, 1, 2, 3} or stores.count(0) != 3 or stores.count(4) != 1:
        raise GaudiLoweringError(
            "packed Q/K convolution must read its four inputs, update three history taps, "
            "and emit one Q/K block")
    channel_mapping = [{
        "tensor_dim": 0,
        "index_space_dim": 0,
        "a": qk_tile,
        "start_b": 0,
        "end_b": qk_tile - 1,
    }]
    batch_mapping = [{
        "tensor_dim": 0,
        "index_space_dim": 1,
        "a": 1,
        "start_b": 0,
        "end_b": 0,
    }]
    conv_mapping = [
        *channel_mapping,
        {"tensor_dim": 1, "index_space_dim": 0, "a": 0,
         "start_b": 0, "end_b": 2, "all_required": True},
        {"tensor_dim": 2, "index_space_dim": 1, "a": 0,
         "start_b": 0, "end_b": 0, "all_required": True},
    ]
    matrix_mapping = [
        *channel_mapping,
        {"tensor_dim": 1, "index_space_dim": 1, "a": 1,
         "start_b": 0, "end_b": 0},
    ]
    weight_mapping = [
        *channel_mapping,
        {"tensor_dim": 1, "index_space_dim": 0, "a": 0,
         "start_b": 0, "end_b": 3, "all_required": True},
    ]
    return GaudiProgram(
        name=name,
        source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        arguments=arguments,
        operations=operations,
        block_size=qk_tile,
        vector_lanes=128,
        output_arg=4,
        input_args=(0, 1, 2, 3),
        bound_arg=None,
        expression={"op": "gdn_qk_conv_packed"},
        access_patterns=(
            {"arg": 0, "role": "mutable_input", "mapping": conv_mapping},
            {"arg": 1, "role": "input", "mapping": matrix_mapping},
            {"arg": 2, "role": "input", "mapping": batch_mapping},
            {"arg": 3, "role": "input", "mapping": weight_mapping},
            {"arg": 4, "role": "output", "mapping": matrix_mapping},
        ),
        index_space_rank=2,
        program_id_axes=(0, 1),
        kind="gdn_qk_conv_packed",
        parameters={
            "channels": 4096,
            "packed_width": 10240,
            "conv_width": 4,
            "qk_tile": qk_tile,
            "conv_slots_arg": 5,
            "mutates_args": [0],
            "vlm_bytes": 0,
        },
    )


def _match_gdn_decode_value_conv_packed(
    name: str,
    source: str,
    arguments: tuple[Argument, ...],
    argument_values: dict[int, int],
    operations: tuple[Operation, ...],
    producers: dict[int, Operation],
) -> GaudiProgram | None:
    if "gdn_decode_value_conv_packed" not in name:
        return None
    expected_arguments = (
        Argument(0, "tensor", "bf16"),
        Argument(1, "tensor", "f32"),
        Argument(2, "tensor", "bf16"),
        Argument(3, "tensor", "bf16"),
        Argument(4, "tensor", "bf16"),
        Argument(5, "tensor", "bf16"),
        Argument(6, "tensor", "f32"),
        Argument(7, "tensor", "f32"),
        Argument(8, "tensor", "i32"),
        Argument(9, "tensor", "bf16"),
        Argument(10, "tensor", "bf16"),
        Argument(11, "scalar", "i32"),
        Argument(12, "scalar", "i32"),
    )
    if arguments != expected_arguments:
        raise GaudiLoweringError(
            f"fused value-conv + GDN has an incompatible ABI: {arguments!r}")
    program_ids = {
        operation.attributes.get("axis"): operation
        for operation in operations if operation.name == "tt.get_program_id"
    }
    if set(program_ids) != {0, 1, 2}:
        raise GaudiLoweringError(
            "fused value-conv + GDN requires tile, value-head, and batch program IDs")
    ranges = [operation for operation in operations if operation.name == "tt.make_range"]
    if (len(ranges) != 1 or ranges[0].attributes.get("start") != 0 or
            ranges[0].attributes.get("end") != 128):
        raise GaudiLoweringError(
            "fused value-conv + GDN requires one canonical K=128 vector range")
    tile_candidates: set[int] = set()
    value_pid = program_ids[0].results[0]
    for multiply in (operation for operation in operations
                     if operation.name == "arith.muli"):
        if value_pid not in multiply.operands:
            continue
        other = (multiply.operands[0] if multiply.operands[1] == value_pid
                 else multiply.operands[1])
        constant = producers.get(other)
        if constant is not None and constant.name == "arith.constant":
            value = constant.attributes.get("value")
            if value in (16, 32, 64, 128):
                tile_candidates.add(value)
    if len(tile_candidates) != 1:
        raise GaudiLoweringError(
            "fused value-conv + GDN VALUE_TILE must be 16, 32, 64, or 128")
    value_tile = tile_candidates.pop()
    operation_names = {operation.name for operation in operations}
    required = {
        "arith.addf",
        "arith.addi",
        "arith.cmpf",
        "arith.divsi",
        "arith.extf",
        "arith.mulf",
        "arith.muli",
        "arith.remsi",
        "arith.select",
        "arith.truncf",
        "math.exp",
        "math.log",
        "math.rsqrt",
        "tt.addptr",
        "tt.load",
        "tt.reduce",
        "tt.store",
    }
    missing = sorted(required - operation_names)
    if missing:
        raise GaudiLoweringError(
            "fused value-conv + GDN is missing canonical operations: " +
            ", ".join(missing))
    loaded = {
        _specialized_pointer_argument(
            operation.operands[0], producers, argument_values,
            "fused value-conv + GDN")
        for operation in operations if operation.name == "tt.load"
    }
    stores = [
        _specialized_pointer_argument(
            operation.operands[0], producers, argument_values,
            "fused value-conv + GDN")
        for operation in operations if operation.name == "tt.store"
    ]
    if loaded != set(range(10)):
        raise GaudiLoweringError(
            "fused value-conv + GDN must read all cache, Q/K, QKV, gate, index, and weight inputs")
    if (stores.count(0) != 3 * value_tile or
            stores.count(1) != value_tile or stores.count(10) != value_tile):
        raise GaudiLoweringError(
            "fused value-conv + GDN has non-canonical cache/output stores")
    all_value_channels = [{
        "tensor_dim": 0,
        "index_space_dim": 0,
        "a": 0,
        "start_b": 4096,
        "end_b": 10239,
        "all_required": True,
    }]
    conv_mapping = [
        *all_value_channels,
        {"tensor_dim": 1, "index_space_dim": 0, "a": 0,
         "start_b": 0, "end_b": 2, "all_required": True},
        {"tensor_dim": 2, "index_space_dim": 2, "a": 0,
         "start_b": 0, "end_b": 0, "all_required": True},
    ]
    recurrent_mapping = [
        {"tensor_dim": 0, "index_space_dim": 0, "a": 0,
         "start_b": 0, "end_b": 127, "all_required": True},
        {"tensor_dim": 1, "index_space_dim": 0, "a": value_tile,
         "start_b": 0, "end_b": value_tile - 1},
        {"tensor_dim": 2, "index_space_dim": 1, "a": 1,
         "start_b": 0, "end_b": 0},
        {"tensor_dim": 3, "index_space_dim": 2, "a": 0,
         "start_b": 0, "end_b": 0, "all_required": True},
    ]
    qk_mapping = [
        {"tensor_dim": 0, "index_space_dim": 1, "a": 0,
         "start_b": 0, "end_b": 4095, "all_required": True},
        {"tensor_dim": 1, "index_space_dim": 2, "a": 1,
         "start_b": 0, "end_b": 0},
    ]
    packed_mapping = [
        *all_value_channels,
        {"tensor_dim": 1, "index_space_dim": 2, "a": 1,
         "start_b": 0, "end_b": 0},
    ]
    gate_mapping = [
        {"tensor_dim": 0, "index_space_dim": 1, "a": 1,
         "start_b": 0, "end_b": 0},
        {"tensor_dim": 1, "index_space_dim": 2, "a": 1,
         "start_b": 0, "end_b": 0},
    ]
    head_mapping = [{
        "tensor_dim": 0, "index_space_dim": 1, "a": 1,
        "start_b": 0, "end_b": 0,
    }]
    batch_mapping = [{
        "tensor_dim": 0, "index_space_dim": 2, "a": 1,
        "start_b": 0, "end_b": 0,
    }]
    weight_mapping = [
        *all_value_channels,
        {"tensor_dim": 1, "index_space_dim": 0, "a": 0,
         "start_b": 0, "end_b": 3, "all_required": True},
    ]
    output_mapping = [
        {"tensor_dim": 0, "index_space_dim": 0, "a": value_tile,
         "start_b": 0, "end_b": value_tile - 1},
        {"tensor_dim": 1, "index_space_dim": 1, "a": 1,
         "start_b": 0, "end_b": 0},
        {"tensor_dim": 2, "index_space_dim": 2, "a": 1,
         "start_b": 0, "end_b": 0},
    ]
    return GaudiProgram(
        name=name,
        source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        arguments=arguments,
        operations=operations,
        block_size=value_tile,
        vector_lanes=64,
        output_arg=10,
        input_args=tuple(range(10)),
        bound_arg=None,
        expression={"op": "gdn_decode_value_conv_packed"},
        access_patterns=(
            {"arg": 0, "role": "mutable_input", "mapping": conv_mapping},
            {"arg": 1, "role": "mutable_input", "mapping": recurrent_mapping},
            {"arg": 2, "role": "input", "mapping": qk_mapping},
            {"arg": 3, "role": "input", "mapping": packed_mapping},
            {"arg": 4, "role": "input", "mapping": gate_mapping},
            {"arg": 5, "role": "input", "mapping": gate_mapping},
            {"arg": 6, "role": "input", "mapping": head_mapping},
            {"arg": 7, "role": "input", "mapping": head_mapping},
            {"arg": 8, "role": "input", "mapping": batch_mapping},
            {"arg": 9, "role": "input", "mapping": weight_mapping},
            {"arg": 10, "role": "output", "mapping": output_mapping},
        ),
        index_space_rank=3,
        program_id_axes=(0, 1, 2),
        kind="gdn_decode_value_conv_packed",
        parameters={
            "key_heads": 16,
            "value_heads": 48,
            "key_dim": 128,
            "value_dim": 128,
            "packed_width": 10240,
            "qk_width": 4096,
            "conv_width": 4,
            "value_tile": value_tile,
            "conv_slots_arg": 11,
            "state_slots_arg": 12,
            "mutates_args": [0, 1],
            "vlm_bytes": 0,
        },
    )


def _match_gdn_decode_conv_packed(
    name: str,
    source: str,
    arguments: tuple[Argument, ...],
    argument_values: dict[int, int],
    operations: tuple[Operation, ...],
    producers: dict[int, Operation],
) -> GaudiProgram | None:
    """Match the barrier-free fused width-4 conv + Qwen3.5 GDN kernel."""
    if "gdn_decode_conv_packed" not in name:
        return None

    expected_arguments = (
        Argument(0, "tensor", "bf16"),  # convolution state cache
        Argument(1, "tensor", "f32"),   # recurrent state cache
        Argument(2, "tensor", "bf16"),  # raw packed q/k/v
        Argument(3, "tensor", "bf16"),  # decay gate input
        Argument(4, "tensor", "bf16"),  # beta gate input
        Argument(5, "tensor", "f32"),   # log decay coefficient
        Argument(6, "tensor", "f32"),   # decay bias
        Argument(7, "tensor", "i32"),   # state indices
        Argument(8, "tensor", "bf16"),  # transposed width-4 conv weights
        Argument(9, "tensor", "bf16"),  # output
        Argument(10, "scalar", "i32"),  # convolution slot count
        Argument(11, "scalar", "i32"),  # recurrent slot count
    )
    if arguments != expected_arguments:
        raise GaudiLoweringError(
            "fused conv+GDN ABI requires BF16 conv state/raw QKV/gates/weights/output, "
            "FP32 recurrent state/decay parameters, i32 indices, and two i32 slot counts; "
            f"got {arguments!r}")

    program_ids = {
        operation.attributes.get("axis"): operation
        for operation in operations if operation.name == "tt.get_program_id"
    }
    if set(program_ids) != {0, 1}:
        raise GaudiLoweringError(
            "fused conv+GDN requires key-head and batch program IDs")

    ranges = [operation for operation in operations if operation.name == "tt.make_range"]
    if (len(ranges) != 1 or ranges[0].attributes.get("start") != 0 or
            ranges[0].attributes.get("end") != 128):
        raise GaudiLoweringError(
            "fused conv+GDN requires one canonical K=128 vector range")

    operation_names = {operation.name for operation in operations}
    required_operations = {
        "arith.addf",
        "arith.addi",
        "arith.cmpf",
        "arith.extf",
        "arith.mulf",
        "arith.muli",
        "arith.remsi",
        "arith.select",
        "arith.truncf",
        "math.exp",
        "math.log",
        "math.rsqrt",
        "scf.for",
        "tt.addptr",
        "tt.load",
        "tt.reduce",
        "tt.store",
    }
    missing = sorted(required_operations - operation_names)
    if missing:
        raise GaudiLoweringError(
            "fused conv+GDN is missing canonical TTIR operations: " +
            ", ".join(missing))

    allowed_operations = {
        "builtin.module",
        "tt.func",
        "tt.return",
        "tt.reduce.return",
        "tt.get_program_id",
        "tt.make_range",
        "tt.splat",
        "tt.broadcast",
        "tt.expand_dims",
        "tt.addptr",
        "tt.load",
        "tt.store",
        "tt.reduce",
        "scf.for",
        "scf.yield",
        "arith.constant",
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.negf",
        "arith.extf",
        "arith.truncf",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.divsi",
        "arith.remsi",
        "arith.cmpf",
        "arith.cmpi",
        "arith.select",
        "math.exp",
        "math.log",
        "math.rsqrt",
    }
    unsupported = sorted(operation_names - allowed_operations)
    if unsupported:
        raise GaudiLoweringError(
            "fused conv+GDN contains unsupported TTIR operations: " +
            ", ".join(unsupported))

    def pointer_argument(pointer: int) -> int:
        addptr = _producer(pointer, producers, "tt.addptr")
        base = addptr.operands[0]
        while base not in argument_values:
            wrapper = producers.get(base)
            if wrapper is None or wrapper.name not in (
                    "tt.splat", "tt.broadcast", "tt.expand_dims", "tt.addptr"):
                raise GaudiLoweringError(
                    "fused conv+GDN pointers must derive directly from tensor arguments; "
                    f"encountered {wrapper.name if wrapper is not None else 'block argument'}")
            base = wrapper.operands[0]
        return argument_values[base]

    loaded_arguments = {
        pointer_argument(operation.operands[0])
        for operation in operations if operation.name == "tt.load"
    }
    if loaded_arguments != set(range(9)):
        raise GaudiLoweringError(
            "fused conv+GDN must read both caches, raw QKV, gates, decay parameters, "
            "indices, and transposed convolution weights")

    stored_arguments = [
        pointer_argument(operation.operands[0])
        for operation in operations if operation.name == "tt.store"
    ]
    if set(stored_arguments) != {0, 1, 9}:
        raise GaudiLoweringError(
            "fused conv+GDN stores must target only the two mutable caches and output")
    if any(stored_arguments.count(index) == 0 for index in (0, 1, 9)):
        raise GaudiLoweringError("fused conv+GDN must update both caches and emit output")

    counts: dict[str, int] = {}
    for operation in operations:
        counts[operation.name] = counts.get(operation.name, 0) + 1
    if counts.get("math.rsqrt") != 2 or counts.get("tt.reduce", 0) < 4:
        raise GaudiLoweringError(
            "fused conv+GDN must contain canonical Q/K normalization and recurrent reductions")
    if (counts.get("math.exp", 0) < 6 or counts.get("math.log", 0) < 1 or
            counts.get("arith.cmpf", 0) < 1 or counts.get("arith.select", 0) < 1):
        raise GaudiLoweringError(
            "fused conv+GDN must contain convolution SiLU and gate nonlinearities")

    number_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
    canonical_f32 = {
        float(value)
        for value in (
            re.findall(rf"arith\.constant\s+({number_pattern})\s*:\s*f32", source) +
            re.findall(
                rf"arith\.constant\s+dense<({number_pattern})>\s*:\s*tensor<[^>]*f32>",
                source,
            )
        )
    }
    for required in (1.0, 20.0, 1.0e-6, 0.08838834764831845):
        if not any(abs(value - required) <= max(1.0e-12, abs(required) * 1.0e-6)
                   for value in canonical_f32):
            raise GaudiLoweringError(
                f"fused conv+GDN is missing canonical FP32 constant {required}")

    channel_window = [
        {"tensor_dim": 0, "index_space_dim": 0, "a": 128,
         "start_b": 0, "end_b": 4479},
    ]
    conv_state_mapping = [
        *channel_window,
        {"tensor_dim": 1, "index_space_dim": 0, "a": 0,
         "start_b": 0, "end_b": 2, "all_required": True},
        {"tensor_dim": 2, "index_space_dim": 1, "a": 0,
         "start_b": 0, "end_b": 0, "all_required": True},
    ]
    recurrent_state_mapping = [
        {"tensor_dim": 0, "index_space_dim": 0, "a": 0,
         "start_b": 0, "end_b": 127, "all_required": True},
        {"tensor_dim": 1, "index_space_dim": 0, "a": 0,
         "start_b": 0, "end_b": 127, "all_required": True},
        {"tensor_dim": 2, "index_space_dim": 0, "a": 3,
         "start_b": 0, "end_b": 2},
        {"tensor_dim": 3, "index_space_dim": 1, "a": 0,
         "start_b": 0, "end_b": 0, "all_required": True},
    ]
    packed_mapping = [
        *channel_window,
        {"tensor_dim": 1, "index_space_dim": 1, "a": 1,
         "start_b": 0, "end_b": 0},
    ]
    head_batch_mapping = [
        {"tensor_dim": 0, "index_space_dim": 0, "a": 3,
         "start_b": 0, "end_b": 2},
        {"tensor_dim": 1, "index_space_dim": 1, "a": 1,
         "start_b": 0, "end_b": 0},
    ]
    head_mapping = [{
        "tensor_dim": 0,
        "index_space_dim": 0,
        "a": 3,
        "start_b": 0,
        "end_b": 2,
    }]
    batch_mapping = [{
        "tensor_dim": 0,
        "index_space_dim": 1,
        "a": 1,
        "start_b": 0,
        "end_b": 0,
    }]
    weight_mapping = [
        *channel_window,
        {"tensor_dim": 1, "index_space_dim": 0, "a": 0,
         "start_b": 0, "end_b": 3, "all_required": True},
    ]
    output_mapping = [
        {"tensor_dim": 0, "index_space_dim": 0, "a": 0,
         "start_b": 0, "end_b": 127, "all_required": True},
        {"tensor_dim": 1, "index_space_dim": 0, "a": 3,
         "start_b": 0, "end_b": 2},
        {"tensor_dim": 2, "index_space_dim": 1, "a": 1,
         "start_b": 0, "end_b": 0},
    ]
    return GaudiProgram(
        name=name,
        source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        arguments=arguments,
        operations=operations,
        block_size=128,
        vector_lanes=64,
        output_arg=9,
        input_args=tuple(range(9)),
        bound_arg=None,
        expression={
            "op": "gdn_decode_conv_packed",
            "conv_state": 0,
            "state_cache": 1,
            "packed_qkv": 2,
            "gate_a": 3,
            "gate_b": 4,
            "a_log": 5,
            "dt_bias": 6,
            "state_indices": 7,
            "conv_weight_t": 8,
            "output": 9,
            "conv_slots": 10,
            "state_slots": 11,
        },
        access_patterns=(
            {"arg": 0, "role": "mutable_input", "mapping": conv_state_mapping},
            {"arg": 1, "role": "mutable_input", "mapping": recurrent_state_mapping},
            {"arg": 2, "role": "input", "mapping": packed_mapping},
            {"arg": 3, "role": "input", "mapping": head_batch_mapping},
            {"arg": 4, "role": "input", "mapping": head_batch_mapping},
            {"arg": 5, "role": "input", "mapping": head_mapping},
            {"arg": 6, "role": "input", "mapping": head_mapping},
            {"arg": 7, "role": "input", "mapping": batch_mapping},
            {"arg": 8, "role": "input", "mapping": weight_mapping},
            {"arg": 9, "role": "output", "mapping": output_mapping},
        ),
        index_space_rank=2,
        program_id_axes=(0, 1),
        kind="gdn_decode_conv_packed",
        parameters={
            "key_heads": 16,
            "value_heads": 48,
            "key_dim": 128,
            "value_dim": 128,
            "packed_width": 10240,
            "conv_width": 4,
            "conv_slots_arg": 10,
            "state_slots_arg": 11,
            "mutates_args": [0, 1],
            "vlm_bytes": 0,
        },
    )


def _match_gdn_decode_packed(
    name: str,
    source: str,
    arguments: tuple[Argument, ...],
    argument_values: dict[int, int],
    operations: tuple[Operation, ...],
    producers: dict[int, Operation],
) -> GaudiProgram | None:
    """Match the Qwen3.5 single-token packed GDN decode fast path.

    This is deliberately a strict, model-shape specialization.  The backend
    emits a state-mutating TPC kernel, so accepting a merely similar graph
    would be unsafe: all tensor dtypes, program axes, reductions, nonlinear
    operations, state/output stores, and the value-row tile must agree with
    the canonical Triton source.
    """
    if "gdn_decode_packed" not in name:
        return None

    def pointer_argument(pointer: int) -> int:
        addptr = _producer(pointer, producers, "tt.addptr")
        base = addptr.operands[0]
        while base not in argument_values:
            wrapper = producers.get(base)
            if wrapper is None or wrapper.name not in (
                    "tt.splat", "tt.broadcast", "tt.expand_dims", "tt.addptr"):
                raise GaudiLoweringError(
                    "packed GDN decode pointers must be derived directly from tensor arguments; "
                    f"encountered {wrapper.name if wrapper is not None else 'block argument'}")
            base = wrapper.operands[0]
        return argument_values[base]

    expected_arguments = (
        Argument(0, "tensor", "f32"),   # recurrent state cache
        Argument(1, "tensor", "bf16"),  # packed q/k/v
        Argument(2, "tensor", "bf16"),  # decay gate input
        Argument(3, "tensor", "bf16"),  # beta gate input
        Argument(4, "tensor", "f32"),   # log decay coefficient
        Argument(5, "tensor", "f32"),   # decay bias
        Argument(6, "tensor", "i32"),   # state indices
        Argument(7, "tensor", "bf16"),  # output
        Argument(8, "scalar", "i32"),   # state slot count
    )
    if arguments != expected_arguments:
        raise GaudiLoweringError(
            "packed GDN decode ABI requires f32 state, bf16 packed QKV, "
            "bf16 gates, f32 decay parameters, i32 indices, bf16 output, "
            "and i32 slot count; "
            f"got {arguments!r}")

    program_ids = {
        operation.attributes.get("axis"): operation
        for operation in operations if operation.name == "tt.get_program_id"
    }
    if set(program_ids) != {0, 1, 2}:
        raise GaudiLoweringError(
            "packed GDN decode requires value-tile, value-head, and batch program IDs")

    ranges = [operation for operation in operations if operation.name == "tt.make_range"]
    if len(ranges) != 1 or ranges[0].attributes.get("start") != 0 or ranges[0].attributes.get("end") != 128:
        raise GaudiLoweringError("packed GDN decode requires one canonical K=128 vector range")

    # The canonical source starts each value tile at pid(0) * VALUE_TILE.
    tile_candidates: set[int] = set()
    value_pid = program_ids[0].results[0]
    for multiply in (operation for operation in operations if operation.name == "arith.muli"):
        if value_pid not in multiply.operands:
            continue
        other = multiply.operands[0] if multiply.operands[1] == value_pid else multiply.operands[1]
        constant = producers.get(other)
        if constant is not None and constant.name == "arith.constant":
            value = constant.attributes.get("value")
            if value in (16, 32, 64, 128):
                tile_candidates.add(value)
    if len(tile_candidates) != 1:
        raise GaudiLoweringError(
            "packed GDN decode VALUE_TILE must be exactly one of 16, 32, 64, or 128")
    value_tile = tile_candidates.pop()

    operation_names = {operation.name for operation in operations}
    required_operations = {
        "arith.addf",
        "arith.addi",
        "arith.cmpf",
        "arith.divsi",
        "arith.extf",
        "arith.mulf",
        "arith.muli",
        "arith.remsi",
        "arith.select",
        "arith.truncf",
        "math.exp",
        "math.log",
        "math.rsqrt",
        "tt.addptr",
        "tt.load",
        "tt.reduce",
        "tt.store",
    }
    missing = sorted(required_operations - operation_names)
    if missing:
        raise GaudiLoweringError(
            "packed GDN decode is missing canonical TTIR operations: " + ", ".join(missing))

    allowed_operations = {
        "builtin.module",
        "tt.func",
        "tt.return",
        "tt.reduce.return",
        "tt.get_program_id",
        "tt.make_range",
        "tt.splat",
        "tt.broadcast",
        "tt.expand_dims",
        "tt.addptr",
        "tt.load",
        "tt.store",
        "tt.reduce",
        "arith.constant",
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.negf",
        "arith.extf",
        "arith.truncf",
        "arith.addi",
        "arith.muli",
        "arith.divsi",
        "arith.remsi",
        "arith.cmpf",
        "arith.select",
        "arith.cmpi",
        "math.exp",
        "math.log",
        "math.rsqrt",
    }
    unsupported = sorted(operation_names - allowed_operations)
    if unsupported:
        raise GaudiLoweringError(
            "packed GDN decode contains unsupported TTIR operations: " + ", ".join(unsupported))

    counts: dict[str, int] = {}
    for operation in operations:
        counts[operation.name] = counts.get(operation.name, 0) + 1
    if counts.get("math.rsqrt") != 2:
        raise GaudiLoweringError(
            "packed GDN decode must contain canonical Q/K normalization; "
            f"got rsqrt={counts.get('math.rsqrt', 0)}")
    if (counts.get("math.exp") != 4 or counts.get("math.log") != 1 or
            counts.get("arith.cmpf") != 1 or counts.get("arith.select") != 1):
        raise GaudiLoweringError(
            "packed GDN decode must contain the canonical gate nonlinearities; "
            f"got exp={counts.get('math.exp', 0)}, log={counts.get('math.log', 0)}, "
            f"cmpf={counts.get('arith.cmpf', 0)}, select={counts.get('arith.select', 0)}")
    if counts.get("tt.reduce") != 2 + 2 * value_tile:
        raise GaudiLoweringError(
            "packed GDN decode must perform two norm reductions and two K reductions per value row")

    stores = [operation for operation in operations if operation.name == "tt.store"]
    if len(stores) != 2 * value_tile:
        raise GaudiLoweringError(
            "packed GDN decode must update every state row and emit every output row")
    stored_arguments: list[int] = []
    for store in stores:
        stored_arguments.append(pointer_argument(store.operands[0]))
    if stored_arguments.count(0) != value_tile or stored_arguments.count(7) != value_tile:
        raise GaudiLoweringError(
            "packed GDN decode stores must target only the mutable state cache and BF16 output")

    loaded_arguments: set[int] = set()
    for load in (operation for operation in operations if operation.name == "tt.load"):
        loaded_arguments.add(pointer_argument(load.operands[0]))
    if loaded_arguments != set(range(7)):
        raise GaudiLoweringError(
            "packed GDN decode must read state, packed QKV, gates, decay parameters, and indices")

    # Reject changes to constants that would silently alter the model rule.
    number_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
    canonical_f32 = {
        float(value)
        for value in (
            re.findall(rf"arith\.constant\s+({number_pattern})\s*:\s*f32", source) +
            re.findall(
                rf"arith\.constant\s+dense<({number_pattern})>\s*:\s*tensor<[^>]*f32>",
                source,
            )
        )
    }
    for required in (1.0, 20.0, 1.0e-6, 0.08838834764831845):
        if not any(abs(value - required) <= max(1.0e-12, abs(required) * 1.0e-6) for value in canonical_f32):
            raise GaudiLoweringError(
                f"packed GDN decode is missing canonical FP32 constant {required}; "
                f"got {sorted(canonical_f32)!r}")

    state_mapping = [
        {"tensor_dim": 0, "index_space_dim": 0, "a": 0, "start_b": 0, "end_b": 127,
         "all_required": True},
        {"tensor_dim": 1, "index_space_dim": 0, "a": value_tile, "start_b": 0,
         "end_b": value_tile - 1},
        {"tensor_dim": 2, "index_space_dim": 1, "a": 1, "start_b": 0, "end_b": 0},
        {"tensor_dim": 3, "index_space_dim": 2, "a": 0, "start_b": 0, "end_b": 0,
         "all_required": True},
    ]
    packed_mapping = [
        {"tensor_dim": 0, "index_space_dim": 1, "a": 0, "start_b": 0, "end_b": 10239,
         "all_required": True},
        {"tensor_dim": 1, "index_space_dim": 2, "a": 1, "start_b": 0, "end_b": 0},
    ]
    head_batch_mapping = [
        {"tensor_dim": 0, "index_space_dim": 1, "a": 1, "start_b": 0, "end_b": 0},
        {"tensor_dim": 1, "index_space_dim": 2, "a": 1, "start_b": 0, "end_b": 0},
    ]
    head_mapping = [{
        "tensor_dim": 0,
        "index_space_dim": 1,
        "a": 1,
        "start_b": 0,
        "end_b": 0,
    }]
    batch_mapping = [{
        "tensor_dim": 0,
        "index_space_dim": 2,
        "a": 1,
        "start_b": 0,
        "end_b": 0,
    }]
    output_mapping = [
        {"tensor_dim": 0, "index_space_dim": 0, "a": value_tile, "start_b": 0,
         "end_b": value_tile - 1},
        {"tensor_dim": 1, "index_space_dim": 1, "a": 1, "start_b": 0, "end_b": 0},
        {"tensor_dim": 2, "index_space_dim": 2, "a": 1, "start_b": 0, "end_b": 0},
    ]
    return GaudiProgram(
        name=name,
        source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        arguments=arguments,
        operations=operations,
        block_size=value_tile,
        vector_lanes=64,
        output_arg=7,
        input_args=(0, 1, 2, 3, 4, 5, 6),
        bound_arg=None,
        expression={
            "op": "gdn_decode_packed",
            "state_cache": 0,
            "packed_qkv": 1,
            "gate_a": 2,
            "gate_b": 3,
            "a_log": 4,
            "dt_bias": 5,
            "state_indices": 6,
            "output": 7,
            "state_slots": 8,
        },
        access_patterns=(
            {"arg": 0, "role": "mutable_input", "mapping": state_mapping},
            {"arg": 1, "role": "input", "mapping": packed_mapping},
            {"arg": 2, "role": "input", "mapping": head_batch_mapping},
            {"arg": 3, "role": "input", "mapping": head_batch_mapping},
            {"arg": 4, "role": "input", "mapping": head_mapping},
            {"arg": 5, "role": "input", "mapping": head_mapping},
            {"arg": 6, "role": "input", "mapping": batch_mapping},
            {"arg": 7, "role": "output", "mapping": output_mapping},
        ),
        index_space_rank=3,
        program_id_axes=(0, 1, 2),
        kind="gdn_decode_packed",
        parameters={
            "key_heads": 16,
            "value_heads": 48,
            "key_dim": 128,
            "value_dim": 128,
            "packed_width": 10240,
            "value_tile": value_tile,
            "state_slots_arg": 8,
            "mutates_arg": 0,
            "vlm_bytes": 0,
        },
    )


def _match_linear_offset(value: int, producers: dict[int, Operation]) -> tuple[int, int]:
    add = _producer(value, producers, "arith.addi")
    lhs, rhs = add.operands
    range_op = producers.get(lhs)
    base_value = rhs
    if range_op is None or range_op.name != "tt.make_range":
        range_op = producers.get(rhs)
        base_value = lhs
    if range_op is None or range_op.name != "tt.make_range":
        raise GaudiLoweringError("Gaudi FCD lowering requires program_id * BLOCK + arange offsets")
    if range_op.attributes.get("start") != 0:
        raise GaudiLoweringError("Gaudi FCD lowering currently requires tl.arange to start at zero")
    block_size = range_op.attributes.get("end")
    if not isinstance(block_size, int) or block_size <= 0:
        raise GaudiLoweringError("Gaudi FCD lowering requires a static positive block size")

    splat = _producer(base_value, producers, "tt.splat")
    multiply = _producer(splat.operands[0], producers, "arith.muli")
    left, right = multiply.operands
    pid = producers.get(left)
    constant_value = right
    if pid is None or pid.name != "tt.get_program_id":
        pid = producers.get(right)
        constant_value = left
    if pid is None or pid.name != "tt.get_program_id" or pid.attributes.get("axis") != 0:
        raise GaudiLoweringError("initial Gaudi FCD lowering supports tl.program_id(0) only")
    if _constant(constant_value, producers) != block_size:
        raise GaudiLoweringError("program_id stride must equal the static Triton block size")
    return block_size, value


def _trace_pointer(value: int, producers: dict[int, Operation], argument_values: dict[int, int]) -> tuple[int, int]:
    addptr = _producer(value, producers, "tt.addptr")
    if len(addptr.operands) != 2:
        raise GaudiLoweringError("malformed tt.addptr")
    return _argument_from_splat(addptr.operands[0], producers, argument_values), addptr.operands[1]


def _trace_mask(value: int, offset: int, producers: dict[int, Operation],
                argument_values: dict[int, int]) -> int:
    compare = _producer(value, producers, "arith.cmpi")
    # MLIR's signed-less-than predicate is 2.
    if compare.attributes.get("predicate") != 2 or compare.operands[0] != offset:
        raise GaudiLoweringError("only contiguous `offset < runtime_bound` masks are supported")
    return _argument_from_splat(compare.operands[1], producers, argument_values)


def _trace_expression(value: int, offset: int, mask: int | None, producers: dict[int, Operation],
                      argument_values: dict[int, int], inputs: list[int]) -> dict[str, Any]:
    op = _producer(value, producers)
    if op.name == "tt.load":
        tensor_arg, load_offset = _trace_pointer(op.operands[0], producers, argument_values)
        if load_offset != offset:
            raise GaudiLoweringError("all initial Gaudi elementwise loads must use the store's contiguous offset")
        if mask is not None and (len(op.operands) < 2 or op.operands[1] != mask):
            raise GaudiLoweringError("load and store masks must match in the initial Gaudi fast path")
        if tensor_arg not in inputs:
            inputs.append(tensor_arg)
        return {"op": "load", "arg": tensor_arg}
    if op.name in _ELEMENTWISE_OPS:
        return {
            "op": _ELEMENTWISE_OPS[op.name],
            "lhs": _trace_expression(op.operands[0], offset, mask, producers, argument_values, inputs),
            "rhs": _trace_expression(op.operands[1], offset, mask, producers, argument_values, inputs),
        }
    raise GaudiLoweringError(
        f"TTIR operation `{op.name}` is not supported by the initial Gaudi elementwise lowering")


def lower_ttir(module) -> GaudiProgram:
    name = module.get_entry_func_name()
    source = module.str_nodebug()
    arguments, argument_values = _parse_arguments(module)
    operations, producers = _collect_operations(module)

    qk_conv_program = _match_gdn_qk_conv_packed(
        name,
        source,
        arguments,
        argument_values,
        operations,
        producers,
    )
    if qk_conv_program is not None:
        return qk_conv_program

    value_conv_program = _match_gdn_decode_value_conv_packed(
        name,
        source,
        arguments,
        argument_values,
        operations,
        producers,
    )
    if value_conv_program is not None:
        return value_conv_program

    fused_gdn_program = _match_gdn_decode_conv_packed(
        name,
        source,
        arguments,
        argument_values,
        operations,
        producers,
    )
    if fused_gdn_program is not None:
        return fused_gdn_program

    gdn_program = _match_gdn_decode_packed(
        name,
        source,
        arguments,
        argument_values,
        operations,
        producers,
    )
    if gdn_program is not None:
        return gdn_program

    dynamic_quant_program = _match_dynamic_quant(
        name,
        source,
        arguments,
        argument_values,
        operations,
        producers,
    )
    if dynamic_quant_program is not None:
        return dynamic_quant_program

    fused_program = _match_fused_add_rms_norm(
        name,
        source,
        arguments,
        argument_values,
        operations,
        producers,
    )
    if fused_program is not None:
        return fused_program

    silu_program = _match_silu_and_mul(
        name,
        source,
        arguments,
        argument_values,
        operations,
        producers,
    )
    if silu_program is not None:
        return silu_program

    unsupported = sorted({
        op.name
        for op in operations
        if op.name not in _IGNORED_OPS and op.name not in _STRUCTURAL_OPS and op.name not in _ELEMENTWISE_OPS
    })
    if unsupported:
        raise GaudiLoweringError("unsupported TTIR operations for Gaudi2 strict mode: " + ", ".join(unsupported))

    stores = [op for op in operations if op.name == "tt.store"]
    if len(stores) != 1:
        raise GaudiLoweringError("initial Gaudi elementwise lowering requires exactly one tt.store")
    store = stores[0]
    if len(store.operands) not in (2, 3):
        raise GaudiLoweringError("malformed tt.store")
    output_arg, offset = _trace_pointer(store.operands[0], producers, argument_values)
    block_size, canonical_offset = _match_linear_offset(offset, producers)
    if canonical_offset != offset:
        raise AssertionError("offset canonicalization invariant failed")
    mask_value = store.operands[2] if len(store.operands) == 3 else None
    bound_arg = _trace_mask(mask_value, offset, producers, argument_values) if mask_value is not None else None

    output = arguments[output_arg]
    if output.kind != "tensor" or output.dtype not in ("f32", "bf16"):
        raise GaudiLoweringError("initial Gaudi TPC codegen supports FP32 and BF16 tensors only")
    inputs: list[int] = []
    expression = _trace_expression(store.operands[1], offset, mask_value, producers, argument_values, inputs)
    for input_arg in inputs:
        if arguments[input_arg].kind != "tensor" or arguments[input_arg].dtype != output.dtype:
            raise GaudiLoweringError("all initial Gaudi elementwise tensors must have the same dtype")

    tensor_arguments = [arg.index for arg in arguments if arg.kind == "tensor"]
    if tensor_arguments != [*inputs, output_arg]:
        raise GaudiLoweringError(
            "initial Gaudi TPC ABI requires tensor parameters to be ordered as inputs followed by output")
    scalar_arguments = [arg.index for arg in arguments if arg.kind == "scalar"]
    if scalar_arguments and min(scalar_arguments) < max(tensor_arguments):
        raise GaudiLoweringError("initial Gaudi TPC ABI requires scalar parameters after all tensor parameters")

    vector_lanes = 2048 // (32 if output.dtype == "f32" else 16)
    touched = [*inputs, output_arg]
    access_patterns = tuple({
        "arg": arg,
        "role": "output" if arg == output_arg else "input",
        "mapping": [{
            "tensor_dim": 0,
            "index_space_dim": 0,
            "a": block_size,
            "start_b": 0,
            "end_b": block_size - 1,
        }],
    } for arg in touched)
    return GaudiProgram(
        name=name,
        source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        arguments=arguments,
        operations=operations,
        block_size=block_size,
        vector_lanes=vector_lanes,
        output_arg=output_arg,
        input_args=tuple(inputs),
        bound_arg=bound_arg,
        expression=expression,
        access_patterns=access_patterns,
    )


def _emit_expression(
    expression: dict[str, Any],
    lines: list[str],
    next_temp: list[int],
    vector_type: str,
    intrinsic_prefix: str,
) -> str:
    name = f"value_{next_temp[0]}"
    next_temp[0] += 1
    operation = expression["op"]
    if operation == "load":
        lines.extend([
            f"        {vector_type} {name};",
            "        if (remaining >= VECTOR_LANES)",
            f"            {name} = v_{intrinsic_prefix}_ld_tnsr_b(coords, arg{expression['arg']});",
            "        else",
            f"            {name} = v_{intrinsic_prefix}_ld_tnsr_partial_b(",
            f"                coords, arg{expression['arg']}, remaining - 1, 0);",
        ])
        return name
    lhs = _emit_expression(expression["lhs"], lines, next_temp, vector_type, intrinsic_prefix)
    rhs = _emit_expression(expression["rhs"], lines, next_temp, vector_type, intrinsic_prefix)
    intrinsic = f"v_{intrinsic_prefix}_{operation}_b"
    lines.append(f"        {vector_type} {name} = {intrinsic}({lhs}, {rhs});")
    return name


def _emit_dynamic_quant_tpc_c(program: GaudiProgram) -> TpcCSource:
    if program.input_args != (0,) or program.output_args != (1, 2):
        raise GaudiLoweringError("TPC-C dynamic quantization requires one input and FP8/scale outputs")
    expected = (
        ("tensor", "bf16"),
        ("tensor", "fp8e4nv"),
        ("tensor", "f32"),
    )
    if tuple((argument.kind, argument.dtype) for argument in program.arguments) != expected:
        raise GaudiLoweringError("TPC-C dynamic quantization requires the canonical mixed tensor ABI")
    n_cols = int((program.parameters or {}).get("n_cols", 0))
    if n_cols <= 0 or n_cols > program.block_size or program.block_size > 16384:
        raise GaudiLoweringError("TPC-C dynamic quantization has invalid constexpr n_cols metadata")

    source = f"""// Generated by Triton Gaudi2 backend. Do not edit.
#define TRITON_GAUDI_BLOCK_SIZE {program.block_size}
#define TRITON_GAUDI_N_COLS {n_cols}
#define TRITON_GAUDI_BF16_LANES 128
#define TRITON_GAUDI_FP8_LANES 256
void main(tensor arg0, tensor arg1, tensor arg2)
{{
    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;
    const int n_cols = TRITON_GAUDI_N_COLS;
    const int full_columns = n_cols - n_cols % TRITON_GAUDI_BF16_LANES;
    const int tail_columns = n_cols - full_columns;
    const uchar256 broadcast_lane_zero = 0x80;

    for (int row = index_space_start[0]; row < index_space_end[0]; ++row)
    {{
        float128 maximum = {{0}};
        for (int lane = 0;
             lane < full_columns;
             lane += TRITON_GAUDI_BF16_LANES)
        {{
            int5 coords = {{row * n_cols + lane, 0, 0, 0, 0}};
            const bfloat128 values = v_bf16_ld_tnsr_b(coords, arg0);
            const float128 values_f32 =
                convert_bfloat128_to_float128(values, SW_LINEAR);
            maximum.v1 = v_f32_max_b(
                maximum.v1, v_f32_abs_b(values_f32.v1));
            maximum.v2 = v_f32_max_b(
                maximum.v2, v_f32_abs_b(values_f32.v2));
        }}
        if (tail_columns > 0)
        {{
            int5 coords = {{row * n_cols + full_columns, 0, 0, 0, 0}};
            const bfloat128 values = v_bf16_ld_tnsr_partial_b(
                coords, arg0, tail_columns - 1, 0);
            const float128 values_f32 =
                convert_bfloat128_to_float128(values, SW_LINEAR);
            maximum.v1 = v_f32_max_b(
                maximum.v1, v_f32_abs_b(values_f32.v1));
            maximum.v2 = v_f32_max_b(
                maximum.v2, v_f32_abs_b(values_f32.v2));
        }}

        float64 row_maximum = v_f32_max_b(maximum.v1, maximum.v2);
        row_maximum = v_f32_reduce_max(row_maximum);
        row_maximum = v_f32_shuffle_b(
            row_maximum, broadcast_lane_zero, 0, row_maximum);
        const float64 scale = (row_maximum + 1.0e-8f) / 240.0f;
        const float64 inverse_scale = 1.0f / scale;
        int5 scale_coords = {{row, 0, 0, 0, 0}};
        v_f32_st_tnsr_partial(scale_coords, arg2, scale, 0, 0);

        for (int lane = 0;
             lane < n_cols;
             lane += TRITON_GAUDI_FP8_LANES)
        {{
            int5 first_coords = {{row * n_cols + lane, 0, 0, 0, 0}};
            const int first_remaining = n_cols - lane;
            bfloat128 first = {{0}};
            if (first_remaining >= TRITON_GAUDI_BF16_LANES)
            {{
                first = v_bf16_ld_tnsr_b(first_coords, arg0);
            }}
            else
            {{
                first = v_bf16_ld_tnsr_partial_b(
                    first_coords, arg0, first_remaining - 1, 0);
            }}
            bfloat128 second = {{0}};
            const int second_start = lane + TRITON_GAUDI_BF16_LANES;
            if (second_start < n_cols)
            {{
                int5 second_coords = {{
                    row * n_cols + second_start, 0, 0, 0, 0}};
                const int second_remaining = n_cols - second_start;
                if (second_remaining >= TRITON_GAUDI_BF16_LANES)
                {{
                    second = v_bf16_ld_tnsr_b(second_coords, arg0);
                }}
                else
                {{
                    second = v_bf16_ld_tnsr_partial_b(
                        second_coords, arg0, second_remaining - 1, 0);
                }}
            }}
            const float128 first_f32 =
                convert_bfloat128_to_float128(first, SW_LINEAR);
            const float128 second_f32 =
                convert_bfloat128_to_float128(second, SW_LINEAR);
            float256 scaled = {{0}};
            scaled.v1 = first_f32.v1 * inverse_scale;
            scaled.v2 = first_f32.v2 * inverse_scale;
            scaled.v3 = second_f32.v1 * inverse_scale;
            scaled.v4 = second_f32.v2 * inverse_scale;
            const minifloat256 quantized =
                v_convert_f32_to_f8_all_b(
                    scaled, SW_RHNE | SW_FP8_BIAS7 | SW_LINEAR);
            int5 output_coords = {{row * n_cols + lane, 0, 0, 0, 0}};
            const int remaining = n_cols - lane;
            if (remaining >= TRITON_GAUDI_FP8_LANES)
            {{
                v_f8_st_tnsr(output_coords, arg1, quantized);
            }}
            else
            {{
                v_f8_st_tnsr_partial(
                    output_coords, arg1, quantized, remaining - 1, 0);
            }}
        }}
    }}
}}
"""
    return TpcCSource(program, source)


def _emit_fused_add_rms_norm_tpc_c(program: GaudiProgram) -> TpcCSource:
    if program.tensor_dtype != "bf16" or program.input_args != (0, 1, 2) or program.output_args != (3, 4):
        raise GaudiLoweringError("TPC-C fused add+RMSNorm requires the canonical BF16 tensor ABI")
    if tuple((argument.kind, argument.dtype) for argument in program.arguments[5:]) != (("scalar", "f32"),):
        raise GaudiLoweringError("TPC-C fused add+RMSNorm requires a runtime f32 epsilon")
    n_cols = int((program.parameters or {}).get("n_cols", 0))
    if n_cols <= 0 or n_cols > program.block_size:
        raise GaudiLoweringError("TPC-C fused add+RMSNorm has invalid constexpr n_cols metadata")

    source = f"""// Generated by Triton Gaudi2 backend. Do not edit.
#define TRITON_GAUDI_BLOCK_SIZE {program.block_size}
#define TRITON_GAUDI_VECTOR_LANES 128
#define TRITON_GAUDI_N_COLS {n_cols}
#define TRITON_GAUDI_CHUNK_COUNT \\
    ((TRITON_GAUDI_BLOCK_SIZE + TRITON_GAUDI_VECTOR_LANES - 1) / TRITON_GAUDI_VECTOR_LANES)

__local__ bfloat128 triton_gaudi_summed[TRITON_GAUDI_CHUNK_COUNT];

void main(
    tensor arg0,
    tensor arg1,
    tensor arg2,
    tensor arg3,
    tensor arg4,
    float arg5)
{{
    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;
    const int n_cols = TRITON_GAUDI_N_COLS;
    const int full_columns =
        n_cols - n_cols % TRITON_GAUDI_VECTOR_LANES;
    const int tail_columns = n_cols - full_columns;
    const uchar256 broadcast_lane_zero = 0x80;

    for (int row = index_space_start[0]; row < index_space_end[0]; ++row)
    {{
        float128 squared_accumulator = {{0}};
        for (int lane = 0;
             lane < full_columns;
             lane += TRITON_GAUDI_VECTOR_LANES)
        {{
            int5 coords = {{row * n_cols + lane, 0, 0, 0, 0}};
            const bfloat128 hidden_values = v_bf16_ld_tnsr_b(coords, arg0);
            const bfloat128 residual_values = v_bf16_ld_tnsr_b(coords, arg1);
            const bfloat128 summed = v_bf16_add_b(hidden_values, residual_values);
            triton_gaudi_summed[lane / TRITON_GAUDI_VECTOR_LANES] = summed;
            v_bf16_st_tnsr(coords, arg4, summed);
            squared_accumulator = v_bf16_mac_acc32_b(
                summed,
                summed,
                squared_accumulator,
                (e_no_negation) << 1);
        }}
        if (tail_columns > 0)
        {{
            int5 coords = {{row * n_cols + full_columns, 0, 0, 0, 0}};
            const bfloat128 hidden_values = v_bf16_ld_tnsr_partial_b(
                coords, arg0, tail_columns - 1, 0);
            const bfloat128 residual_values = v_bf16_ld_tnsr_partial_b(
                coords, arg1, tail_columns - 1, 0);
            const bfloat128 summed = v_bf16_add_b(hidden_values, residual_values);
            triton_gaudi_summed[full_columns / TRITON_GAUDI_VECTOR_LANES] = summed;
            v_bf16_st_tnsr_partial(
                coords, arg4, summed, tail_columns - 1, 0);
            squared_accumulator = v_bf16_mac_acc32_b(
                summed,
                summed,
                squared_accumulator,
                (e_no_negation) << 1);
        }}

        float64 total_sumsq = squared_accumulator.v1 + squared_accumulator.v2;
        total_sumsq = v_f32_reduce_add(total_sumsq);
        total_sumsq = v_f32_shuffle_b(
        total_sumsq, broadcast_lane_zero, 0, total_sumsq);
        const float64 reciprocal_rms =
            v_rsqrt_f32(total_sumsq / (float)n_cols + arg5);
        for (int lane = 0;
             lane < full_columns;
             lane += TRITON_GAUDI_VECTOR_LANES)
        {{
            int5 output_coords = {{row * n_cols + lane, 0, 0, 0, 0}};
            int5 weight_coords = {{lane, 0, 0, 0, 0}};
            const bfloat128 weight_values =
                v_bf16_ld_tnsr_b(weight_coords, arg2);
            const float128 summed_f32 =
                convert_bfloat128_to_float128(
                    triton_gaudi_summed[lane / TRITON_GAUDI_VECTOR_LANES],
                    SW_LINEAR);
            const float128 weight_f32 =
                convert_bfloat128_to_float128(weight_values, SW_LINEAR);
            float128 normalized = {{0}};
            normalized.v1 = summed_f32.v1 * reciprocal_rms * weight_f32.v1;
            normalized.v2 = summed_f32.v2 * reciprocal_rms * weight_f32.v2;
            const bfloat128 result = convert_float128_to_bfloat128(
                normalized, SW_RHNE | SW_LINEAR);
            v_bf16_st_tnsr(output_coords, arg3, result);
        }}
        if (tail_columns > 0)
        {{
            int5 output_coords = {{row * n_cols + full_columns, 0, 0, 0, 0}};
            int5 weight_coords = {{full_columns, 0, 0, 0, 0}};
            const bfloat128 weight_values = v_bf16_ld_tnsr_partial_b(
                weight_coords, arg2, tail_columns - 1, 0);
            const float128 summed_f32 =
                convert_bfloat128_to_float128(
                    triton_gaudi_summed[
                        full_columns / TRITON_GAUDI_VECTOR_LANES],
                    SW_LINEAR);
            const float128 weight_f32 =
                convert_bfloat128_to_float128(weight_values, SW_LINEAR);
            float128 normalized = {{0}};
            normalized.v1 = summed_f32.v1 * reciprocal_rms * weight_f32.v1;
            normalized.v2 = summed_f32.v2 * reciprocal_rms * weight_f32.v2;
            const bfloat128 result = convert_float128_to_bfloat128(
                normalized, SW_RHNE | SW_LINEAR);
            v_bf16_st_tnsr_partial(
                output_coords, arg3, result, tail_columns - 1, 0);
        }}
    }}
}}
"""
    return TpcCSource(program, source)


def _emit_silu_and_mul_tpc_c(program: GaudiProgram) -> TpcCSource:
    if program.tensor_dtype != "bf16" or program.input_args != (0,) or program.output_args != (1,):
        raise GaudiLoweringError("TPC-C SiLU-and-mul requires the canonical BF16 tensor ABI")
    if any(argument.kind != "tensor" for argument in program.arguments):
        raise GaudiLoweringError("TPC-C SiLU-and-mul does not accept runtime scalar arguments")
    n_cols = int((program.parameters or {}).get("n_cols", 0))
    input_row_stride = int((program.parameters or {}).get("input_row_stride", 0))
    chunks_per_row = int((program.parameters or {}).get("chunks_per_row", 0))
    if (n_cols <= 0 or input_row_stride != 2 * n_cols or chunks_per_row <= 0 or
            chunks_per_row != (n_cols + program.block_size - 1) // program.block_size or
            program.vector_lanes != 128 or program.index_space_rank != 2):
        raise GaudiLoweringError("TPC-C SiLU-and-mul has invalid constexpr row metadata")

    source = f"""// Generated by Triton Gaudi2 backend. Do not edit.
#define TRITON_GAUDI_VECTOR_LANES 128
#define TRITON_GAUDI_N_COLS {n_cols}
#define TRITON_GAUDI_CHUNK_SIZE {program.block_size}
#define TRITON_GAUDI_CHUNKS_PER_ROW {chunks_per_row}

void main(tensor arg0, tensor arg1)
{{
    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;
    const int n_cols = TRITON_GAUDI_N_COLS;
    for (int row = index_space_start[1]; row < index_space_end[1]; ++row)
    {{
        for (int chunk = index_space_start[0];
             chunk < index_space_end[0];
             ++chunk)
        {{
            const int chunk_base = chunk * TRITON_GAUDI_CHUNK_SIZE;
            for (int lane = 0;
                 lane < TRITON_GAUDI_CHUNK_SIZE;
                 lane += TRITON_GAUDI_VECTOR_LANES)
            {{
                const int column = chunk_base + lane;
                const int remaining = n_cols - column;
                if (remaining <= 0)
                    break;
                int5 gate_coords = {{column, row, 0, 0, 0}};
                int5 up_coords = {{n_cols + column, row, 0, 0, 0}};
                int5 output_coords = {{column, row, 0, 0, 0}};
                bfloat128 gate_values;
                bfloat128 up_values;
                if (remaining >= TRITON_GAUDI_VECTOR_LANES)
                {{
                    gate_values = v_bf16_ld_tnsr_b(gate_coords, arg0);
                    up_values = v_bf16_ld_tnsr_b(up_coords, arg0);
                }}
                else
                {{
                    gate_values = v_bf16_ld_tnsr_partial_b(
                        gate_coords, arg0, remaining - 1, 0);
                    up_values = v_bf16_ld_tnsr_partial_b(
                        up_coords, arg0, remaining - 1, 0);
                }}
                const float128 gate_f32 =
                    convert_bfloat128_to_float128(gate_values, SW_LINEAR);
                const float128 up_f32 =
                    convert_bfloat128_to_float128(up_values, SW_LINEAR);
                float128 result_f32 = {{0}};
                result_f32.v1 =
                    gate_f32.v1 * v_sigmoid_f32(gate_f32.v1) * up_f32.v1;
                result_f32.v2 =
                    gate_f32.v2 * v_sigmoid_f32(gate_f32.v2) * up_f32.v2;
                const bfloat128 result = convert_float128_to_bfloat128(
                    result_f32, SW_RHNE | SW_LINEAR);
                if (remaining >= TRITON_GAUDI_VECTOR_LANES)
                    v_bf16_st_tnsr(output_coords, arg1, result);
                else
                    v_bf16_st_tnsr_partial(
                        output_coords, arg1, result, remaining - 1, 0);
            }}
        }}
    }}
}}
"""
    return TpcCSource(program, source)


def _emit_gdn_decode_packed_tpc_c(program: GaudiProgram) -> TpcCSource:
    expected = (
        ("tensor", "f32"),
        ("tensor", "bf16"),
        ("tensor", "bf16"),
        ("tensor", "bf16"),
        ("tensor", "f32"),
        ("tensor", "f32"),
        ("tensor", "i32"),
        ("tensor", "bf16"),
        ("scalar", "i32"),
    )
    if tuple((argument.kind, argument.dtype) for argument in program.arguments) != expected:
        raise GaudiLoweringError("TPC-C packed GDN decode requires the canonical mixed-dtype ABI")
    parameters = program.parameters or {}
    value_tile = int(parameters.get("value_tile", 0))
    if (value_tile not in (16, 32, 64, 128) or program.block_size != value_tile or
            program.index_space_rank != 3 or program.vector_lanes != 64):
        raise GaudiLoweringError("TPC-C packed GDN decode has invalid value-tile metadata")

    source = f"""// Generated by Triton Gaudi2 backend. Do not edit.
#define TRITON_GAUDI_GDN_KEY_HEADS 16
#define TRITON_GAUDI_GDN_VALUE_HEADS 48
#define TRITON_GAUDI_GDN_KEY_DIM 128
#define TRITON_GAUDI_GDN_VALUE_DIM 128
#define TRITON_GAUDI_GDN_VALUE_TILE {value_tile}

void main(
    tensor arg0,
    tensor arg1,
    tensor arg2,
    tensor arg3,
    tensor arg4,
    tensor arg5,
    tensor arg6,
    tensor arg7,
    int arg8)
{{
    const uchar256 broadcast_lane_zero = 0x80;
    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;
    const int key_offset =
        TRITON_GAUDI_GDN_KEY_HEADS * TRITON_GAUDI_GDN_KEY_DIM;
    const int value_offset = 2 * key_offset;

    for (int batch = index_space_start[2]; batch < index_space_end[2]; ++batch)
    {{
        int5 index_coords = {{batch, 0, 0, 0, 0}};
        const int raw_slot = s_i32_ld_g(gen_addr(index_coords, arg6));
        const int state_slot = ((raw_slot % arg8) + arg8) % arg8;

        for (int value_head = index_space_start[1];
             value_head < index_space_end[1];
             ++value_head)
        {{
            const int key_head = value_head / 3;
            int5 packed_coords = {{
                key_head * TRITON_GAUDI_GDN_KEY_DIM, batch, 0, 0, 0}};
            const bfloat128 q_bf16 = v_bf16_ld_tnsr_b(packed_coords, arg1);
            packed_coords[0] = key_offset + key_head * TRITON_GAUDI_GDN_KEY_DIM;
            const bfloat128 k_bf16 = v_bf16_ld_tnsr_b(packed_coords, arg1);
            float128 q = convert_bfloat128_to_float128(q_bf16, SW_LINEAR);
            float128 k = convert_bfloat128_to_float128(k_bf16, SW_LINEAR);

            float64 q_norm = v_f32_mac_b(q.v1, q.v1, q.v2 * q.v2);
            q_norm = v_f32_reduce_add(q_norm);
            q_norm = v_f32_shuffle_b(q_norm, broadcast_lane_zero, 0, q_norm);
            q_norm = v_rsqrt_f32(q_norm + 1.0e-6f) * 0.08838834764831845f;
            q.v1 = q.v1 * q_norm;
            q.v2 = q.v2 * q_norm;

            float64 k_norm = v_f32_mac_b(k.v1, k.v1, k.v2 * k.v2);
            k_norm = v_f32_reduce_add(k_norm);
            k_norm = v_f32_shuffle_b(k_norm, broadcast_lane_zero, 0, k_norm);
            k_norm = v_rsqrt_f32(k_norm + 1.0e-6f);
            k.v1 = k.v1 * k_norm;
            k.v2 = k.v2 * k_norm;

            int5 gate_coords = {{value_head, batch, 0, 0, 0}};
            int5 head_coords = {{value_head, 0, 0, 0, 0}};
            const float gate_x =
                s_convert_bf16_to_f32(s_bf16_ld_g(gen_addr(gate_coords, arg2))) +
                s_f32_ld_g(gen_addr(head_coords, arg5));
            float64 gate_x_vector = gate_x;
            float64 softplus = gate_x_vector;
            if (gate_x <= 20.0f)
                softplus = v_log_f32(v_exp_f32(gate_x_vector) + 1.0f);
            const float64 a_log = s_f32_ld_g(gen_addr(head_coords, arg4));
            const float64 decay = v_exp_f32(-v_exp_f32(a_log) * softplus);
            const float64 beta = v_sigmoid_f32(
                s_convert_bf16_to_f32(
                    s_bf16_ld_g(gen_addr(gate_coords, arg3))));

            const int value_start =
                index_space_start[0] * TRITON_GAUDI_GDN_VALUE_TILE;
            const int value_end =
                index_space_end[0] * TRITON_GAUDI_GDN_VALUE_TILE;
#pragma loop_unroll(2)
            for (int value_row = value_start; value_row < value_end; ++value_row)
            {{
                int5 state_coords = {{0, value_row, value_head, state_slot, 0}};
                float64 state_lo = v_f32_ld_tnsr_b(state_coords, arg0) * decay;
                state_coords[0] = 64;
                float64 state_hi = v_f32_ld_tnsr_b(state_coords, arg0) * decay;

                int5 value_coords = {{
                    value_offset + value_head * TRITON_GAUDI_GDN_VALUE_DIM + value_row,
                    batch,
                    0,
                    0,
                    0}};
                const float value = s_convert_bf16_to_f32(
                    s_bf16_ld_g(gen_addr(value_coords, arg1)));

                float64 projection = state_lo * k.v1;
                projection = v_f32_mac_b(state_hi, k.v2, projection);
                projection = v_f32_reduce_add(projection);
                projection = v_f32_shuffle_b(
                    projection, broadcast_lane_zero, 0, projection);
                const float64 delta = (value - projection) * beta;

                state_lo = v_f32_mac_b(k.v1, delta, state_lo);
                state_hi = v_f32_mac_b(k.v2, delta, state_hi);

                float64 result = state_lo * q.v1;
                result = v_f32_mac_b(state_hi, q.v2, result);
                result = v_f32_reduce_add(result);

                state_coords[0] = 0;
                v_f32_st_tnsr(state_coords, arg0, state_lo);
                state_coords[0] = 64;
                v_f32_st_tnsr(state_coords, arg0, state_hi);
                int5 output_coords = {{value_row, value_head, batch, 0, 0}};
                float128 result_f32 = {{0}};
                result_f32.v1 = result;
                const bfloat128 result_bf16 = convert_float128_to_bfloat128(
                    result_f32, SW_RHNE | SW_LINEAR);
                v_bf16_st_tnsr_partial(
                    output_coords, arg7, result_bf16, 0, 0);
            }}
        }}
    }}
}}
"""
    return TpcCSource(program, source)


def _emit_gdn_decode_conv_packed_tpc_c(program: GaudiProgram) -> TpcCSource:
    expected = (
        ("tensor", "bf16"),
        ("tensor", "f32"),
        ("tensor", "bf16"),
        ("tensor", "bf16"),
        ("tensor", "bf16"),
        ("tensor", "f32"),
        ("tensor", "f32"),
        ("tensor", "i32"),
        ("tensor", "bf16"),
        ("tensor", "bf16"),
        ("scalar", "i32"),
        ("scalar", "i32"),
    )
    if tuple((argument.kind, argument.dtype) for argument in program.arguments) != expected:
        raise GaudiLoweringError(
            "TPC-C fused conv+GDN requires the canonical mixed-dtype ABI")
    parameters = program.parameters or {}
    if (program.block_size != 128 or program.index_space_rank != 2 or
            program.vector_lanes != 64 or parameters.get("conv_width") != 4 or
            parameters.get("mutates_args") != [0, 1]):
        raise GaudiLoweringError(
            "TPC-C fused conv+GDN has invalid ownership or convolution metadata")

    source = """// Generated by Triton Gaudi2 backend. Do not edit.
#define TRITON_GAUDI_GDN_KEY_HEADS 16
#define TRITON_GAUDI_GDN_VALUE_HEADS 48
#define TRITON_GAUDI_GDN_KEY_DIM 128
#define TRITON_GAUDI_GDN_VALUE_DIM 128
#define TRITON_GAUDI_GDN_PACKED_WIDTH 10240
#define TRITON_GAUDI_GDN_CONV_WIDTH 4

void main(
    tensor arg0,
    tensor arg1,
    tensor arg2,
    tensor arg3,
    tensor arg4,
    tensor arg5,
    tensor arg6,
    tensor arg7,
    tensor arg8,
    tensor arg9,
    int arg10,
    int arg11)
{
    const uchar256 broadcast_lane_zero = 0x80;
    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;
    const int key_offset =
        TRITON_GAUDI_GDN_KEY_HEADS * TRITON_GAUDI_GDN_KEY_DIM;
    const int value_offset = 2 * key_offset;

    for (int batch = index_space_start[1]; batch < index_space_end[1]; ++batch)
    {
        int5 index_coords = {batch, 0, 0, 0, 0};
        const int raw_slot = s_i32_ld_g(gen_addr(index_coords, arg7));
        const int conv_slot = ((raw_slot % arg10) + arg10) % arg10;
        const int state_slot = ((raw_slot % arg11) + arg11) % arg11;

        for (int key_head = index_space_start[0];
             key_head < index_space_end[0];
             ++key_head)
        {
            const int q_channel = key_head * TRITON_GAUDI_GDN_KEY_DIM;
            const int k_channel = key_offset + q_channel;
            int5 q_raw_coords = {q_channel, batch, 0, 0, 0};
            int5 k_raw_coords = {k_channel, batch, 0, 0, 0};
            const bfloat128 q_raw = v_bf16_ld_tnsr_b(q_raw_coords, arg2);
            const bfloat128 k_raw = v_bf16_ld_tnsr_b(k_raw_coords, arg2);

            int5 q_state0_coords = {q_channel, 0, conv_slot, 0, 0};
            int5 q_state1_coords = {q_channel, 1, conv_slot, 0, 0};
            int5 q_state2_coords = {q_channel, 2, conv_slot, 0, 0};
            int5 k_state0_coords = {k_channel, 0, conv_slot, 0, 0};
            int5 k_state1_coords = {k_channel, 1, conv_slot, 0, 0};
            int5 k_state2_coords = {k_channel, 2, conv_slot, 0, 0};
            const bfloat128 q_h0 = v_bf16_ld_tnsr_b(q_state0_coords, arg0);
            const bfloat128 q_h1 = v_bf16_ld_tnsr_b(q_state1_coords, arg0);
            const bfloat128 q_h2 = v_bf16_ld_tnsr_b(q_state2_coords, arg0);
            const bfloat128 k_h0 = v_bf16_ld_tnsr_b(k_state0_coords, arg0);
            const bfloat128 k_h1 = v_bf16_ld_tnsr_b(k_state1_coords, arg0);
            const bfloat128 k_h2 = v_bf16_ld_tnsr_b(k_state2_coords, arg0);

            int5 q_weight0_coords = {q_channel, 0, 0, 0, 0};
            int5 q_weight1_coords = {q_channel, 1, 0, 0, 0};
            int5 q_weight2_coords = {q_channel, 2, 0, 0, 0};
            int5 q_weight3_coords = {q_channel, 3, 0, 0, 0};
            int5 k_weight0_coords = {k_channel, 0, 0, 0, 0};
            int5 k_weight1_coords = {k_channel, 1, 0, 0, 0};
            int5 k_weight2_coords = {k_channel, 2, 0, 0, 0};
            int5 k_weight3_coords = {k_channel, 3, 0, 0, 0};
            const bfloat128 q_w0 = v_bf16_ld_tnsr_b(q_weight0_coords, arg8);
            const bfloat128 q_w1 = v_bf16_ld_tnsr_b(q_weight1_coords, arg8);
            const bfloat128 q_w2 = v_bf16_ld_tnsr_b(q_weight2_coords, arg8);
            const bfloat128 q_w3 = v_bf16_ld_tnsr_b(q_weight3_coords, arg8);
            const bfloat128 k_w0 = v_bf16_ld_tnsr_b(k_weight0_coords, arg8);
            const bfloat128 k_w1 = v_bf16_ld_tnsr_b(k_weight1_coords, arg8);
            const bfloat128 k_w2 = v_bf16_ld_tnsr_b(k_weight2_coords, arg8);
            const bfloat128 k_w3 = v_bf16_ld_tnsr_b(k_weight3_coords, arg8);

            const float128 q_h0_f32 =
                convert_bfloat128_to_float128(q_h0, SW_LINEAR);
            const float128 q_h1_f32 =
                convert_bfloat128_to_float128(q_h1, SW_LINEAR);
            const float128 q_h2_f32 =
                convert_bfloat128_to_float128(q_h2, SW_LINEAR);
            const float128 q_raw_f32 =
                convert_bfloat128_to_float128(q_raw, SW_LINEAR);
            const float128 q_w0_f32 =
                convert_bfloat128_to_float128(q_w0, SW_LINEAR);
            const float128 q_w1_f32 =
                convert_bfloat128_to_float128(q_w1, SW_LINEAR);
            const float128 q_w2_f32 =
                convert_bfloat128_to_float128(q_w2, SW_LINEAR);
            const float128 q_w3_f32 =
                convert_bfloat128_to_float128(q_w3, SW_LINEAR);
            float128 q_conv = {0};
            q_conv.v1 = q_h0_f32.v1 * q_w0_f32.v1;
            q_conv.v1 += q_h1_f32.v1 * q_w1_f32.v1;
            q_conv.v1 += q_h2_f32.v1 * q_w2_f32.v1;
            q_conv.v1 += q_raw_f32.v1 * q_w3_f32.v1;
            q_conv.v2 = q_h0_f32.v2 * q_w0_f32.v2;
            q_conv.v2 += q_h1_f32.v2 * q_w1_f32.v2;
            q_conv.v2 += q_h2_f32.v2 * q_w2_f32.v2;
            q_conv.v2 += q_raw_f32.v2 * q_w3_f32.v2;

            const float128 k_h0_f32 =
                convert_bfloat128_to_float128(k_h0, SW_LINEAR);
            const float128 k_h1_f32 =
                convert_bfloat128_to_float128(k_h1, SW_LINEAR);
            const float128 k_h2_f32 =
                convert_bfloat128_to_float128(k_h2, SW_LINEAR);
            const float128 k_raw_f32 =
                convert_bfloat128_to_float128(k_raw, SW_LINEAR);
            const float128 k_w0_f32 =
                convert_bfloat128_to_float128(k_w0, SW_LINEAR);
            const float128 k_w1_f32 =
                convert_bfloat128_to_float128(k_w1, SW_LINEAR);
            const float128 k_w2_f32 =
                convert_bfloat128_to_float128(k_w2, SW_LINEAR);
            const float128 k_w3_f32 =
                convert_bfloat128_to_float128(k_w3, SW_LINEAR);
            float128 k_conv = {0};
            k_conv.v1 = k_h0_f32.v1 * k_w0_f32.v1;
            k_conv.v1 += k_h1_f32.v1 * k_w1_f32.v1;
            k_conv.v1 += k_h2_f32.v1 * k_w2_f32.v1;
            k_conv.v1 += k_raw_f32.v1 * k_w3_f32.v1;
            k_conv.v2 = k_h0_f32.v2 * k_w0_f32.v2;
            k_conv.v2 += k_h1_f32.v2 * k_w1_f32.v2;
            k_conv.v2 += k_h2_f32.v2 * k_w2_f32.v2;
            k_conv.v2 += k_raw_f32.v2 * k_w3_f32.v2;

            const bfloat128 q_rounded = convert_float128_to_bfloat128(
                q_conv, SW_RHNE | SW_LINEAR);
            const bfloat128 k_rounded = convert_float128_to_bfloat128(
                k_conv, SW_RHNE | SW_LINEAR);
            float128 q = convert_bfloat128_to_float128(q_rounded, SW_LINEAR);
            float128 k = convert_bfloat128_to_float128(k_rounded, SW_LINEAR);
            q.v1 = q.v1 * v_sigmoid_f32(q.v1);
            q.v2 = q.v2 * v_sigmoid_f32(q.v2);
            k.v1 = k.v1 * v_sigmoid_f32(k.v1);
            k.v2 = k.v2 * v_sigmoid_f32(k.v2);
            const bfloat128 q_activated = convert_float128_to_bfloat128(
                q, SW_RHNE | SW_LINEAR);
            const bfloat128 k_activated = convert_float128_to_bfloat128(
                k, SW_RHNE | SW_LINEAR);
            q = convert_bfloat128_to_float128(q_activated, SW_LINEAR);
            k = convert_bfloat128_to_float128(k_activated, SW_LINEAR);

            v_bf16_st_tnsr(q_state0_coords, arg0, q_h1);
            v_bf16_st_tnsr(q_state1_coords, arg0, q_h2);
            v_bf16_st_tnsr(q_state2_coords, arg0, q_raw);
            v_bf16_st_tnsr(k_state0_coords, arg0, k_h1);
            v_bf16_st_tnsr(k_state1_coords, arg0, k_h2);
            v_bf16_st_tnsr(k_state2_coords, arg0, k_raw);

            float64 q_norm = v_f32_mac_b(q.v1, q.v1, q.v2 * q.v2);
            q_norm = v_f32_reduce_add(q_norm);
            q_norm = v_f32_shuffle_b(q_norm, broadcast_lane_zero, 0, q_norm);
            q_norm = v_rsqrt_f32(q_norm + 1.0e-6f) * 0.08838834764831845f;
            q.v1 = q.v1 * q_norm;
            q.v2 = q.v2 * q_norm;

            float64 k_norm = v_f32_mac_b(k.v1, k.v1, k.v2 * k.v2);
            k_norm = v_f32_reduce_add(k_norm);
            k_norm = v_f32_shuffle_b(k_norm, broadcast_lane_zero, 0, k_norm);
            k_norm = v_rsqrt_f32(k_norm + 1.0e-6f);
            k.v1 = k.v1 * k_norm;
            k.v2 = k.v2 * k_norm;

            for (int head_group = 0; head_group < 3; ++head_group)
            {
                const int value_head = key_head * 3 + head_group;
                int5 gate_coords = {value_head, batch, 0, 0, 0};
                int5 head_coords = {value_head, 0, 0, 0, 0};
                const float gate_x =
                    s_convert_bf16_to_f32(s_bf16_ld_g(gen_addr(gate_coords, arg3))) +
                    s_f32_ld_g(gen_addr(head_coords, arg6));
                float64 gate_x_vector = gate_x;
                float64 softplus = gate_x_vector;
                if (gate_x <= 20.0f)
                    softplus = v_log_f32(v_exp_f32(gate_x_vector) + 1.0f);
                const float64 a_log = s_f32_ld_g(gen_addr(head_coords, arg5));
                const float64 decay = v_exp_f32(-v_exp_f32(a_log) * softplus);
                const float64 beta = v_sigmoid_f32(
                    s_convert_bf16_to_f32(
                        s_bf16_ld_g(gen_addr(gate_coords, arg4))));

#pragma loop_unroll(1)
                for (int value_row = 0;
                     value_row < TRITON_GAUDI_GDN_VALUE_DIM;
                     ++value_row)
                {
                    const int channel =
                        value_offset + value_head * TRITON_GAUDI_GDN_VALUE_DIM +
                        value_row;
                    int5 raw_coords = {channel, batch, 0, 0, 0};
                    int5 conv0_coords = {channel, 0, conv_slot, 0, 0};
                    int5 conv1_coords = {channel, 1, conv_slot, 0, 0};
                    int5 conv2_coords = {channel, 2, conv_slot, 0, 0};
                    int5 weight0_coords = {channel, 0, 0, 0, 0};
                    int5 weight1_coords = {channel, 1, 0, 0, 0};
                    int5 weight2_coords = {channel, 2, 0, 0, 0};
                    int5 weight3_coords = {channel, 3, 0, 0, 0};
                    const bf16 raw_value = s_bf16_ld_g(gen_addr(raw_coords, arg2));
                    const bf16 value_h0 = s_bf16_ld_g(gen_addr(conv0_coords, arg0));
                    const bf16 value_h1 = s_bf16_ld_g(gen_addr(conv1_coords, arg0));
                    const bf16 value_h2 = s_bf16_ld_g(gen_addr(conv2_coords, arg0));
                    float value_conv =
                        s_convert_bf16_to_f32(value_h0) *
                        s_convert_bf16_to_f32(
                            s_bf16_ld_g(gen_addr(weight0_coords, arg8)));
                    value_conv +=
                        s_convert_bf16_to_f32(value_h1) *
                        s_convert_bf16_to_f32(
                            s_bf16_ld_g(gen_addr(weight1_coords, arg8)));
                    value_conv +=
                        s_convert_bf16_to_f32(value_h2) *
                        s_convert_bf16_to_f32(
                            s_bf16_ld_g(gen_addr(weight2_coords, arg8)));
                    value_conv +=
                        s_convert_bf16_to_f32(raw_value) *
                        s_convert_bf16_to_f32(
                            s_bf16_ld_g(gen_addr(weight3_coords, arg8)));
                    const bf16 value_rounded =
                        s_convert_f32_to_bf16(value_conv, SW_RHNE);
                    const float value_rounded_f32 =
                        s_convert_bf16_to_f32(value_rounded);
                    float64 value_vector = value_rounded_f32;
                    value_vector =
                        value_vector * v_sigmoid_f32(value_vector);
                    float128 value_activated_f32 = {0};
                    value_activated_f32.v1 = value_vector;
                    const bfloat128 value_activated_bf16 =
                        convert_float128_to_bfloat128(
                            value_activated_f32, SW_RHNE | SW_LINEAR);
                    const float128 value_activated =
                        convert_bfloat128_to_float128(
                            value_activated_bf16, SW_LINEAR);
                    const float64 value = value_activated.v1;

                    s_bf16_st_g(gen_addr(conv0_coords, arg0), value_h1);
                    s_bf16_st_g(gen_addr(conv1_coords, arg0), value_h2);
                    s_bf16_st_g(gen_addr(conv2_coords, arg0), raw_value);

                    int5 state_coords = {0, value_row, value_head, state_slot, 0};
                    float64 state_lo = v_f32_ld_tnsr_b(state_coords, arg1) * decay;
                    state_coords[0] = 64;
                    float64 state_hi = v_f32_ld_tnsr_b(state_coords, arg1) * decay;
                    float64 projection = state_lo * k.v1;
                    projection = v_f32_mac_b(state_hi, k.v2, projection);
                    projection = v_f32_reduce_add(projection);
                    projection = v_f32_shuffle_b(
                        projection, broadcast_lane_zero, 0, projection);
                    const float64 delta = (value - projection) * beta;
                    state_lo = v_f32_mac_b(k.v1, delta, state_lo);
                    state_hi = v_f32_mac_b(k.v2, delta, state_hi);
                    float64 result = state_lo * q.v1;
                    result = v_f32_mac_b(state_hi, q.v2, result);
                    result = v_f32_reduce_add(result);

                    state_coords[0] = 0;
                    v_f32_st_tnsr(state_coords, arg1, state_lo);
                    state_coords[0] = 64;
                    v_f32_st_tnsr(state_coords, arg1, state_hi);
                    int5 output_coords = {value_row, value_head, batch, 0, 0};
                    float128 result_f32 = {0};
                    result_f32.v1 = result;
                    const bfloat128 result_bf16 = convert_float128_to_bfloat128(
                        result_f32, SW_RHNE | SW_LINEAR);
                    v_bf16_st_tnsr_partial(
                        output_coords, arg9, result_bf16, 0, 0);
                }
            }
        }
    }
}
"""
    return TpcCSource(program, source)


def _emit_gdn_qk_conv_packed_tpc_c(program: GaudiProgram) -> TpcCSource:
    expected = (
        ("tensor", "bf16"),
        ("tensor", "bf16"),
        ("tensor", "i32"),
        ("tensor", "bf16"),
        ("tensor", "bf16"),
        ("scalar", "i32"),
    )
    if tuple((argument.kind, argument.dtype) for argument in program.arguments) != expected:
        raise GaudiLoweringError(
            "TPC-C packed Q/K convolution requires the canonical ABI")
    qk_tile = int((program.parameters or {}).get("qk_tile", 0))
    if (qk_tile not in (128, 256, 512) or
            program.block_size != qk_tile or program.vector_lanes != 128 or
            program.index_space_rank != 2):
        raise GaudiLoweringError(
            "TPC-C packed Q/K convolution has invalid index-space metadata")
    source = """// Generated by Triton Gaudi2 backend. Do not edit.
#define TRITON_GAUDI_QK_CHANNELS 4096
#define TRITON_GAUDI_PACKED_WIDTH 10240
#define TRITON_GAUDI_CHANNEL_TILE 128
#define TRITON_GAUDI_VECTOR_LANES 128

void main(
    tensor arg0,
    tensor arg1,
    tensor arg2,
    tensor arg3,
    tensor arg4,
    int arg5)
{
    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;
    for (int batch = index_space_start[1]; batch < index_space_end[1]; ++batch)
    {
        int5 index_coords = {batch, 0, 0, 0, 0};
        const int raw_slot = s_i32_ld_g(gen_addr(index_coords, arg2));
        const int conv_slot = ((raw_slot % arg5) + arg5) % arg5;
        for (int channel_block = index_space_start[0];
             channel_block < index_space_end[0];
             ++channel_block)
        {
            for (int channel_lane = 0;
                 channel_lane < TRITON_GAUDI_CHANNEL_TILE;
                 channel_lane += TRITON_GAUDI_VECTOR_LANES)
            {
            const int channel =
                channel_block * TRITON_GAUDI_CHANNEL_TILE + channel_lane;
            int5 raw_coords = {channel, batch, 0, 0, 0};
            int5 history0_coords = {channel, 0, conv_slot, 0, 0};
            int5 history1_coords = {channel, 1, conv_slot, 0, 0};
            int5 history2_coords = {channel, 2, conv_slot, 0, 0};
            int5 weight0_coords = {channel, 0, 0, 0, 0};
            int5 weight1_coords = {channel, 1, 0, 0, 0};
            int5 weight2_coords = {channel, 2, 0, 0, 0};
            int5 weight3_coords = {channel, 3, 0, 0, 0};
            const bfloat128 raw = v_bf16_ld_tnsr_b(raw_coords, arg1);
            const bfloat128 history0 =
                v_bf16_ld_tnsr_b(history0_coords, arg0);
            const bfloat128 history1 =
                v_bf16_ld_tnsr_b(history1_coords, arg0);
            const bfloat128 history2 =
                v_bf16_ld_tnsr_b(history2_coords, arg0);
            const bfloat128 weight0 =
                v_bf16_ld_tnsr_b(weight0_coords, arg3);
            const bfloat128 weight1 =
                v_bf16_ld_tnsr_b(weight1_coords, arg3);
            const bfloat128 weight2 =
                v_bf16_ld_tnsr_b(weight2_coords, arg3);
            const bfloat128 weight3 =
                v_bf16_ld_tnsr_b(weight3_coords, arg3);
            const float128 history0_f32 =
                convert_bfloat128_to_float128(history0, SW_LINEAR);
            const float128 history1_f32 =
                convert_bfloat128_to_float128(history1, SW_LINEAR);
            const float128 history2_f32 =
                convert_bfloat128_to_float128(history2, SW_LINEAR);
            const float128 raw_f32 =
                convert_bfloat128_to_float128(raw, SW_LINEAR);
            const float128 weight0_f32 =
                convert_bfloat128_to_float128(weight0, SW_LINEAR);
            const float128 weight1_f32 =
                convert_bfloat128_to_float128(weight1, SW_LINEAR);
            const float128 weight2_f32 =
                convert_bfloat128_to_float128(weight2, SW_LINEAR);
            const float128 weight3_f32 =
                convert_bfloat128_to_float128(weight3, SW_LINEAR);
            float128 conv = {0};
            conv.v1 = history0_f32.v1 * weight0_f32.v1;
            conv.v1 += history1_f32.v1 * weight1_f32.v1;
            conv.v1 += history2_f32.v1 * weight2_f32.v1;
            conv.v1 += raw_f32.v1 * weight3_f32.v1;
            conv.v2 = history0_f32.v2 * weight0_f32.v2;
            conv.v2 += history1_f32.v2 * weight1_f32.v2;
            conv.v2 += history2_f32.v2 * weight2_f32.v2;
            conv.v2 += raw_f32.v2 * weight3_f32.v2;
            const bfloat128 rounded = convert_float128_to_bfloat128(
                conv, SW_RHNE | SW_LINEAR);
            float128 activated =
                convert_bfloat128_to_float128(rounded, SW_LINEAR);
            activated.v1 = activated.v1 * v_sigmoid_f32(activated.v1);
            activated.v2 = activated.v2 * v_sigmoid_f32(activated.v2);
            const bfloat128 result = convert_float128_to_bfloat128(
                activated, SW_RHNE | SW_LINEAR);
            v_bf16_st_tnsr(history0_coords, arg0, history1);
            v_bf16_st_tnsr(history1_coords, arg0, history2);
            v_bf16_st_tnsr(history2_coords, arg0, raw);
            int5 output_coords = {channel, batch, 0, 0, 0};
            v_bf16_st_tnsr(output_coords, arg4, result);
            }
        }
    }
}
"""
    source = source.replace(
        "#define TRITON_GAUDI_CHANNEL_TILE 128",
        f"#define TRITON_GAUDI_CHANNEL_TILE {qk_tile}",
    )
    return TpcCSource(program, source)


def _emit_gdn_decode_value_conv_packed_tpc_c(
    program: GaudiProgram,
) -> TpcCSource:
    expected = (
        ("tensor", "bf16"),
        ("tensor", "f32"),
        ("tensor", "bf16"),
        ("tensor", "bf16"),
        ("tensor", "bf16"),
        ("tensor", "bf16"),
        ("tensor", "f32"),
        ("tensor", "f32"),
        ("tensor", "i32"),
        ("tensor", "bf16"),
        ("tensor", "bf16"),
        ("scalar", "i32"),
        ("scalar", "i32"),
    )
    if tuple((argument.kind, argument.dtype) for argument in program.arguments) != expected:
        raise GaudiLoweringError(
            "TPC-C fused value-conv + GDN requires the canonical ABI")
    parameters = program.parameters or {}
    value_tile = int(parameters.get("value_tile", 0))
    if (value_tile not in (16, 32, 64, 128) or
            program.block_size != value_tile or program.vector_lanes != 64 or
            program.index_space_rank != 3):
        raise GaudiLoweringError(
            "TPC-C fused value-conv + GDN has invalid tile metadata")
    if value_tile == 16:
        value_broadcast = """                    const uchar256 value_lane_selector =
                        0x80 | (tile_row & 7) | ((tile_row >> 3) << 5);
                    const float64 value = v_f32_shuffle_b(
                        value_activated.v1,
                        value_lane_selector,
                        0,
                        value_activated.v1);"""
    else:
        value_broadcast = """                    const int value_half_row = tile_row & 63;
                    const uchar256 value_lane_selector =
                        0x80 | (value_half_row & 7) |
                        (((value_half_row >> 3) & 1) << 5);
                    const float64 value_source = tile_row < 64 ?
                        value_activated.v1 : value_activated.v2;
                    const float64 value_in_source_group = v_f32_shuffle_b(
                        value_source,
                        value_lane_selector,
                        0,
                        value_source);
                    float64 value = value_in_source_group;
                    if (value_half_row < 16)
                        value = v_f32_mov_dual_group_all_b(
                            value_in_source_group,
                            0xffffffff,
                            0, 0, 0, 0,
                            MkWrA(3, 3, 3, 3),
                            value_in_source_group);
                    else if (value_half_row < 32)
                        value = v_f32_mov_dual_group_all_b(
                            value_in_source_group,
                            0xffffffff,
                            1, 1, 1, 1,
                            MkWrA(3, 3, 3, 3),
                            value_in_source_group);
                    else if (value_half_row < 48)
                        value = v_f32_mov_dual_group_all_b(
                            value_in_source_group,
                            0xffffffff,
                            2, 2, 2, 2,
                            MkWrA(3, 3, 3, 3),
                            value_in_source_group);
                    else
                        value = v_f32_mov_dual_group_all_b(
                            value_in_source_group,
                            0xffffffff,
                            3, 3, 3, 3,
                            MkWrA(3, 3, 3, 3),
                            value_in_source_group);"""
    source = f"""// Generated by Triton Gaudi2 backend. Do not edit.
#define TRITON_GAUDI_GDN_KEY_HEADS 16
#define TRITON_GAUDI_GDN_VALUE_HEADS 48
#define TRITON_GAUDI_GDN_KEY_DIM 128
#define TRITON_GAUDI_GDN_VALUE_DIM 128
#define TRITON_GAUDI_GDN_QK_WIDTH 4096
#define TRITON_GAUDI_GDN_PACKED_WIDTH 10240
#define TRITON_GAUDI_GDN_VALUE_TILE {value_tile}

void main(
    tensor arg0,
    tensor arg1,
    tensor arg2,
    tensor arg3,
    tensor arg4,
    tensor arg5,
    tensor arg6,
    tensor arg7,
    tensor arg8,
    tensor arg9,
    tensor arg10,
    int arg11,
    int arg12)
{{
    const uchar256 broadcast_lane_zero = 0x80;
    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;
    const int value_offset = TRITON_GAUDI_GDN_QK_WIDTH;
    for (int batch = index_space_start[2]; batch < index_space_end[2]; ++batch)
    {{
        int5 index_coords = {{batch, 0, 0, 0, 0}};
        const int raw_slot = s_i32_ld_g(gen_addr(index_coords, arg8));
        const int conv_slot = ((raw_slot % arg11) + arg11) % arg11;
        const int state_slot = ((raw_slot % arg12) + arg12) % arg12;
        for (int value_head = index_space_start[1];
             value_head < index_space_end[1];
             ++value_head)
        {{
            const int key_head = value_head / 3;
            int5 q_coords = {{
                key_head * TRITON_GAUDI_GDN_KEY_DIM, batch, 0, 0, 0}};
            int5 k_coords = {{
                2048 + key_head * TRITON_GAUDI_GDN_KEY_DIM,
                batch,
                0,
                0,
                0}};
            const bfloat128 q_bf16 = v_bf16_ld_tnsr_b(q_coords, arg2);
            const bfloat128 k_bf16 = v_bf16_ld_tnsr_b(k_coords, arg2);
            float128 q = convert_bfloat128_to_float128(q_bf16, SW_LINEAR);
            float128 k = convert_bfloat128_to_float128(k_bf16, SW_LINEAR);
            float64 q_norm = v_f32_mac_b(q.v1, q.v1, q.v2 * q.v2);
            q_norm = v_f32_reduce_add(q_norm);
            q_norm = v_f32_shuffle_b(q_norm, broadcast_lane_zero, 0, q_norm);
            q_norm = v_rsqrt_f32(q_norm + 1.0e-6f) * 0.08838834764831845f;
            q.v1 = q.v1 * q_norm;
            q.v2 = q.v2 * q_norm;
            float64 k_norm = v_f32_mac_b(k.v1, k.v1, k.v2 * k.v2);
            k_norm = v_f32_reduce_add(k_norm);
            k_norm = v_f32_shuffle_b(k_norm, broadcast_lane_zero, 0, k_norm);
            k_norm = v_rsqrt_f32(k_norm + 1.0e-6f);
            k.v1 = k.v1 * k_norm;
            k.v2 = k.v2 * k_norm;

            int5 gate_coords = {{value_head, batch, 0, 0, 0}};
            int5 head_coords = {{value_head, 0, 0, 0, 0}};
            const float gate_x =
                s_convert_bf16_to_f32(s_bf16_ld_g(gen_addr(gate_coords, arg4))) +
                s_f32_ld_g(gen_addr(head_coords, arg7));
            float64 gate_x_vector = gate_x;
            float64 softplus = gate_x_vector;
            if (gate_x <= 20.0f)
                softplus = v_log_f32(v_exp_f32(gate_x_vector) + 1.0f);
            const float64 a_log = s_f32_ld_g(gen_addr(head_coords, arg6));
            const float64 decay = v_exp_f32(-v_exp_f32(a_log) * softplus);
            const float64 beta = v_sigmoid_f32(
                s_convert_bf16_to_f32(
                    s_bf16_ld_g(gen_addr(gate_coords, arg5))));
            for (int value_block = index_space_start[0];
                 value_block < index_space_end[0];
                 ++value_block)
            {{
                const int value_start =
                    value_block * TRITON_GAUDI_GDN_VALUE_TILE;
                const int channel_start =
                    value_offset + value_head * TRITON_GAUDI_GDN_VALUE_DIM +
                    value_start;
                int5 raw_coords = {{channel_start, batch, 0, 0, 0}};
                int5 conv0_coords = {{channel_start, 0, conv_slot, 0, 0}};
                int5 conv1_coords = {{channel_start, 1, conv_slot, 0, 0}};
                int5 conv2_coords = {{channel_start, 2, conv_slot, 0, 0}};
                int5 weight0_coords = {{channel_start, 0, 0, 0, 0}};
                int5 weight1_coords = {{channel_start, 1, 0, 0, 0}};
                int5 weight2_coords = {{channel_start, 2, 0, 0, 0}};
                int5 weight3_coords = {{channel_start, 3, 0, 0, 0}};
                const bfloat128 raw_values = v_bf16_ld_tnsr_partial_b(
                    raw_coords,
                    arg3,
                    TRITON_GAUDI_GDN_VALUE_TILE - 1,
                    0);
                const bfloat128 history0 = v_bf16_ld_tnsr_partial_b(
                    conv0_coords,
                    arg0,
                    TRITON_GAUDI_GDN_VALUE_TILE - 1,
                    0);
                const bfloat128 history1 = v_bf16_ld_tnsr_partial_b(
                    conv1_coords,
                    arg0,
                    TRITON_GAUDI_GDN_VALUE_TILE - 1,
                    0);
                const bfloat128 history2 = v_bf16_ld_tnsr_partial_b(
                    conv2_coords,
                    arg0,
                    TRITON_GAUDI_GDN_VALUE_TILE - 1,
                    0);
                const bfloat128 weight0 = v_bf16_ld_tnsr_partial_b(
                    weight0_coords,
                    arg9,
                    TRITON_GAUDI_GDN_VALUE_TILE - 1,
                    0);
                const bfloat128 weight1 = v_bf16_ld_tnsr_partial_b(
                    weight1_coords,
                    arg9,
                    TRITON_GAUDI_GDN_VALUE_TILE - 1,
                    0);
                const bfloat128 weight2 = v_bf16_ld_tnsr_partial_b(
                    weight2_coords,
                    arg9,
                    TRITON_GAUDI_GDN_VALUE_TILE - 1,
                    0);
                const bfloat128 weight3 = v_bf16_ld_tnsr_partial_b(
                    weight3_coords,
                    arg9,
                    TRITON_GAUDI_GDN_VALUE_TILE - 1,
                    0);
                const float128 raw_f32 =
                    convert_bfloat128_to_float128(raw_values, SW_LINEAR);
                const float128 history0_f32 =
                    convert_bfloat128_to_float128(history0, SW_LINEAR);
                const float128 history1_f32 =
                    convert_bfloat128_to_float128(history1, SW_LINEAR);
                const float128 history2_f32 =
                    convert_bfloat128_to_float128(history2, SW_LINEAR);
                const float128 weight0_f32 =
                    convert_bfloat128_to_float128(weight0, SW_LINEAR);
                const float128 weight1_f32 =
                    convert_bfloat128_to_float128(weight1, SW_LINEAR);
                const float128 weight2_f32 =
                    convert_bfloat128_to_float128(weight2, SW_LINEAR);
                const float128 weight3_f32 =
                    convert_bfloat128_to_float128(weight3, SW_LINEAR);
                float128 value_conv = {{0}};
                value_conv.v1 = history0_f32.v1 * weight0_f32.v1;
                value_conv.v1 += history1_f32.v1 * weight1_f32.v1;
                value_conv.v1 += history2_f32.v1 * weight2_f32.v1;
                value_conv.v1 += raw_f32.v1 * weight3_f32.v1;
                value_conv.v2 = history0_f32.v2 * weight0_f32.v2;
                value_conv.v2 += history1_f32.v2 * weight1_f32.v2;
                value_conv.v2 += history2_f32.v2 * weight2_f32.v2;
                value_conv.v2 += raw_f32.v2 * weight3_f32.v2;
                const bfloat128 value_rounded =
                    convert_float128_to_bfloat128(
                        value_conv, SW_RHNE | SW_LINEAR);
                float128 value_activated_f32 =
                    convert_bfloat128_to_float128(value_rounded, SW_LINEAR);
                value_activated_f32.v1 = value_activated_f32.v1 *
                    v_sigmoid_f32(value_activated_f32.v1);
                value_activated_f32.v2 = value_activated_f32.v2 *
                    v_sigmoid_f32(value_activated_f32.v2);
                const bfloat128 value_activated_bf16 =
                    convert_float128_to_bfloat128(
                        value_activated_f32, SW_RHNE | SW_LINEAR);
                const float128 value_activated =
                    convert_bfloat128_to_float128(
                        value_activated_bf16, SW_LINEAR);
                v_bf16_st_tnsr_partial(
                    conv0_coords,
                    arg0,
                    history1,
                    TRITON_GAUDI_GDN_VALUE_TILE - 1,
                    0);
                v_bf16_st_tnsr_partial(
                    conv1_coords,
                    arg0,
                    history2,
                    TRITON_GAUDI_GDN_VALUE_TILE - 1,
                    0);
                v_bf16_st_tnsr_partial(
                    conv2_coords,
                    arg0,
                    raw_values,
                    TRITON_GAUDI_GDN_VALUE_TILE - 1,
                    0);

#pragma loop_unroll(2)
                for (int tile_row = 0;
                     tile_row < TRITON_GAUDI_GDN_VALUE_TILE;
                     ++tile_row)
                {{
                    const int value_row = value_start + tile_row;
{value_broadcast}
                    int5 state_coords = {{
                        0, value_row, value_head, state_slot, 0}};
                    float64 state_lo =
                        v_f32_ld_tnsr_b(state_coords, arg1) * decay;
                    state_coords[0] = 64;
                    float64 state_hi =
                        v_f32_ld_tnsr_b(state_coords, arg1) * decay;
                    float64 projection = state_lo * k.v1;
                    projection = v_f32_mac_b(state_hi, k.v2, projection);
                    projection = v_f32_reduce_add(projection);
                    projection = v_f32_shuffle_b(
                        projection, broadcast_lane_zero, 0, projection);
                    const float64 delta = (value - projection) * beta;
                    state_lo = v_f32_mac_b(k.v1, delta, state_lo);
                    state_hi = v_f32_mac_b(k.v2, delta, state_hi);
                    float64 result = state_lo * q.v1;
                    result = v_f32_mac_b(state_hi, q.v2, result);
                    result = v_f32_reduce_add(result);
                    state_coords[0] = 0;
                    v_f32_st_tnsr(state_coords, arg1, state_lo);
                    state_coords[0] = 64;
                    v_f32_st_tnsr(state_coords, arg1, state_hi);
                    int5 output_coords = {{
                        value_row, value_head, batch, 0, 0}};
                    float128 result_f32 = {{0}};
                    result_f32.v1 = result;
                    const bfloat128 result_bf16 =
                        convert_float128_to_bfloat128(
                            result_f32, SW_RHNE | SW_LINEAR);
                    v_bf16_st_tnsr_partial(
                        output_coords, arg10, result_bf16, 0, 0);
                }}
            }}
        }}
    }}
}}
"""
    return TpcCSource(program, source)


def emit_tpc_c(program: GaudiProgram) -> TpcCSource:
    if program.kind == "dynamic_quant":
        return _emit_dynamic_quant_tpc_c(program)
    if program.kind == "fused_add_rms_norm":
        return _emit_fused_add_rms_norm_tpc_c(program)
    if program.kind == "silu_and_mul":
        return _emit_silu_and_mul_tpc_c(program)
    if program.kind == "gdn_decode_packed":
        return _emit_gdn_decode_packed_tpc_c(program)
    if program.kind == "gdn_decode_conv_packed":
        return _emit_gdn_decode_conv_packed_tpc_c(program)
    if program.kind == "gdn_qk_conv_packed":
        return _emit_gdn_qk_conv_packed_tpc_c(program)
    if program.kind == "gdn_decode_value_conv_packed":
        return _emit_gdn_decode_value_conv_packed_tpc_c(program)

    dtype_config = {
        "f32": ("float64", "f32"),
        "bf16": ("bfloat128", "bf16"),
    }
    try:
        vector_type, intrinsic_prefix = dtype_config[program.tensor_dtype]
    except KeyError as exc:
        raise GaudiLoweringError(f"TPC-C does not support tensor dtype {program.tensor_dtype}") from exc

    parameters = []
    for argument in program.arguments:
        if argument.kind == "tensor":
            parameters.append(f"tensor arg{argument.index}")
        elif argument.dtype == "i32":
            parameters.append(f"int arg{argument.index}")
        elif argument.dtype == "u32":
            parameters.append(f"unsigned arg{argument.index}")
        elif argument.dtype == "f32":
            parameters.append(f"float arg{argument.index}")
        else:
            raise GaudiLoweringError(f"TPC-C scalar ABI does not yet support {argument.dtype}")

    bound = f"arg{program.bound_arg}" if program.bound_arg is not None else "block_base + BLOCK_SIZE"
    lines = [
        "// Generated by Triton Gaudi2 backend. Do not edit.",
        f"void main({', '.join(parameters)})",
        "{",
        f"    const int BLOCK_SIZE = {program.block_size};",
        f"    const int VECTOR_LANES = {program.vector_lanes};",
        "    const int5 index_space_start = get_index_space_offset();",
        "    const int5 index_space_end = get_index_space_size() + index_space_start;",
        "    for (int program_id = index_space_start[0];",
        "         program_id < index_space_end[0];",
        "         ++program_id)",
        "    {",
        "    const int block_base = program_id * BLOCK_SIZE;",
        f"    const int valid_end = {bound};",
        "",
        "    for (int lane = 0; lane < BLOCK_SIZE; lane += VECTOR_LANES)",
        "    {",
        "        const int element = block_base + lane;",
        "        const int remaining = valid_end - element;",
        "        if (remaining <= 0)",
        "            break;",
        "        int5 coords = {element, 0, 0, 0, 0};",
    ]
    value = _emit_expression(program.expression, lines, [0], vector_type, intrinsic_prefix)
    lines.extend([
        "        if (remaining >= VECTOR_LANES)",
        f"            v_{intrinsic_prefix}_st_tnsr(coords, arg{program.output_arg}, {value});",
        "        else",
        f"            v_{intrinsic_prefix}_st_tnsr_partial(",
        f"                coords, arg{program.output_arg}, {value}, remaining - 1, 0);",
        "    }",
        "    }",
        "}",
        "",
    ])
    return TpcCSource(program, "\n".join(lines))
