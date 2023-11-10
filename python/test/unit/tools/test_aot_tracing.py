import json
import re
import shutil
from pathlib import Path
from typing import List

import pytest

from triton.compiler.compiler import instance_descriptor
from triton.runtime.jit import JITFunction
from triton.tools.aot.compiler import AOT_C_CUDA_Compiler as AOTCompiler
from triton.tools.aot.compiler import AOTCompilationResult
from triton.tools.aot.linker import AOT_C_CUDA_Linker as AOTLinker
from triton.tools.aot.linker import AOTLinkerResult
from triton.tools.aot.tracing import MatMulConfig, MatMulKernelTracer, TraceConfig

FIXTURES_DIR = Path(__file__).parent.absolute() / "fixtures"

from .matmul_configs import (
    ALL_HINTS,
    DEFAULT_MATMUL_CONSTANTS,
    DEFAULT_MATMUL_DTYPES,
    DEFAULT_MATMUL_HINTS,
    DEFAULT_SIGNATURE,
    NO_HINT_SIGNATURE,
    NO_HINTS,
    STRIDE_AM_HINTS,
    STRIDE_AM_SIGNATURE,
    STRIDE_CM_AM_HINTS,
    STRIDE_CM_AM_SIGNATURE,
    STRIDE_CM_HINTS,
    STRIDE_CM_SIGNATURE,
)
from .matmul_utils import (
    AOTScriptResult,
    AOTScriptRunner,
    MatmulTestConfig,
    tt_to_torch,
)

# ------------------------------------------------------------------------------------------------------------ #
# Tests for `AOTScriptRunner`, which is a wrapper around `triton.tools.compile` and `triton.tools.link`


## Test `generate_signature` for creating signatures to pass to `triton.tools.compile` CLI given dtypes, hints, and constants.
# See `matmul_configs.py` for details
@pytest.mark.parametrize(
    "dtypes, hints, constants, expected_signature",
    [
        (
            DEFAULT_MATMUL_DTYPES,
            DEFAULT_MATMUL_HINTS,
            DEFAULT_MATMUL_CONSTANTS,
            DEFAULT_SIGNATURE,
        ),
        (DEFAULT_MATMUL_DTYPES, NO_HINTS, DEFAULT_MATMUL_CONSTANTS, NO_HINT_SIGNATURE),
        (
            DEFAULT_MATMUL_DTYPES,
            STRIDE_CM_HINTS,
            DEFAULT_MATMUL_CONSTANTS,
            STRIDE_CM_SIGNATURE,
        ),
        (
            DEFAULT_MATMUL_DTYPES,
            STRIDE_AM_HINTS,
            DEFAULT_MATMUL_CONSTANTS,
            STRIDE_AM_SIGNATURE,
        ),
        (
            DEFAULT_MATMUL_DTYPES,
            STRIDE_CM_AM_HINTS,
            DEFAULT_MATMUL_CONSTANTS,
            STRIDE_CM_AM_SIGNATURE,
        ),
    ],
    ids=["default", "no_hints", "stride_cm", "stride_am", "stride_cm_am"],
)
def test_signature(dtypes, hints, constants, expected_signature):
    signature = AOTScriptRunner.generate_signature(
        dtypes=dtypes,
        hints=hints,
        constant_vals=constants,
    )

    assert (
        signature == expected_signature
    ), f"Expected signature: {expected_signature}, Actual signature: {signature}"


## Test wrapper for `triton.tools.compile` for generating reference kernels
# See
def test_kernel_compilation():
    out_dir = FIXTURES_DIR / "aot_reference_kernels"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    AOTScriptRunner.compile_matmul_kernels(
        NO_HINT_SIGNATURE, 1, "M/16, N/16, 1", out_dir=out_dir
    )
    kernel_headers = list(out_dir.glob("*.h"))
    kernel_sources = list(out_dir.glob("*.c"))
    assert len(kernel_headers) == 1
    assert len(kernel_sources) == 1


"""
Tests to replicate the matmul kernels in `test_aot.py` using the refactored object-oriented AOTCompiler and AOTLinker classes.
Currently AOTCompiler and AOTLinker are abstract classes for generating kernel headers / sources and linking these into dispatchable kernels.
The AOT_C_CUDA_Compiler and AOT_C_CUDA_Linker classes are concrete implementations of these abstract classes for generating C/CUDA kernels.

See `triton/tools/aot/compiler.py` and `triton/tools/aot/linker.py` for details.

    - "default" - no specialization for `stride_cm`, `stride_am`
    ```
    {
        "C": 16,
        "A": 16,
        "B": 16,
        "M": None,
        "N": None,
        "K": None,
        "stride_cm": None,
        "stride_cn": 1,
        "stride_am": None,
        "stride_ak": 1,
        "stride_bk": 16,
        "stride_bn": 1,
    }
    ```
    - "stride_cm" - `stride_cm` specialized to 16
    - "stride_am" - `stride_am` specialized to 16
    - "stride_cm_am" - `stride_cm` and `stride_am` specialized to 16

Additional test cases:
    - "no_hints" - no specialization for any arg
    - "all_hints" - specialization for all args

See `matmul_configs.py` for config details
"""

TEST_CONFIGS = {
    "default": MatmulTestConfig(
        dtypes=DEFAULT_MATMUL_DTYPES,
        hints=DEFAULT_MATMUL_HINTS,
        constants=DEFAULT_MATMUL_CONSTANTS,
        num_warps=1,
        grid="M/16, N/16, 1",
    ),
    # Same as default except `stride_cm` also specialized to 16
    "stride_cm": MatmulTestConfig(
        dtypes=DEFAULT_MATMUL_DTYPES,
        hints=STRIDE_CM_HINTS,
        constants=DEFAULT_MATMUL_CONSTANTS,
        num_warps=1,
        grid="M/16, N/16, 1",
    ),
    # Same as default except `stride_am` also specialized to 16
    "stride_am": MatmulTestConfig(
        dtypes=DEFAULT_MATMUL_DTYPES,
        hints=STRIDE_AM_HINTS,
        constants=DEFAULT_MATMUL_CONSTANTS,
        num_warps=1,
        grid="M/16, N/16, 1",
    ),
    # Same as default except `stride_cm` and `stride_am` both specialized to 16
    "stride_cm_am": MatmulTestConfig(
        dtypes=DEFAULT_MATMUL_DTYPES,
        hints=STRIDE_CM_AM_HINTS,
        constants=DEFAULT_MATMUL_CONSTANTS,
        num_warps=1,
        grid="M/16, N/16, 1",
    ),
    "no_hints": MatmulTestConfig(
        dtypes=DEFAULT_MATMUL_DTYPES,
        hints=NO_HINTS,
        constants=DEFAULT_MATMUL_CONSTANTS,
        num_warps=1,
        grid="M/16, N/16, 1",
    ),
    "all_hints": MatmulTestConfig(
        dtypes=DEFAULT_MATMUL_DTYPES,
        hints=ALL_HINTS,
        constants=DEFAULT_MATMUL_CONSTANTS,
        num_warps=1,
        grid="M/16, N/16, 1",
    ),
}

# ------------------------------------------------------------------------------------------------------------ #


# Small utilities for checking that generated code matches reference code
def _preprocess_src(src):
    return list(filter(lambda x: x.strip(), src.split("\n")))


def check_codegen(actual: str, expected: str, skip: List[str] = None, verbose=False):
    """Check that the generated code is the same as the reference code

    Checks for exact text line by line. Ignores lines containing text in `ignore`.

    Args:
        actual (str): generated code
        expected (str): reference
        ignore (List[str], optional): skip line if contains text in `ignore`. Defaults to None.
    """
    actual_lines, expected_lines = _preprocess_src(actual), _preprocess_src(expected)
    mismatches = []

    for lineno, (actual, expected) in enumerate(zip(actual_lines, expected_lines), 1):
        if skip and any(i in actual for i in skip):
            continue

        if actual != expected:
            mismatches.append((lineno, actual, expected))
            if verbose:
                print(
                    f"Line {lineno} mismatch:\n  Actual: {actual[:100]}\n  Expected: {expected[:100]}"
                )
    # Special handling for linked source code where the order of dispatching conditions can differ
    # if multiple configs have the same number of specializations
    # In these cases, we test that the actual dispatching condition exists in the expected code
    ok_lines = []
    for i, (_, actual, expected) in enumerate(mismatches):
        if actual.lstrip().startswith("if") and expected.lstrip().startswith("if"):
            # test that actual dispatch condition exists in expected source
            if actual in expected_lines:
                ok_lines.append(i)
    mismatches = [m for i, m in enumerate(mismatches) if i not in ok_lines]

    assert (
        not mismatches
    ), f'Mismatch in generated code at lines {", ".join(str(l) for l in mismatches)}'
    # if mismatches:
    #     mismatch_str = "\n".join(mismatches)
    #     raise ValueError(f"Mismatches:\n {mismatch_str}")


def check_linked_source(
    actual: str, expected: str, skip: List[str] = None, verbose=False
):
    actual_lines, expected_lines = _preprocess_src(actual), _preprocess_src(expected)
    mismatches = []
    for i in range(len(actual_lines)):
        actual_line = actual_lines[i]

        if actual_line.lstrip().startswith("if") and "return" in actual_lines[i + 1]:
            if verbose:
                print(
                    f"Checking dispatch condition:\nActual: {actual_line}\n{actual_lines[i+1]}"
                )
            # Check that the actual dispatch condition exists in the expected source
            assert actual_line in expected_lines
            # Parse return statement for args -- function name won't match because of suffix
            actual_dispatch_fn = actual_lines[i + 1]
            match = re.search(r"\((.*?)\)", actual_dispatch_fn)
            assert match is not None
            actual_dispatch_args = match.group(1).split(",")
            expected_line = expected_lines[expected_lines.index(actual_line)]
            expected_dispatch_fn = expected_lines[expected_lines.index(actual_line) + 1]
            if verbose:
                print(f"Expected: {expected_line}\n{expected_dispatch_fn}")
            assert "return" in expected_dispatch_fn
            match = re.search(r"\((.*?)\)", expected_dispatch_fn)
            assert match is not None
            expected_dispatch_args = match.group(1).split(",")
            assert actual_dispatch_args == expected_dispatch_args
        else:
            if skip and any(i in actual_line for i in skip):
                continue

            if actual_line != expected_lines[i]:
                mismatches.append((i, actual, expected))
                if verbose:
                    print(
                        f"Line {i} mismatch:\n  Actual: {actual[:100]}\n  Expected: {expected[:100]}"
                    )
        assert not mismatches


# ------------------------------------------------------------------------------------------------------------ #
class TestMatMulCodegen:
    @pytest.fixture(
        scope="class",
        params=[
            # Single config tests
            ("default",),
            ("stride_cm",),
            ("stride_am",),
            ("stride_cm_am",),
            ("all_hints",),
            ("no_hints",),
            # Multi config tests
            ("all_hints", "no_hints"),
            ("default", "stride_cm", "stride_am", "stride_cm_am"),
        ],
        ids=lambda params: "-".join([p.upper() for p in params]),
    )
    def configs(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def kernel_configs(self, configs):
        return [TEST_CONFIGS[cfg] for cfg in configs]

    @pytest.fixture(scope="class")
    def test_dir(self, configs):
        test_dir = (
            Path(__file__).parent
            / "matmul_codegen_test"
            / "_".join(cfg for cfg in configs)
        ).absolute()
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
        return test_dir

    @pytest.fixture(scope="class")
    def reference_dir(self, test_dir):
        reference_dir = test_dir / "reference_aot_kernels"
        reference_dir.mkdir(parents=True, exist_ok=True)
        return reference_dir

    @pytest.fixture(scope="class")
    def codegen_dir(self, test_dir):
        codegen_dir = test_dir / "codegen_kernels"
        codegen_dir.mkdir(parents=True, exist_ok=True)
        return codegen_dir

    @pytest.fixture(scope="class")
    def kernel_path(self):
        kernel_path = FIXTURES_DIR / "kernels" / "matmul_kernel.py"
        return kernel_path

    @pytest.fixture(scope="class")
    def kernel_name(self):
        """Must match the name of the kernel in `matmul_kernel.py`"""
        return "matmul"

    @pytest.fixture(scope="class")
    def expected_kernels(
        self, kernel_name, reference_dir: Path, kernel_configs, kernel_path: Path
    ):
        signatures = [
            AOTScriptRunner.generate_signature(
                kernel_config.dtypes, kernel_config.hints, kernel_config.constants
            )
            for kernel_config in kernel_configs
        ]

        num_warps = [kernel_config.num_warps for kernel_config in kernel_configs]
        grids = [kernel_config.grid for kernel_config in kernel_configs]

        AOTScriptRunner.compile_matmul_kernels(
            kernel_name,
            signatures,
            num_warps=num_warps,
            grids=grids,
            out_dir=reference_dir,
            kernel_path=kernel_path,
        )
        headers = list(reference_dir.glob("*.h"))
        sources = list(reference_dir.glob("*.c"))

        linked_header = list(reference_dir.glob(f"{kernel_name}.h"))
        kernel_headers = list(set(headers) - set(linked_header))
        linked_source = list(reference_dir.glob(f"{kernel_name}.c"))
        kernel_sources = list(set(sources) - set(linked_source))

        jit_args = list(reference_dir.glob("*jit_args.json"))
        compiler_params = list(reference_dir.glob("*params.json"))

        return AOTScriptResult(
            kernel_headers=kernel_headers,
            kernel_sources=kernel_sources,
            linked_header=linked_header,
            linked_source=linked_source,
            jit_args=jit_args,
            compiler_params=compiler_params,
        )

    def test_script_gen(self, expected_kernels):
        assert len(expected_kernels.linked_header) == 1
        assert len(expected_kernels.kernel_headers) >= 1

        assert len(expected_kernels.linked_source) == 1
        assert len(expected_kernels.kernel_sources) >= 1

    def _parse_jit_args(self, args_path):
        class JITArgTypes:
            """Expected types for JIT args"""

            INT = ["num_warps", "num_stages", "num_ctas", "device"]
            STRING = ["device_type"]
            DICT = [
                "signature",
                "constants",
                "_original_signature",
                "_original_constants",
            ]
            LIST = ["grid", "configs"]
            BOOL = ["enable_warp_specialization", "enable_fp_fusion", "debug"]

        class JITArgDeserializer:  # Need special handling for
            @staticmethod
            def deserialize(args):
                parsed_args = {}
                for k, v in args.items():
                    if k in JITArgTypes.INT:
                        parsed_args[k] = int(v)
                    elif k in JITArgTypes.BOOL:
                        if v.lower() == "true":
                            parsed_args[k] = True
                        elif v.lower() == "false":
                            parsed_args[k] = False
                        else:
                            raise ValueError(f"Invalid bool value {v}")
                    elif k in JITArgTypes.DICT:
                        # Cast arg positions to ints for signature and constants
                        parsed_args[k] = {int(k): v for k, v in v.items()}
                    elif k == "configs":
                        # Create instance descriptors from dict representation
                        parsed_args[k] = [instance_descriptor(**cfg) for cfg in v]
                    else:
                        parsed_args[k] = v
                return parsed_args

        raw_args = json.load(args_path.open())
        return JITArgDeserializer.deserialize(raw_args)

    @pytest.fixture(scope="class")
    def codegen_kernels(self, kernel_name, kernel_path, expected_kernels, codegen_dir):
        jit_args = []
        for p in expected_kernels.jit_args:
            jit_args.append(self._parse_jit_args(p))

        compiled_results = []
        for args in jit_args:
            jit_fn = JITFunction(kernel_path)
            compiler = AOTCompiler(
                kernel_name=kernel_name,
                jit_args=args,
                jit_fn=jit_fn,
                save_dir=codegen_dir,
            )
            compiled_result: AOTCompilationResult = compiler.generate()
            compiled_results.append(compiled_result)
        return compiled_results

    @pytest.fixture(scope="class")
    def codegen_linked_kernels(self, kernel_name, codegen_kernels, codegen_dir: Path):
        headers = [t.header_path for t in codegen_kernels]
        linker = AOTLinker(
            kernel_name=kernel_name,
            headers=headers,
            save_dir=codegen_dir,
        )

        linker_result: AOTLinkerResult = linker.generate()
        return linker_result

    def test_aot_compiler_params(
        self,
        expected_kernels,
        codegen_kernels,
    ) -> List[AOTCompilationResult]:
        # Load
        # headers, sources, jit_args, compiler_params = self.expected_kernels
        actual_params = [k._compiler_params for k in codegen_kernels]
        actual_params = sorted(actual_params, key=lambda x: x["kernel_name"])
        for actual in actual_params:
            kernel_sig = "_".join(actual["kernel_name"].split("_")[1:])
            expected = [
                p for p in expected_kernels.compiler_params if kernel_sig in str(p)
            ][0]
            expected = json.load(expected.open())
            for k in actual.keys():
                assert (
                    actual[k] == expected[k]
                ), f"{k.upper()} not equal\n\tExpected: {expected[k]}, Actual: {actual[k]}"

    def test_aot_codegen_kernel_headers(
        self,
        expected_kernels,
        codegen_kernels,
    ) -> List[AOTCompilationResult]:
        # Load
        # headers, sources, jit_args, compiler_params = self.expected_kernels
        actual_headers = [k.header for k in codegen_kernels]
        for actual, expected in zip(
            sorted(actual_headers), sorted(expected_kernels.kernel_headers)
        ):
            expected = expected.read_text()
            check_codegen(actual, expected)

    def test_aot_codegen_kernel_sources(
        self,
        expected_kernels,
        codegen_kernels,
    ) -> List[AOTCompilationResult]:
        # Load
        # headers, sources, jit_args, compiler_params = self.expected_kernels
        actual_sources = [k.source for k in codegen_kernels]
        for actual, expected in zip(
            sorted(actual_sources), sorted(expected_kernels.kernel_sources)
        ):
            expected = expected.read_text()
            check_codegen(actual, expected)

    def test_aot_codegen_linked_header(
        self,
        expected_kernels,
        codegen_linked_kernels,
    ) -> List[AOTCompilationResult]:
        # Load
        # headers, sources, jit_args, compiler_params = self.expected_kernels

        check_codegen(
            codegen_linked_kernels.header_path.read_text(),
            expected_kernels.linked_header[0].read_text(),
        )

    def test_aot_codegen_linked_source(
        self,
        expected_kernels,
        codegen_linked_kernels,
    ) -> List[AOTCompilationResult]:
        # Load
        # headers, sources, jit_args, compiler_params = self.expected_kernels

        check_codegen(
            codegen_linked_kernels.source_path.read_text(),
            expected_kernels.linked_source[0].read_text(),
        )


"""
Tests for tracing JIT functions and generating AOT kernels from the traced functions.
Motivation is that rather than having the user input a signature, we utilize the existing JIT / compilation machinery to 
generate the necessary inputs to `triton.compiler.compile`.

AOTTracer is an abstract class for tracing JIT functions and generating AOT kernels from the traced functions that utilizes AOTCompiler 
and AOTLinker to generate the AOT kernels. 

AOTTracer is implemented by MatMulKernelTracer, which is a concrete implementation of AOTTracer for the matmul kernel.

Since traced code will differ from scripted (`triton.tools.compile`) due to handling of specializations and signatures,
we don't test for exact codegen matches. In particular, since the AOT script does not handle all specializations in
the JITTed path such as `ids_of_folded_args` and `divisible_by_8`, any codegen'ed that depend on the `config` passed to 
`triton.compiler.compile` will necessarily differ.

This includes:
- the suffix attached generated kernel header and source files
- fields in the `triton.tools.compile.{h,c}` templates such as `kernel_name` and `triton_kernel_name`
- the generated cubin image, which will differ due to the differing specializations.

Hence, when comparing the generated code, we skip the aforementioned fields (`SKIP_PARAMS` attribute).
See `triton/tools/aot/tracing.py` for details.
"""


class TestMatMulTrace:
    SKIP_PARAMS = ["kernel_name", "triton_kernel_name", "bin_size", "bin_data"]

    @pytest.fixture(
        scope="class",
        params=[
            # Single config tests
            ("default",),
            ("stride_cm",),
            ("stride_am",),
            ("stride_cm_am",),
            ("all_hints",),
            ("no_hints",),
            # Multi config tests
            ("all_hints", "no_hints"),
            ("default", "stride_cm", "stride_am", "stride_cm_am"),
        ],  # ("no_hints",),
        ids=lambda params: "-".join([p.upper() for p in params]),
    )
    def configs(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def kernel_configs(self, configs):
        return [TEST_CONFIGS[cfg] for cfg in configs]

    @pytest.fixture(scope="class")
    def test_dir(self, configs):
        test_dir = (
            Path(__file__).parent
            / "matmul_trace_test"
            / "_".join(cfg for cfg in configs)
        ).absolute()
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
        return test_dir

    @pytest.fixture(scope="class")
    def reference_dir(self, test_dir):
        reference_aot_dir = test_dir / "reference_aot_kernels"
        reference_aot_dir.mkdir(parents=True, exist_ok=True)
        return reference_aot_dir

    @pytest.fixture(scope="class")
    def trace_dir(self, test_dir):
        trace_dir = test_dir / "traced_kernels"
        trace_dir.mkdir(parents=True, exist_ok=True)
        return trace_dir

    @pytest.fixture(scope="class")
    def kernel_path(self):
        kernel_path = FIXTURES_DIR / "kernels" / "matmul_kernel.py"
        return kernel_path

    @pytest.fixture(scope="class")
    def kernel_name(self):
        """Must match the name of the kernel in `matmul_kernel.py`"""
        return "matmul"

    @pytest.fixture(scope="class")
    def expected_kernels(
        self, kernel_name, reference_dir: Path, kernel_configs, kernel_path: Path
    ):
        signatures = [
            AOTScriptRunner.generate_signature(
                kernel_config.dtypes, kernel_config.hints, kernel_config.constants
            )
            for kernel_config in kernel_configs
        ]

        num_warps = [kernel_config.num_warps for kernel_config in kernel_configs]
        grids = [kernel_config.grid for kernel_config in kernel_configs]

        AOTScriptRunner.compile_matmul_kernels(
            kernel_name,
            signatures,
            num_warps=num_warps,
            grids=grids,
            out_dir=reference_dir,
            kernel_path=kernel_path,
        )
        headers = list(reference_dir.glob("*.h"))
        sources = list(reference_dir.glob("*.c"))

        linked_header = list(reference_dir.glob(f"{kernel_name}.h"))
        kernel_headers = list(set(headers) - set(linked_header))
        linked_source = list(reference_dir.glob(f"{kernel_name}.c"))
        kernel_sources = list(set(sources) - set(linked_source))

        jit_args = list(reference_dir.glob("*jit_args.json"))
        compiler_params = list(reference_dir.glob("*params.json"))

        return AOTScriptResult(
            kernel_headers=kernel_headers,
            kernel_sources=kernel_sources,
            linked_header=linked_header,
            linked_source=linked_source,
            jit_args=jit_args,
            compiler_params=compiler_params,
        )

    # @pytest.fixture(scope="class")
    @pytest.fixture(scope="class")
    def traced_kernels(
        self,
        trace_dir,
        kernel_path,
        kernel_configs,
    ) -> List[AOTCompilationResult]:
        trace_configs = []
        matmul_configs = []
        for kernel_config in kernel_configs:
            dtype_in = tt_to_torch(kernel_config.dtypes["A"])
            dtype_out = tt_to_torch(kernel_config.dtypes["C"])

            # Assume that M, N, K are divisible by 16; defaults to 16
            matmul_config = MatMulConfig(
                dtype_in=dtype_in,
                dtype_out=dtype_out,
                BLOCK_M=kernel_config.constants["BLOCK_M"],
                BLOCK_N=kernel_config.constants["BLOCK_N"],
                BLOCK_K=kernel_config.constants["BLOCK_K"],
            )
            assert matmul_config.M % matmul_config.BLOCK_M == 0
            assert matmul_config.N % matmul_config.BLOCK_N == 0
            assert matmul_config.K % matmul_config.BLOCK_K == 0

            do_not_specialize = [
                k for k in kernel_config.hints if kernel_config.hints[k] is None
            ]
            trace_config = TraceConfig(
                do_not_specialize=do_not_specialize,
                num_warps=kernel_config.num_warps,
                trace_dir=trace_dir,
                trace_grid=[g.strip() for g in kernel_config.grid.split(",")],
            )
            trace_configs.append(trace_config)
            matmul_configs.append(matmul_config)

        # Use default MatmulConfig (16 x 16 x 16), dtype_in = fp16, dtype_out = fp32
        matmul_tracer = MatMulKernelTracer(kernel_path.parent)
        traces, *_ = matmul_tracer.trace(
            kernel_configs=matmul_configs, trace_configs=trace_configs
        )
        # Save jit args
        for trace in traces:
            with open(trace_dir / f"{trace.kernel_name}-jit_args.json", "w") as fp:
                json.dump({k: str(v) for k, v in trace._jit_args.items()}, fp, indent=2)
            with open(
                trace_dir / f"{trace.kernel_name}-compiler_params.json", "w"
            ) as fp:
                json.dump(
                    {k: str(v) for k, v in trace._compiler_params.items()}, fp, indent=2
                )
        return traces

    @pytest.fixture(scope="class")
    def linked_traces(self, kernel_name, traced_kernels, trace_dir: Path):
        headers = [t.header_path for t in traced_kernels]
        linker = AOTLinker(
            kernel_name=kernel_name,
            headers=headers,
            save_dir=trace_dir,
        )

        linker_result: AOTLinkerResult = linker.generate()
        return linker_result

    @pytest.fixture
    def link_skips(self, traced_kernels):
        skip = []
        for trace in traced_kernels:
            for k, v in trace._compiler_params.items():
                if k in self.SKIP_PARAMS:
                    skip.append(str(v))
        return skip

    def get_skips(self, trace):
        skip = {}
        for k, v in trace._compiler_params.items():
            if k in self.SKIP_PARAMS:
                skip[k] = str(v)
        return skip

    def extract_kernel_sig(self, trace: AOTCompilationResult):
        return "_".join(trace.params["kernel_name"].split("_")[1:])

    def extract_sig_hash(self, trace: AOTCompilationResult):
        return self.extract_kernel_sig(trace).split("_")[0]

    # Only test for presence of `sig hash` in header file name
    # Per above documentation, the suffix will differ due to the differing specializations.
    def test_kernel_header_files(self, traced_kernels, expected_kernels):
        for trace in traced_kernels:
            kernel_sig = self.extract_kernel_sig(trace)
            sig_hash_suffix = kernel_sig.split("_")
            assert len(sig_hash_suffix) == 2

            sig_hash = sig_hash_suffix[0]

            assert any(sig_hash in str(h) for h in expected_kernels.kernel_headers)

    def test_kernel_header_match(self, traced_kernels, expected_kernels):
        for trace in traced_kernels:
            sig_hash = self.extract_sig_hash(trace)
            expected_header = [
                h for h in expected_kernels.kernel_headers if sig_hash in str(h)
            ][0].read_text()
            actual_header = trace.header
            skip = self.get_skips(trace)
            check_codegen(
                actual_header,
                expected_header,
                skip=list(skip.values()),
                verbose=True,
            )

    def test_kernel_source_match(self, traced_kernels, expected_kernels):
        for trace in traced_kernels:
            sig_hash = self.extract_sig_hash(trace)
            expected_source = [
                s for s in expected_kernels.kernel_sources if sig_hash in str(s)
            ][0].read_text()
            actual_source = trace.source
            skip = self.get_skips(trace)
            check_codegen(
                actual_source,
                expected_source,
                skip=list(skip.values()),
                verbose=True,
            )

    def test_linked_header_match(self, linked_traces, expected_kernels, link_skips):
        expected_header = expected_kernels.linked_header[0].read_text()
        actual_header = linked_traces.header

        check_codegen(
            actual_header,
            expected_header,
            skip=link_skips,
            verbose=True,
        )

    def test_linked_source_match(
        self, kernel_name, linked_traces, expected_kernels, link_skips
    ):
        expected_source = expected_kernels.linked_source[0].read_text()
        actual_source = linked_traces.source

        check_linked_source(
            actual_source,
            expected_source,
            skip=link_skips,
            verbose=True,
        )
