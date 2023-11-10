from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import triton
from triton.compiler.compiler import CompiledKernel
from triton.runtime.jit import JITFunction

from .. import DEFAULT_TRACE_DIR
from .codegen import AOT_C_CUDA_ParamsBuilder, AOTCompilerParamsBuilder, JITCompileArgs


class AOT_Compiler(ABC):
    """Interface for generating AOT kernels from `triton.runtime.jit.JITFunction`

    `generate_header` and `generate_source` should be implemented for header and source codegen.
    """

    PARAM_BUILDER_CLS: AOTCompilerParamsBuilder

    def __init__(
        self,
        kernel_name,
        jit_args: JITCompileArgs,
        jit_fn: JITFunction,
        save_dir: Path = None,
        compiled_binary: CompiledKernel = None,
    ):
        self.kernel_name = kernel_name
        self.jit_args = jit_args

        self.save_dir = save_dir or DEFAULT_TRACE_DIR
        self.compiled_binary = compiled_binary or triton.compile(jit_fn, **jit_args)

        self.params_builder = self.PARAM_BUILDER_CLS(
            kernel_name=kernel_name,
            compiled_binary=self.compiled_binary,
            jit_args=jit_args,
            jit_fn=jit_fn,
        )
        self.params = self.build_params()

    def build_params(self):
        return self.params_builder.build()

    @abstractmethod
    def generate_header(self):
        ...

    @abstractmethod
    def generate_source(self):
        ...


@dataclass
class AOTCompilationResult:
    kernel_name: str
    header: str
    source: str
    params: dict
    header_path: str | Path
    source_path: str | Path
    compiled_binary: CompiledKernel
    # For debugging
    _jit_args: JITCompileArgs
    _compiler_params: dict


class AOT_C_CUDA_Compiler(AOT_Compiler):
    """Creates C CUDA library for accessing Triton jitted kernels

    Refactor of `triton.tools.compile.py`
    """

    PARAM_BUILDER_CLS = AOT_C_CUDA_ParamsBuilder

    def generate_header(self):
        # Filter params for header keys
        header_params = {k: v for k, v in self.params.items() if k in self.params_builder.HEADER_TEMPLATE.PARAMS}
        # Generate header
        header = self.params_builder.HEADER_TEMPLATE.TEMPLATE.format(**header_params)
        return header

    def generate_source(self):
        # Filter params for source keys
        source_params = {k: v for k, v in self.params.items() if k in self.params_builder.SOURCE_TEMPLATE.PARAMS}
        # Generate source
        source = self.params_builder.SOURCE_TEMPLATE.TEMPLATE.format(**source_params)
        return source

    def generate(self):
        header = self.generate_header()
        source = self.generate_source()
        suffix = "_".join(f'{self.params["kernel_name"]}'.split("_")[1:])
        file_name = f"{self.kernel_name}.{suffix}"
        header_name = f"{file_name}.h"
        source_name = f"{file_name}.c"

        header_path = self.save_dir / header_name
        source_path = self.save_dir / source_name

        with open(header_path, "w") as fp:
            fp.write(header)
        with open(source_path, "w") as fp:
            fp.write(source)

        return AOTCompilationResult(
            kernel_name=self.kernel_name,
            header=header,
            source=source,
            params=self.params,
            header_path=self.save_dir / header_name,
            source_path=self.save_dir / source_name,
            compiled_binary=self.compiled_binary,
            _jit_args=self.jit_args,
            _compiler_params=self.params,
        )
