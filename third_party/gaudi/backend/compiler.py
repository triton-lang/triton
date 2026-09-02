# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass
import functools
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import ModuleType
from typing import Any, Dict, Literal

from triton._C.libtriton import ir, passes
from triton.backends.compiler import BaseBackend, GPUTarget, Language

from .artifact import GaudiKernelArtifactV1
from .lowering import GaudiProgram, TpcCSource, emit_tpc_c, lower_ttir


@dataclass(frozen=True)
class GaudiConfig:
    """Native scheduling knobs for Gaudi2 TPC code generation."""

    unroll: int = 1
    pipeline_depth: int = 1
    vector_width_bits: int = 2048
    vlm_budget_bytes: int = 0
    engine: Literal["auto", "tpc", "mme"] = "auto"
    mode: Literal["strict", "hybrid"] = "strict"
    codegen: Literal["tpc-c", "tpc-llvm"] = "tpc-c"

    def __post_init__(self):
        if self.unroll <= 0:
            raise ValueError("GaudiConfig.unroll must be positive")
        if self.pipeline_depth <= 0:
            raise ValueError("GaudiConfig.pipeline_depth must be positive")
        if self.vector_width_bits != 2048:
            raise ValueError("Gaudi2 TPC vector width is fixed at 2048 bits")
        if self.vlm_budget_bytes < 0:
            raise ValueError("GaudiConfig.vlm_budget_bytes cannot be negative")
        if self.engine not in ("auto", "tpc", "mme"):
            raise ValueError("GaudiConfig.engine must be auto, tpc, or mme")
        if self.mode not in ("strict", "hybrid"):
            raise ValueError("GaudiConfig.mode must be strict or hybrid")
        if self.codegen not in ("tpc-c", "tpc-llvm"):
            raise ValueError("GaudiConfig.codegen must be tpc-c or tpc-llvm")

    def as_backend_options(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GaudiOptions:
    arch: str = "gaudi2"
    unroll: int = 1
    pipeline_depth: int = 1
    vector_width_bits: int = 2048
    vlm_budget_bytes: int = 0
    engine: str = "auto"
    mode: str = "strict"
    codegen: str = "tpc-c"
    optimization_level: int = 2
    tpc_clang: str | None = None
    ir_override: str | None = None
    debug: bool = False
    instrumentation_mode: str = ""
    fpsan_homomorphic_casts: bool = False
    sanitize_overflow: bool = True
    supported_fp8_dtypes: tuple[str, ...] = ("fp8e4nv", "fp8e5")
    deprecated_fp8_dot_operand_dtypes: tuple[str, ...] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: tuple[str, ...] = ("ieee",)
    max_num_imprecise_acc_default: int = 0
    enable_fp_fusion: bool = True

    def __post_init__(self):
        GaudiConfig(
            unroll=self.unroll,
            pipeline_depth=self.pipeline_depth,
            vector_width_bits=self.vector_width_bits,
            vlm_budget_bytes=self.vlm_budget_bytes,
            engine=self.engine,
            mode=self.mode,
            codegen=self.codegen,
        )
        if self.arch != "gaudi2":
            raise ValueError("the initial Triton Gaudi backend supports Gaudi2 only")
        if self.optimization_level not in (0, 1, 2):
            raise ValueError(
                "optimization_level must be between 0 and 2; "
                "tpc-clang -O3 is disabled for the Gaudi2 backend")
        if self.engine == "mme":
            raise ValueError("MME graph partitioning is not enabled in the initial Gaudi2 backend slice")
        if self.codegen == "tpc-llvm":
            raise ValueError("tpc-llvm is reserved for the LLVM 14-compatible lowering and is not enabled yet")
        if self.instrumentation_mode:
            raise ValueError("Triton instrumentation modes are not implemented for Gaudi2")

    def hash(self) -> str:
        return hashlib.sha256(repr(sorted(asdict(self).items())).encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=8)
def _tool_version(executable: str) -> str:
    try:
        return subprocess.check_output([executable, "--version"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"unable to execute Gaudi TPC compiler `{executable}`") from exc


class GaudiBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "gaudi" and str(target.arch) == "gaudi2"

    def __init__(self, target: GPUTarget) -> None:
        if target.warp_size is not None:
            raise ValueError("Gaudi targets do not define a warp_size")
        super().__init__(target)
        self.binary_ext = "gabin"

    def normalize_options(self, options: dict) -> dict:
        normalized = super().normalize_options(options)
        gpu_only = {
            name: normalized[name]
            for name in ("num_warps", "num_ctas", "num_stages", "maxnreg")
            if name in normalized and normalized[name] is not None
        }
        if gpu_only:
            names = ", ".join(sorted(gpu_only))
            raise ValueError(
                f"Gaudi2 does not accept SIMT launch options ({names}); use GaudiConfig/backend_options instead")
        for name in ("num_warps", "num_ctas", "num_stages", "maxnreg"):
            normalized.pop(name, None)
        return normalized

    def parse_options(self, opts) -> GaudiOptions:
        values = {"arch": str(self.target.arch)}
        values.update({
            name: opts[name]
            for name in GaudiOptions.__dataclass_fields__
            if name in opts and opts[name] is not None
        })
        return GaudiOptions(**values)

    def get_launch_options(self, options: GaudiOptions) -> dict:
        return {
            "engine": options.engine,
            "unroll": options.unroll,
            "pipeline_depth": options.pipeline_depth,
            "vlm_budget_bytes": options.vlm_budget_bytes,
            "mode": options.mode,
            "codegen": options.codegen,
        }

    def pack_metadata(self, metadata):
        return (
            metadata.artifact_hash,
            metadata.index_space_rank,
            metadata.block_size,
        )

    def get_codegen_implementation(self, options):
        return {"min_dot_size": lambda lhs, rhs: (1, 1, 1)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    def load_dialects(self, context):
        # GaudiTile is serialized between the optimized TTIR and TPC stages in
        # M1.  The native dialect registration is introduced with the C++
        # engine-partition/access-pattern passes.
        return None

    @staticmethod
    def make_ttir(module, metadata, options):
        pm = ir.pass_manager(module.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(module, "make_gaudi_ttir")
        return module

    @staticmethod
    def make_gtir(module, metadata, options) -> GaudiProgram:
        program = lower_ttir(module)
        vlm_bytes = int((program.parameters or {}).get("vlm_bytes", 0))
        if options.vlm_budget_bytes and vlm_bytes > options.vlm_budget_bytes:
            raise ValueError(
                f"Gaudi2 kernel requires {vlm_bytes} bytes of TPC VLM, "
                f"exceeding the configured {options.vlm_budget_bytes}-byte budget")
        metadata.update({
            "name": program.name,
            "shared": 0,
            "vlm_bytes": vlm_bytes,
            "engine": program.engine,
            "index_space_rank": program.index_space_rank,
            "block_size": program.block_size,
            "vector_width_bits": options.vector_width_bits,
        })
        return program

    @staticmethod
    def make_tpc_c(program: GaudiProgram, metadata, options) -> TpcCSource:
        if options.codegen != "tpc-c":
            raise ValueError(f"unsupported Gaudi codegen path: {options.codegen}")
        return emit_tpc_c(program)

    @staticmethod
    def make_gabin(source: TpcCSource, metadata, options) -> bytes:
        executable = options.tpc_clang or os.environ.get("TRITON_GAUDI_TPC_CLANG") or shutil.which("tpc-clang")
        if not executable:
            raise RuntimeError("tpc-clang was not found; install SynapseAI 1.24.1 or set TRITON_GAUDI_TPC_CLANG")
        _tool_version(executable)
        with tempfile.TemporaryDirectory(prefix="triton-gaudi-") as directory:
            input_path = Path(directory) / "kernel.c"
            output_path = Path(directory) / "kernel.o"
            input_path.write_text(source.source)
            command = [
                executable,
                "-Wall",
                "-Werror",
                f"-march={options.arch}",
                f"-O{options.optimization_level}",
                "-c",
                str(input_path),
                "-o",
                str(output_path),
            ]
            process = subprocess.run(command, text=True, capture_output=True)
            if process.returncode != 0:
                detail = process.stderr.strip() or process.stdout.strip()
                raise RuntimeError(f"tpc-clang failed for {source.program.name}:\n{detail}")
            elf = output_path.read_bytes()

        manifest = {
            **source.program.manifest(),
            "compiler": {
                "name": "tpc-clang",
                "version": _tool_version(executable),
                "optimization_level": options.optimization_level,
                "codegen": options.codegen,
            },
        }
        artifact = GaudiKernelArtifactV1.create(manifest, elf)
        metadata["artifact_hash"] = artifact.artifact_hash
        metadata["elf_size"] = len(elf)
        return artifact.to_bytes()

    def add_stages(self, stages, options, language):
        if language != Language.TRITON:
            raise ValueError("the Gaudi2 backend currently accepts Triton TTIR only")
        stages["ttir"] = lambda source, metadata: self.make_ttir(source, metadata, options)
        stages["gtir"] = lambda source, metadata: self.make_gtir(source, metadata, options)
        stages["tpc_c"] = lambda source, metadata: self.make_tpc_c(source, metadata, options)
        stages["gabin"] = lambda source, metadata: self.make_gabin(source, metadata, options)

    @functools.lru_cache()
    def hash(self) -> str:
        executable = os.environ.get("TRITON_GAUDI_TPC_CLANG") or shutil.which("tpc-clang")
        tool = _tool_version(executable) if executable else "tpc-clang-missing"
        sources = []
        for name in ("artifact.py", "compiler.py", "lowering.py"):
            sources.append(hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest())
        return hashlib.sha256(f"gaudi2-v1:{tool}:{':'.join(sources)}".encode("utf-8")).hexdigest()

    def validate_kernel_resources(self, metadata, device_properties: dict, *, n_max_threads=None) -> None:
        vlm_bytes = getattr(metadata, "vlm_bytes", 0)
        max_vlm = device_properties.get("max_vlm_bytes")
        if max_vlm is not None and vlm_bytes > max_vlm:
            from triton.runtime.autotuner import OutOfResources
            raise OutOfResources(vlm_bytes, max_vlm, "TPC VLM")
