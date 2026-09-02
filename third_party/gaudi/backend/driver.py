# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import threading
import time

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase

from .artifact import GaudiKernelArtifactV1


_LAUNCH_ABI_V1_MINOR = 8
_LAUNCH_ABI_V2_MINOR = 0
_LAUNCH_TARGET = "gaudi2"
_KERNEL_GUID_V1 = "triton_gaudi2_v1"
_KERNEL_GUID_V2 = "triton_gaudi2_v2"


def artifact_directory() -> Path:
    configured = os.environ.get("TRITON_GAUDI_ARTIFACT_DIR")
    if configured:
        directory = Path(configured)
    else:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        directory = cache_root / "triton" / "gaudi2" / "artifacts"
        os.environ["TRITON_GAUDI_ARTIFACT_DIR"] = str(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("TRITON_GAUDI_ARTIFACT_DIR must be a real directory")
    if directory.stat().st_uid != os.getuid():
        raise RuntimeError("TRITON_GAUDI_ARTIFACT_DIR must be owned by the current user")
    directory.chmod(0o700)
    return directory


def perf_library_path() -> Path:
    configured = os.environ.get("TRITON_GAUDI_PERF_LIB")
    if configured:
        return Path(configured)
    return Path(__file__).parent / "lib" / "libtriton_gaudi_perf.so"


def prepare_environment() -> Path:
    """Prepare the fixed GUID perf library before Synapse initialization."""
    artifact_directory()
    library = perf_library_path()
    if not library.is_file():
        raise RuntimeError(
            f"Triton Gaudi perf library was not found at {library}; install a Gaudi-enabled Triton wheel")
    current = [entry for entry in os.environ.get("GC_KERNEL_PATH", "").split(":") if entry]
    if str(library) not in current:
        os.environ["GC_KERNEL_PATH"] = ":".join([str(library), *current])
    return library


def _bridge_module():
    try:
        from habana_frameworks.torch import _hpu_C
    except ImportError as exc:
        raise RuntimeError("the Triton Gaudi backend requires the SynapseAI PyTorch Bridge") from exc
    return _hpu_C


def validate_bridge_launch_abi(bridge=None) -> dict:
    """Fail closed when Triton and the installed Bridge cannot interoperate."""
    bridge = bridge or _bridge_module()
    if not hasattr(bridge, "_triton_gaudi_launch_abi"):
        raise RuntimeError(
            "Gaudi PyTorch Bridge is missing the Triton launch ABI: "
            "_triton_gaudi_launch_abi. Install the matching "
            "triton-gaudi2 Bridge wheel.")
    description = dict(bridge._triton_gaudi_launch_abi())
    major = description.get("major")

    # ArtifactV1 remains a deliberately supported compatibility surface on a
    # v2 Bridge. This backend uses it until its compiler emits the v2 envelope;
    # validate the advertised v2 contract as well as the legacy entry points
    # that will actually be called.
    required = (
        "_triton_gaudi_register_artifact",
        "_triton_gaudi_unregister_artifact",
        "_triton_gaudi_launch",
    )
    if major == 1:
        expected = {
            "major": 1,
            "target": _LAUNCH_TARGET,
            "kernel_guid": _KERNEL_GUID_V1,
            "graph_op": True,
        }
        minimum_minor = _LAUNCH_ABI_V1_MINOR
    elif major == 2:
        expected = {
            "major": 2,
            "target": _LAUNCH_TARGET,
            "kernel_guid": _KERNEL_GUID_V2,
            "graph_op": True,
            "artifact_abi": 2,
            "typed_scalars": True,
        }
        minimum_minor = _LAUNCH_ABI_V2_MINOR
    else:
        raise RuntimeError(
            f"incompatible Gaudi Triton launch ABI: major={major!r}")
    missing = [name for name in required if not hasattr(bridge, name)]
    if missing:
        raise RuntimeError(
            "Gaudi PyTorch Bridge is missing the Triton launch ABI: " + ", ".join(missing) +
            ". Install the matching triton-gaudi2 Bridge wheel.")
    mismatches = {
        name: (description.get(name), value)
        for name, value in expected.items()
        if description.get(name) != value
    }
    if not isinstance(description.get("minor"), int) or description["minor"] < minimum_minor:
        mismatches["minor"] = (description.get("minor"), f">={minimum_minor}")
    if mismatches:
        detail = ", ".join(
            f"{name}={actual!r} (expected {wanted!r})"
            for name, (actual, wanted) in sorted(mismatches.items()))
        raise RuntimeError(f"incompatible Gaudi Triton launch ABI: {detail}")
    return description


class GaudiUtils:

    def __init__(self):
        self._bridge = _bridge_module()
        self._launch_abi = None
        self._artifacts = {}
        self._artifact_lock = threading.Lock()

    def _require_launch_abi(self):
        if self._launch_abi is not None:
            return self._launch_abi
        with self._artifact_lock:
            if self._launch_abi is None:
                self._launch_abi = validate_bridge_launch_abi(self._bridge)
            return self._launch_abi

    def get_device_properties(self, device):
        properties = {"max_shared_mem": 0}
        getter = getattr(self._bridge, "_triton_gaudi_device_properties", None)
        if getter is not None:
            properties.update(getter(device))
        return properties

    def load_binary(self, name, artifact_bytes, shared, device):
        self._require_launch_abi()
        artifact = GaudiKernelArtifactV1.from_bytes(artifact_bytes)
        manifest_json = json.dumps(artifact.manifest, sort_keys=True, separators=(",", ":"))
        handle = self._bridge._triton_gaudi_register_artifact(
            artifact.artifact_hash,
            artifact.elf,
            manifest_json,
            device,
        )
        with self._artifact_lock:
            registered = self._artifacts.get(handle)
            if registered is None:
                self._artifacts[handle] = (artifact, 1)
            else:
                previous, references = registered
                if previous.artifact_hash != artifact.artifact_hash:
                    self._bridge._triton_gaudi_unregister_artifact(handle)
                    raise RuntimeError("Gaudi Bridge reused an artifact handle for a different kernel")
                self._artifacts[handle] = (previous, references + 1)
        # CompiledKernel's tuple is GPU-shaped.  The Gaudi resource validator
        # deliberately ignores register/spill/thread fields.
        return handle, handle, 0, 0, 0

    def unload_module(self, handle):
        with self._artifact_lock:
            registered = self._artifacts.get(handle)
            if registered is None:
                return
            self._bridge._triton_gaudi_unregister_artifact(handle)
            artifact, references = registered
            if references == 1:
                del self._artifacts[handle]
            else:
                self._artifacts[handle] = (artifact, references - 1)

    def launch(self, handle, grid, stream, arguments):
        self._require_launch_abi()
        with self._artifact_lock:
            registered = self._artifacts.get(handle)
        if registered is None:
            raise RuntimeError("unknown or released Triton Gaudi artifact handle")
        artifact, _ = registered
        tensors = []
        scalar_params = []
        for spec in artifact.manifest["arguments"]:
            value = arguments[spec["index"]]
            if spec["kind"] == "tensor":
                tensors.append(value)
            elif spec["dtype"] == "i32":
                scalar_params.append(struct.unpack("<I", struct.pack("<i", int(value)))[0])
            elif spec["dtype"] == "u32":
                scalar_params.append(int(value) & 0xFFFFFFFF)
            elif spec["dtype"] == "f32":
                scalar_params.append(struct.unpack("<I", struct.pack("<f", float(value)))[0])
            else:
                raise RuntimeError(f"unsupported Gaudi scalar ABI type: {spec['dtype']}")
        self._bridge._triton_gaudi_launch(
            handle,
            list(grid),
            int(stream),
            tensors,
            scalar_params,
        )


class GaudiLauncher:

    def __init__(self, source, metadata):
        self.metadata = metadata

    def __call__(self, grid_x, grid_y, grid_z, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                 launch_exit_hook, *arguments):
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)
        try:
            from triton.runtime.driver import driver
            driver.active.utils.launch(function, (grid_x, grid_y, grid_z), stream, arguments)
        finally:
            if launch_exit_hook is not None:
                launch_exit_hook(launch_metadata)


class GaudiDriver(DriverBase):

    def __init__(self):
        prepare_environment()
        self.utils = GaudiUtils()
        self.launcher_cls = GaudiLauncher

    @staticmethod
    def is_active():
        if os.environ.get("TRITON_GAUDI_FORCE_ACTIVE") == "1":
            return True
        try:
            import torch
            return hasattr(torch, "hpu") and torch.hpu.is_available()
        except (ImportError, RuntimeError):
            return False

    def get_current_target(self):
        return GPUTarget("gaudi", "gaudi2")

    def get_current_device(self):
        import torch
        return torch.hpu.current_device()

    def set_current_device(self, device):
        import torch
        torch.hpu.set_device(device)

    def get_current_stream(self, device):
        getter = getattr(self.utils._bridge, "_hpu_getCurrentRawStream", None)
        if getter is None:
            raise RuntimeError("Gaudi Bridge does not expose the current HPU stream")
        return getter(device)

    def get_active_torch_device(self):
        import torch
        return torch.device("hpu", self.get_current_device())

    def map_python_to_cpp_type(self, dtype: str) -> str:
        if dtype.startswith("*"):
            return "uint64_t"
        return {
            "i1": "int8_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u1": "uint8_t",
            "u8": "uint8_t",
            "u16": "uint16_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "double",
            "bf16": "double",
            "fp32": "double",
            "f32": "double",
            "fp64": "double",
        }[dtype]

    def get_benchmarker(self):

        def benchmark(kernel_call, *, quantiles, warmup=5, rep=20, **kwargs):
            import torch
            for _ in range(warmup):
                kernel_call()
            torch.hpu.synchronize()
            samples = []
            for _ in range(rep):
                start = time.perf_counter_ns()
                kernel_call()
                torch.hpu.synchronize()
                samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
            ordered = sorted(samples)
            return [ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))] for q in quantiles]

        return benchmark

    def allocate_default_profile_scratch(self, size: int, alignment: int, stream):
        import torch
        return torch.empty(size, dtype=torch.int8, device=self.get_active_torch_device())
