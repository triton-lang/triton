#!/usr/bin/env python3

"""Compare Proton memory growth across CUPTI library variants.

Examples:

  Compare Triton's packaged generic and Blackwell CUPTI builds in a clean
  process that only loads the selected ``libcupti.so``:

    python third_party/proton/tutorials/cupti_memory_growth_cuda_core.py \
      --output-dir /tmp/proton-cuda-core-direct-200x32 \
      --iterations 200 \
      --warmup 5 \
      --phase-every 1 \
      --sample-every 20 \
      --lifecycle step \
      --kernels-per-step 32 \
      --clear-completed-phases \
      --workload direct

  Reproduce the CUDA-graph warning path by capturing the graph before Proton
  starts:

    python third_party/proton/tutorials/cupti_memory_growth_cuda_core.py \
      --output-dir /tmp/proton-cuda-core-graph-pre-200x32 \
      --iterations 200 \
      --warmup 5 \
      --phase-every 1 \
      --sample-every 20 \
      --lifecycle step \
      --kernels-per-step 32 \
      --clear-completed-phases \
      --workload graph \
      --capture-before-start

This variant drives CUDA work through ``cuda.core.experimental`` and controls
Proton through ``triton._C.libproton`` directly so the process can stay free of
framework-bundled CUPTI libraries.

  Reproduce the same graph-warning path in a process that preloads Torch before
  Proton starts:

    python third_party/proton/tutorials/cupti_memory_growth_cuda_core.py \
      --output-dir /tmp/proton-cuda-core-torch-graph-1000x32 \
      --iterations 1000 \
      --warmup 5 \
      --phase-every 1 \
      --sample-every 100 \
      --lifecycle step \
      --kernels-per-step 32 \
      --clear-completed-phases \
      --workload graph \
      --capture-before-start \
      --post-finalize-sleep-ms 1000 \
      --preload-torch
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a standalone CUDA-core + Proton workload and record process memory "
            "growth for one or more CUPTI library directories."
        )
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run only one CUPTI configuration in the current process.",
    )
    parser.add_argument(
        "--label",
        default="run",
        help="Label used for the output files in --single-run mode.",
    )
    parser.add_argument(
        "--cupti-dir",
        default=None,
        help=(
            "CUPTI directory to force for this run. When set, both "
            "TRITON_CUPTI_LIB_PATH and TRITON_CUPTI_LIB_BLACKWELL_PATH are set "
            "to this directory before libproton is imported."
        ),
    )
    parser.add_argument(
        "--generic-cupti-dir",
        default=None,
        help="Override the Triton-packaged generic CUPTI directory for compare mode.",
    )
    parser.add_argument(
        "--blackwell-cupti-dir",
        default=None,
        help="Override the Triton-packaged Blackwell CUPTI directory for compare mode.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/proton-cupti-memory-growth-cuda-core",
        help="Directory for CSV, JSON, and Proton profile outputs.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of profiled iterations to execute per run.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of unprofiled warmup iterations before measurements begin.",
    )
    parser.add_argument(
        "--phase-every",
        type=int,
        default=1,
        help="Advance Proton data phase every N iterations.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=20,
        help="Capture memory samples every N iterations.",
    )
    parser.add_argument(
        "--numel",
        type=int,
        default=1 << 20,
        help="Number of float32 elements in the device buffer.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="CUDA block size for the add kernel.",
    )
    parser.add_argument(
        "--kernels-per-step",
        type=int,
        default=32,
        help="Number of kernel launches or graph replays to issue inside each profiled step.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index to use.",
    )
    parser.add_argument(
        "--data-format",
        default="hatchet_msgpack",
        choices=["hatchet", "hatchet_msgpack", "chrome_trace"],
        help="Periodic flushing output format.",
    )
    parser.add_argument(
        "--lifecycle",
        default="step",
        choices=["continuous", "step"],
        help=(
            "How to drive the Proton session. 'step' repeatedly activates, "
            "records one profiled iteration, then deactivates before the next "
            "phase advance."
        ),
    )
    parser.add_argument(
        "--clear-completed-phases",
        action="store_true",
        help="Clear completed Proton phases as the run progresses.",
    )
    parser.add_argument(
        "--sleep-ms",
        type=float,
        default=0.0,
        help="Optional sleep between iterations to slow the workload down.",
    )
    parser.add_argument(
        "--post-finalize-sleep-ms",
        type=float,
        default=0.0,
        help="Optional sleep after finalize before collecting a settled RSS sample.",
    )
    parser.add_argument(
        "--workload",
        default="direct",
        choices=["direct", "graph"],
        help=(
            "CUDA workload shape. 'direct' launches kernels individually. "
            "'graph' captures a CUDA graph once and replays it."
        ),
    )
    parser.add_argument(
        "--capture-before-start",
        action="store_true",
        help="When --workload=graph, capture the graph before Proton starts.",
    )
    parser.add_argument(
        "--preload-torch",
        action="store_true",
        help=(
            "Import torch before libproton so the process also maps any framework-bundled "
            "libcupti DSOs (for example, torch/lib/libcupti.so.13)."
        ),
    )
    return parser


def _read_proc_status_bytes(field_name: str) -> int | None:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    prefix = f"{field_name}:"
    for line in status_path.read_text().splitlines():
        if not line.startswith(prefix):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        return int(parts[1]) * 1024
    return None


def _query_nvidia_smi_process_memory_mb(pid: int) -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        if parts[0] != str(pid):
            continue
        try:
            return int(parts[1])
        except ValueError:
            return None
    return 0


def _read_loaded_shared_objects(needle: str, *, exact_basename: bool = False) -> list[str]:
    maps_path = Path("/proc/self/maps")
    if not maps_path.exists():
        return []

    loaded: list[str] = []
    seen: set[str] = set()
    for line in maps_path.read_text().splitlines():
        if needle not in line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        path = parts[-1]
        if exact_basename and Path(path).name != needle:
            continue
        if path in seen:
            continue
        seen.add(path)
        loaded.append(path)
    return loaded


def _maybe_preload_torch(enabled: bool) -> dict[str, Any] | None:
    if not enabled:
        return None

    torch = importlib.import_module("torch")
    return {
        "version": getattr(torch, "__version__", None),
        "file": str(Path(torch.__file__).resolve()) if getattr(torch, "__file__", None) else None,
    }


def _load_cupti_info(cupti_dir: Path) -> dict[str, Any]:
    lib_path = cupti_dir / "libcupti.so"
    info: dict[str, Any] = {
        "cupti_dir": str(cupti_dir),
        "lib_path": str(lib_path),
        "lib_exists": lib_path.exists(),
    }
    if not lib_path.exists():
        return info

    version = ctypes.c_uint32()
    cupti = ctypes.CDLL(str(lib_path))
    rc = int(cupti.cuptiGetVersion(ctypes.byref(version)))
    info["cupti_get_version_rc"] = rc
    info["cupti_version"] = int(version.value)
    info["libcupti_entries"] = sorted(path.name for path in cupti_dir.glob("libcupti.so*"))
    return info


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rss_values = [sample["rss_bytes"] for sample in samples if sample["rss_bytes"] is not None]
    gpu_mb_values = [
        sample["nvidia_smi_process_used_memory_mb"]
        for sample in samples
        if sample["nvidia_smi_process_used_memory_mb"] is not None
    ]
    return {
        "num_samples": len(samples),
        "rss_start_bytes": rss_values[0] if rss_values else None,
        "rss_end_bytes": rss_values[-1] if rss_values else None,
        "rss_max_bytes": max(rss_values) if rss_values else None,
        "rss_min_bytes": min(rss_values) if rss_values else None,
        "rss_delta_bytes": (rss_values[-1] - rss_values[0]) if len(rss_values) >= 2 else None,
        "nvidia_smi_gpu_start_mb": gpu_mb_values[0] if gpu_mb_values else None,
        "nvidia_smi_gpu_end_mb": gpu_mb_values[-1] if gpu_mb_values else None,
        "nvidia_smi_gpu_max_mb": max(gpu_mb_values) if gpu_mb_values else None,
        "nvidia_smi_gpu_delta_mb": (gpu_mb_values[-1] - gpu_mb_values[0]) if len(gpu_mb_values) >= 2 else None,
    }


def _write_csv(samples: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "timestamp_s",
        "iteration",
        "stage",
        "phase",
        "pid",
        "rss_bytes",
        "vms_bytes",
        "nvidia_smi_process_used_memory_mb",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def _configure_cupti_env(cupti_dir: str | None) -> None:
    os.environ.setdefault("PROTON_LAUNCH_METADATA_NOSYNC", "1")
    if cupti_dir is None:
        os.environ.pop("TRITON_CUPTI_LIB_PATH", None)
        os.environ.pop("TRITON_CUPTI_LIB_BLACKWELL_PATH", None)
        return

    os.environ["TRITON_CUPTI_LIB_PATH"] = cupti_dir
    os.environ["TRITON_CUPTI_LIB_BLACKWELL_PATH"] = cupti_dir


def _collect_sample(iteration: int | None, stage: str, phase: int, stream: Any) -> dict[str, Any]:
    stream.sync()
    return {
        "timestamp_s": time.time(),
        "iteration": iteration,
        "stage": stage,
        "phase": phase,
        "pid": os.getpid(),
        "rss_bytes": _read_proc_status_bytes("VmRSS"),
        "vms_bytes": _read_proc_status_bytes("VmSize"),
        "nvidia_smi_process_used_memory_mb": _query_nvidia_smi_process_memory_mb(os.getpid()),
    }


def _detect_packaged_cupti_dirs() -> tuple[Path, Path]:
    triton = importlib.import_module("triton")
    root = Path(triton.__file__).resolve().parent / "backends" / "nvidia" / "lib"
    return root / "cupti", root / "cupti-blackwell"


def _compile_kernel(device: Any) -> tuple[Any, Any]:
    from cuda.core.experimental import Program, ProgramOptions

    compute_capability = device.compute_capability
    arch = f"sm_{compute_capability[0]}{compute_capability[1]}"
    code = r"""
extern "C" __global__ void add1(float *x, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    x[i] += 1.0f;
  }
}
"""
    program = Program(code, "c++", ProgramOptions(arch=arch, std="c++17", name="cupti_memory_growth_cuda_core.cu"))
    object_code = program.compile("cubin")
    return program, object_code.get_kernel("add1")


def _build_graph(stream: Any, kernel: Any, buffer: Any, numel: int, block_size: int, kernels_per_step: int) -> Any:
    from cuda.core.experimental import LaunchConfig, launch

    grid = max(1, (numel + block_size - 1) // block_size)
    builder = stream.create_graph_builder()
    builder.begin_building()
    for _ in range(kernels_per_step):
        launch(stream, LaunchConfig(grid=grid, block=block_size), kernel, buffer, numel)
    builder.end_building()
    graph = builder.complete()
    stream.sync()
    return graph


def _run_workload_iteration(
    *,
    workload: str,
    stream: Any,
    kernel: Any,
    buffer: Any,
    numel: int,
    block_size: int,
    kernels_per_step: int,
    graph: Any,
) -> None:
    from cuda.core.experimental import LaunchConfig, launch

    if workload == "graph":
        for _ in range(kernels_per_step):
            graph.launch(stream)
        return

    grid = max(1, (numel + block_size - 1) // block_size)
    for _ in range(kernels_per_step):
        launch(stream, LaunchConfig(grid=grid, block=block_size), kernel, buffer, numel)


def _compare_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    comparison: dict[str, Any] = {"runs": summaries}
    if len(summaries) < 2:
        return comparison

    base = summaries[0]
    other = summaries[1]
    for key in ("rss_delta_bytes", "rss_max_bytes", "nvidia_smi_gpu_delta_mb"):
        base_value = base["samples"][key]
        other_value = other["samples"][key]
        comparison[key] = {
            "base_label": base["label"],
            "other_label": other["label"],
            "base": base_value,
            "other": other_value,
            "difference": None if base_value is None or other_value is None else other_value - base_value,
        }

    base_loaded = base.get("loaded_cupti_libs_after_start", [])
    other_loaded = other.get("loaded_cupti_libs_after_start", [])
    comparison["loaded_cupti_libs_after_start"] = {
        "base_label": base["label"],
        "other_label": other["label"],
        "base": base_loaded,
        "other": other_loaded,
        "same": sorted(base_loaded) == sorted(other_loaded),
    }
    if comparison["loaded_cupti_libs_after_start"]["same"]:
        comparison["warning"] = (
            "Both runs resolved the same loaded libcupti.so path(s). "
            "Verify the target machine actually switched CUPTI variants."
        )

    base_loaded_all = base.get("loaded_libcupti_objects_after_start", [])
    other_loaded_all = other.get("loaded_libcupti_objects_after_start", [])
    comparison["loaded_libcupti_objects_after_start"] = {
        "base_label": base["label"],
        "other_label": other["label"],
        "base": base_loaded_all,
        "other": other_loaded_all,
        "same": sorted(base_loaded_all) == sorted(other_loaded_all),
    }
    return comparison


def _run_single(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_cupti_env(args.cupti_dir)
    preloaded_torch = _maybe_preload_torch(args.preload_torch)

    triton = importlib.import_module("triton")
    proton_data = importlib.import_module("triton.profiler.data")
    from triton._C.libproton import proton as libproton
    from cuda.core.experimental import Device

    device = Device(args.device)
    device.set_current()
    stream = device.create_stream()
    _program, kernel = _compile_kernel(device)
    buffer = device.allocate(args.numel * 4)

    # Warm up module load and kernel launch before starting Proton.
    for _ in range(args.warmup):
        _run_workload_iteration(
            workload="direct",
            stream=stream,
            kernel=kernel,
            buffer=buffer,
            numel=args.numel,
            block_size=args.block_size,
            kernels_per_step=1,
            graph=None,
        )
    stream.sync()

    graph = None
    if args.workload == "graph" and args.capture_before_start:
        graph = _build_graph(stream, kernel, buffer, args.numel, args.block_size, kernels_per_step=1)

    current_phase = 0
    last_cleared_phase = -1
    profile_base = output_dir / f"profile_{args.label}"
    profile_mode = f"periodic_flushing:format={args.data_format}"

    selected_cupti_dir = Path(args.cupti_dir) if args.cupti_dir is not None else None
    if selected_cupti_dir is None:
        generic_dir, blackwell_dir = _detect_packaged_cupti_dirs()
        selected_cupti_dir = blackwell_dir if device.compute_capability[0] >= 10 else generic_dir

    cupti_info = _load_cupti_info(selected_cupti_dir)
    run_metadata = {
        "label": args.label,
        "python_executable": sys.executable,
        "pid": os.getpid(),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "phase_every": args.phase_every,
        "sample_every": args.sample_every,
        "numel": args.numel,
        "block_size": args.block_size,
        "kernels_per_step": args.kernels_per_step,
        "data_format": args.data_format,
        "lifecycle": args.lifecycle,
        "clear_completed_phases": bool(args.clear_completed_phases),
        "sleep_ms": args.sleep_ms,
        "post_finalize_sleep_ms": args.post_finalize_sleep_ms,
        "workload": args.workload,
        "capture_before_start": bool(args.capture_before_start),
        "preload_torch": bool(args.preload_torch),
        "preloaded_torch": preloaded_torch,
        "triton_version_file": str(Path(triton.__file__).resolve()),
        "device_id": args.device,
        "device_name": device.name,
        "device_compute_capability": list(device.compute_capability),
        "selected_cupti": cupti_info,
        "env": {
            "TRITON_CUPTI_LIB_PATH": os.environ.get("TRITON_CUPTI_LIB_PATH"),
            "TRITON_CUPTI_LIB_BLACKWELL_PATH": os.environ.get("TRITON_CUPTI_LIB_BLACKWELL_PATH"),
            "TRITON_PROFILE_BUFFER_SIZE": os.environ.get("TRITON_PROFILE_BUFFER_SIZE"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
        "loaded_cupti_libs_before_start": _read_loaded_shared_objects(
            "libcupti.so", exact_basename=True
        ),
        "loaded_libcupti_objects_before_start": _read_loaded_shared_objects("libcupti"),
    }

    samples: list[dict[str, Any]] = []
    samples.append(_collect_sample(iteration=None, stage="pre_start", phase=current_phase, stream=stream))
    session = libproton.start(str(profile_base), "shadow", "tree", "cupti", profile_mode)
    samples.append(_collect_sample(iteration=None, stage="post_start", phase=current_phase, stream=stream))
    run_metadata["loaded_cupti_libs_after_start"] = _read_loaded_shared_objects(
        "libcupti.so", exact_basename=True
    )
    run_metadata["loaded_libcupti_objects_after_start"] = _read_loaded_shared_objects("libcupti")

    if args.workload == "graph" and not args.capture_before_start:
        graph = _build_graph(stream, kernel, buffer, args.numel, args.block_size, kernels_per_step=1)
        samples.append(_collect_sample(iteration=None, stage="post_graph_capture", phase=current_phase, stream=stream))

    if args.lifecycle == "step":
        libproton.deactivate(session, False)
        samples.append(_collect_sample(iteration=None, stage="post_initial_deactivate", phase=current_phase, stream=stream))

    for iteration in range(args.iterations):
        if args.lifecycle == "step":
            libproton.activate(session)

        _run_workload_iteration(
            workload=args.workload,
            stream=stream,
            kernel=kernel,
            buffer=buffer,
            numel=args.numel,
            block_size=args.block_size,
            kernels_per_step=args.kernels_per_step,
            graph=graph,
        )

        if args.phase_every > 0 and (iteration + 1) % args.phase_every == 0:
            current_phase = int(proton_data.advance_phase(session))

        if args.lifecycle == "step":
            libproton.deactivate(session, False)

        if args.phase_every > 0 and (iteration + 1) % args.phase_every == 0:
            if args.clear_completed_phases and current_phase > 0:
                clear_phase = current_phase - 1
                if clear_phase > last_cleared_phase and proton_data.is_phase_complete(session, clear_phase):
                    proton_data.clear(session, phase=clear_phase, clear_up_to_phase=True)
                    last_cleared_phase = clear_phase

        if iteration == 0 or (iteration + 1) % args.sample_every == 0 or iteration + 1 == args.iterations:
            samples.append(_collect_sample(iteration=iteration + 1, stage="loop", phase=current_phase, stream=stream))

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    libproton.finalize(session, args.data_format)
    samples.append(_collect_sample(iteration=args.iterations, stage="post_finalize", phase=current_phase, stream=stream))
    if args.post_finalize_sleep_ms > 0:
        time.sleep(args.post_finalize_sleep_ms / 1000.0)
        samples.append(
            _collect_sample(
                iteration=args.iterations,
                stage="post_finalize_settled",
                phase=current_phase,
                stream=stream,
            )
        )
    run_metadata["loaded_cupti_libs_after_finalize"] = _read_loaded_shared_objects(
        "libcupti.so", exact_basename=True
    )
    run_metadata["loaded_libcupti_objects_after_finalize"] = _read_loaded_shared_objects("libcupti")

    sample_path = output_dir / f"samples_{args.label}.csv"
    summary_path = output_dir / f"summary_{args.label}.json"
    _write_csv(samples, sample_path)

    summary = run_metadata | {
        "sample_csv": str(sample_path),
        "profile_base": str(profile_base),
        "summary_json": str(summary_path),
        "samples": _summarize_samples(samples),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary["samples"], sort_keys=True))
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generic_dir, blackwell_dir = _detect_packaged_cupti_dirs()
    if args.generic_cupti_dir is not None:
        generic_dir = Path(args.generic_cupti_dir)
    if args.blackwell_cupti_dir is not None:
        blackwell_dir = Path(args.blackwell_cupti_dir)

    runs = [
        ("generic", generic_dir),
        ("blackwell", blackwell_dir),
    ]

    summaries: list[dict[str, Any]] = []
    for label, cupti_dir in runs:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single-run",
            "--label",
            label,
            "--cupti-dir",
            str(cupti_dir),
            "--output-dir",
            str(output_dir),
            "--iterations",
            str(args.iterations),
            "--warmup",
            str(args.warmup),
            "--phase-every",
            str(args.phase_every),
            "--sample-every",
            str(args.sample_every),
            "--numel",
            str(args.numel),
            "--block-size",
            str(args.block_size),
            "--device",
            str(args.device),
            "--kernels-per-step",
            str(args.kernels_per_step),
            "--data-format",
            args.data_format,
            "--lifecycle",
            args.lifecycle,
            "--sleep-ms",
            str(args.sleep_ms),
            "--post-finalize-sleep-ms",
            str(args.post_finalize_sleep_ms),
            "--workload",
            args.workload,
        ]
        if args.clear_completed_phases:
            cmd.append("--clear-completed-phases")
        if args.capture_before_start:
            cmd.append("--capture-before-start")
        if args.preload_torch:
            cmd.append("--preload-torch")

        subprocess.run(cmd, check=True)
        summary_path = output_dir / f"summary_{label}.json"
        with summary_path.open(encoding="utf-8") as handle:
            summaries.append(json.load(handle))

    comparison = _compare_summaries(summaries)
    comparison_path = output_dir / "comparison.json"
    with comparison_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.single_run:
        return _run_single(args)
    return _run_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
