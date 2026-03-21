#!/usr/bin/env python3

"""Compare Proton worker memory growth across CUPTI library variants.

Examples:

  Compare Triton's packaged generic and Blackwell CUPTI builds:

    python third_party/proton/tutorials/cupti_memory_growth.py \
      --output-dir /tmp/proton-cupti-compare \
      --iterations 200 \
      --warmup 5 \
      --phase-every 1 \
      --sample-every 20 \
      --lifecycle step \
      --kernels-per-step 32 \
      --clear-completed-phases

    Inspect /tmp/proton-cupti-compare/comparison.json and compare the
    `rss_delta_bytes` / `rss_max_bytes` fields between the `generic` and
    `blackwell` runs. A CUPTI-specific host-memory issue should show up as
    larger RSS growth for one library variant while GPU memory stays flat.

  Run a single configuration with an explicit CUPTI directory:

    python third_party/proton/tutorials/cupti_memory_growth.py \
      --single-run \
      --label cupti12 \
      --cupti-dir /path/to/triton/backends/nvidia/lib/cupti \
      --output-dir /tmp/proton-cupti12
"""

from __future__ import annotations

import argparse
import csv
import ctypes
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
            "Run a standalone Triton + Proton workload and record process memory "
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
            "to this directory before Triton is imported."
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
        default="/tmp/proton-cupti-memory-growth",
        help="Directory for CSV, JSON, and Proton profile outputs.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20000,
        help="Number of profiled kernel launches to execute per run.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Number of unprofiled warmup launches before measurements begin.",
    )
    parser.add_argument(
        "--phase-every",
        type=int,
        default=1000,
        help="Advance Proton data phase every N iterations.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=200,
        help="Capture memory samples every N iterations.",
    )
    parser.add_argument(
        "--numel",
        type=int,
        default=1 << 20,
        help="Vector size for the Triton add kernel.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Block size for the Triton add kernel.",
    )
    parser.add_argument(
        "--kernels-per-step",
        type=int,
        default=1,
        help="Number of Triton kernel launches to issue inside each profiled step.",
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
            "records one profiled scope, then deactivates before the next "
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
        value_kib = int(parts[1])
        return value_kib * 1024
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
    sonames = sorted(path.name for path in cupti_dir.glob("libcupti.so*"))
    info["libcupti_entries"] = sonames
    return info


def _collect_sample(iteration: int | None, stage: str, phase: int, torch: Any) -> dict[str, Any]:
    rss_bytes = _read_proc_status_bytes("VmRSS")
    vms_bytes = _read_proc_status_bytes("VmSize")
    torch.cuda.synchronize()
    sample = {
        "timestamp_s": time.time(),
        "iteration": iteration,
        "stage": stage,
        "phase": phase,
        "pid": os.getpid(),
        "rss_bytes": rss_bytes,
        "vms_bytes": vms_bytes,
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "nvidia_smi_process_used_memory_mb": _query_nvidia_smi_process_memory_mb(os.getpid()),
    }
    return sample


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rss_values = [sample["rss_bytes"] for sample in samples if sample["rss_bytes"] is not None]
    gpu_mb_values = [
        sample["nvidia_smi_process_used_memory_mb"]
        for sample in samples
        if sample["nvidia_smi_process_used_memory_mb"] is not None
    ]
    summary: dict[str, Any] = {
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
    return summary


def _write_csv(samples: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "timestamp_s",
        "iteration",
        "stage",
        "phase",
        "pid",
        "rss_bytes",
        "vms_bytes",
        "cuda_memory_allocated_bytes",
        "cuda_memory_reserved_bytes",
        "cuda_max_memory_allocated_bytes",
        "cuda_max_memory_reserved_bytes",
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


def _run_single(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_cupti_env(args.cupti_dir)

    import torch
    import triton
    import triton.language as tl
    import triton.profiler as proton

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    @triton.jit
    def add_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        tl.store(z_ptr + offsets, x + y, mask=mask)

    numel = args.numel
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]), )
    x = torch.rand((numel,), device=device, dtype=torch.float32)
    y = torch.rand((numel,), device=device, dtype=torch.float32)
    z = torch.empty_like(x)

    for _ in range(args.warmup):
        add_kernel[grid](x, y, z, numel, BLOCK_SIZE=args.block_size)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    samples: list[dict[str, Any]] = []
    current_phase = 0
    last_cleared_phase = -1
    profile_base = output_dir / f"profile_{args.label}"
    profile_mode = f"periodic_flushing:format={args.data_format}"

    selected_cupti_dir = None
    if args.cupti_dir is not None:
        selected_cupti_dir = Path(args.cupti_dir)
    else:
        target = triton.runtime.driver.active.get_current_target()
        if target.backend == "cuda" and target.arch >= 100:
            selected_cupti_dir = Path(triton.knobs.proton.cupti_lib_blackwell_dir)
        else:
            selected_cupti_dir = Path(triton.knobs.proton.cupti_lib_dir)

    cupti_info = _load_cupti_info(selected_cupti_dir)
    run_metadata = {
        "label": args.label,
        "python_executable": sys.executable,
        "pid": os.getpid(),
        "device": str(device),
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
        "triton_version_file": str(Path(triton.__file__).resolve()),
        "triton_target_backend": triton.runtime.driver.active.get_current_target().backend,
        "triton_target_arch": triton.runtime.driver.active.get_current_target().arch,
        "selected_cupti": cupti_info,
        "env": {
            "TRITON_CUPTI_LIB_PATH": os.environ.get("TRITON_CUPTI_LIB_PATH"),
            "TRITON_CUPTI_LIB_BLACKWELL_PATH": os.environ.get("TRITON_CUPTI_LIB_BLACKWELL_PATH"),
            "TRITON_PROFILE_BUFFER_SIZE": os.environ.get("TRITON_PROFILE_BUFFER_SIZE"),
        },
    }

    samples.append(_collect_sample(iteration=None, stage="pre_start", phase=current_phase, torch=torch))
    session = proton.start(str(profile_base), backend="cupti", context="shadow", mode=profile_mode)
    samples.append(_collect_sample(iteration=None, stage="post_start", phase=current_phase, torch=torch))
    if args.lifecycle == "step":
        proton.deactivate(session=session)
        samples.append(
            _collect_sample(iteration=None, stage="post_initial_deactivate", phase=current_phase, torch=torch)
        )

    for iteration in range(args.iterations):
        if args.lifecycle == "step":
            proton.activate(session=session)

        with proton.scope(f"step_{iteration}" if args.lifecycle == "step" else "step"):
            for _ in range(args.kernels_per_step):
                add_kernel[grid](x, y, z, numel, BLOCK_SIZE=args.block_size)

        if args.phase_every > 0 and (iteration + 1) % args.phase_every == 0:
            current_phase = int(proton.data.advance_phase(session))

        if args.lifecycle == "step":
            proton.deactivate(session=session, flushing=False)

        if args.phase_every > 0 and (iteration + 1) % args.phase_every == 0:
            if args.clear_completed_phases and current_phase > 0:
                clear_phase = current_phase - 1
                if clear_phase > last_cleared_phase and proton.data.is_phase_complete(session, clear_phase):
                    proton.data.clear(session, phase=clear_phase, clear_up_to_phase=True)
                    last_cleared_phase = clear_phase

        if iteration == 0 or (iteration + 1) % args.sample_every == 0 or iteration + 1 == args.iterations:
            samples.append(
                _collect_sample(
                    iteration=iteration + 1,
                    stage="loop",
                    phase=current_phase,
                    torch=torch,
                )
            )

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    proton.finalize(output_format=args.data_format)
    samples.append(_collect_sample(iteration=args.iterations, stage="post_finalize", phase=current_phase, torch=torch))

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


def _detect_packaged_cupti_dirs() -> tuple[Path, Path]:
    import triton

    root = Path(triton.__file__).resolve().parent / "backends" / "nvidia" / "lib"
    return root / "cupti", root / "cupti-blackwell"


def _compare_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    comparison: dict[str, Any] = {"runs": summaries}
    if len(summaries) < 2:
        return comparison

    base = summaries[0]
    other = summaries[1]
    base_samples = base["samples"]
    other_samples = other["samples"]
    comparison["rss_delta_bytes"] = {
        "base_label": base["label"],
        "other_label": other["label"],
        "base": base_samples["rss_delta_bytes"],
        "other": other_samples["rss_delta_bytes"],
        "difference": None
        if base_samples["rss_delta_bytes"] is None or other_samples["rss_delta_bytes"] is None
        else other_samples["rss_delta_bytes"] - base_samples["rss_delta_bytes"],
    }
    comparison["rss_max_bytes"] = {
        "base_label": base["label"],
        "other_label": other["label"],
        "base": base_samples["rss_max_bytes"],
        "other": other_samples["rss_max_bytes"],
        "difference": None
        if base_samples["rss_max_bytes"] is None or other_samples["rss_max_bytes"] is None
        else other_samples["rss_max_bytes"] - base_samples["rss_max_bytes"],
    }
    comparison["nvidia_smi_gpu_delta_mb"] = {
        "base_label": base["label"],
        "other_label": other["label"],
        "base": base_samples["nvidia_smi_gpu_delta_mb"],
        "other": other_samples["nvidia_smi_gpu_delta_mb"],
        "difference": None
        if base_samples["nvidia_smi_gpu_delta_mb"] is None or other_samples["nvidia_smi_gpu_delta_mb"] is None
        else other_samples["nvidia_smi_gpu_delta_mb"] - base_samples["nvidia_smi_gpu_delta_mb"],
    }
    return comparison


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
        ]
        if args.clear_completed_phases:
            cmd.append("--clear-completed-phases")

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
