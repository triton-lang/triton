#!/usr/bin/env python3
"""Reproduce cupti13-specific heap growth during CUDA graph replay under Proton.

This script is meant to be run on a CUDA machine with Triton, Torch, and Proton
available. It drives a small Triton kernel under a Torch CUDA graph, repeatedly
replays that graph while Proton profiling is active, and records jemalloc heap
profiles plus process-memory snapshots at timed checkpoints.

Typical usage on the target machine:

  python third_party/proton/test/reproducers/cupti_graph_replay_heap_growth.py \
    --output-dir /tmp/proton-graph-heap-smoke \
    --duration-seconds 180 \
    --t0-seconds 30 \
    --t1-seconds 90 \
    --t3-seconds 180 \
    --replays-per-step 128 \
    --graph-ops 8

Compare Triton's packaged generic vs Blackwell CUPTI variants in separate
processes:

  python third_party/proton/test/reproducers/cupti_graph_replay_heap_growth.py \
    --output-dir /tmp/proton-graph-heap-compare \
    --duration-seconds 3600 \
    --t0-seconds 60 \
    --t1-seconds 1800 \
    --t3-seconds 3600 \
    --replays-per-step 128 \
    --graph-ops 8

If you preload profiling-enabled jemalloc before launch, each checkpoint emits:

  - jemalloc_stats.json
  - proc_status.txt
  - smaps_rollup.txt
  - top_anon_vmas.tsv
  - heap_profile_<pid>_<ts>.heap

That makes the script suitable for the same style of `t+1h` vs `t+3h` heap diff
used in the RuntimeWorker investigation.

One concrete launch command that produced a clear cupti12 vs cupti13 split on
a GB200 devbox was:

  LD_PRELOAD=/tmp/runtimeworker_memory_debug/jemalloc-prof/lib/libjemalloc.so.2 \
  PYTHONMALLOC=malloc \
  MALLOC_CONF='prof:true,prof_active:true,lg_prof_sample:19,background_thread:true,dirty_decay_ms:5000,muzzy_decay_ms:5000' \
  _RJEM_MALLOC_CONF='prof:true,prof_active:true,lg_prof_sample:19,background_thread:true,dirty_decay_ms:5000,muzzy_decay_ms:5000' \
  python /tmp/cupti_graph_replay_heap_growth.py \
    --output-dir /tmp/triton-cupti-graph-heap-medium1 \
    --duration-seconds 600 \
    --t0-seconds 60 \
    --t1-seconds 300 \
    --t3-seconds 600 \
    --sample-every-seconds 30 \
    --replays-per-step 256 \
    --graph-ops 8 \
    --phase-every 100 \
    --clear-completed-phases

After that run completed, the safest way to identify the blackwell heap-profile
pair is to read `summary_blackwell.json` instead of guessing from the filenames:

  read base cur < <(
    python - <<'PY'
  import json, pathlib
  summary = json.loads(pathlib.Path("/tmp/triton-cupti-graph-heap-medium1/summary_blackwell.json").read_text())
  print(
      summary["checkpoints"]["t_plus_1h"]["heap_profile_path"],
      summary["checkpoints"]["t_plus_3h"]["heap_profile_path"],
  )
  PY
  )

The native heap diff that surfaced the main replay path then came from:

  jeprof=/tmp/runtimeworker_memory_debug/jemalloc-prof/bin/jeprof
  exe=/root/.pyenv/versions/3.12.9/bin/python3.12
  "$jeprof" --text --show_bytes --base="$base" "$exe" "$cur" | \
    egrep 'Total:|at::cuda::CUDAGraph::replay|cuGraphLaunch|cudaGraphLaunch|cuptiEnableAllDomains|cuptiOpenMpInitialize_v2'

That diff highlighted the stack:

  at::cuda::CUDAGraph::replay
  cuGraphLaunch
  cudaGraphLaunch@@libcudart.so.13
  cuptiEnableAllDomains@@libcupti.so.13
  cuptiOpenMpInitialize_v2
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import ctypes.util
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("Run a Triton + Torch CUDA-graph replay workload under Proton and "
                                                  "capture timed heap snapshots for one or more CUPTI variants."))
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run one CUPTI configuration in the current process.",
    )
    parser.add_argument(
        "--label",
        default="run",
        help="Label used for output files in --single-run mode.",
    )
    parser.add_argument(
        "--cupti-dir",
        default=None,
        help=("CUPTI directory to force for this run. When set, both "
              "TRITON_CUPTI_LIB_PATH and TRITON_CUPTI_LIB_BLACKWELL_PATH are set "
              "to this directory before Triton is imported."),
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
        default="/tmp/proton-cupti-graph-heap-growth",
        help="Directory for JSON, CSV, heap, and profile outputs.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=600.0,
        help="Total runtime for each single run.",
    )
    parser.add_argument(
        "--t0-seconds",
        type=float,
        default=60.0,
        help="Checkpoint delay for the steady-state baseline.",
    )
    parser.add_argument(
        "--t1-seconds",
        type=float,
        default=3600.0,
        help="Checkpoint delay for the first comparison point.",
    )
    parser.add_argument(
        "--t3-seconds",
        type=float,
        default=10800.0,
        help="Checkpoint delay for the later comparison point.",
    )
    parser.add_argument(
        "--sample-every-seconds",
        type=float,
        default=30.0,
        help="Emit sample rows every N seconds while the loop is running.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations before Proton starts.",
    )
    parser.add_argument(
        "--phase-every",
        type=int,
        default=100,
        help="Advance Proton data phase every N replay steps.",
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
        "--graph-ops",
        type=int,
        default=8,
        help="Number of captured ops inside a single CUDA graph replay.",
    )
    parser.add_argument(
        "--replays-per-step",
        type=int,
        default=128,
        help="How many CUDA graph replays to issue per outer loop step.",
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
        help=("How to drive the Proton session. 'step' repeatedly activates one "
              "replay scope, then deactivates before the next outer step."),
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
        help="Optional sleep between outer replay steps.",
    )
    parser.add_argument(
        "--capture-before-start",
        action="store_true",
        help="Capture the CUDA graph before Proton starts.",
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
        if len(parts) != 2 or parts[0] != str(pid):
            continue
        try:
            return int(parts[1])
        except ValueError:
            return None
    return 0


def _read_cgroup_memory_fields() -> dict[str, int | None]:
    stat_path = Path("/sys/fs/cgroup/memory.stat")
    current_path = Path("/sys/fs/cgroup/memory.current")
    if not stat_path.exists() or not current_path.exists():
        return {
            "cgroup_memory_current_bytes": None,
            "cgroup_working_set_bytes": None,
            "cgroup_anon_bytes": None,
            "cgroup_file_bytes": None,
            "cgroup_active_file_bytes": None,
            "cgroup_inactive_file_bytes": None,
            "cgroup_shmem_bytes": None,
            "cgroup_slab_bytes": None,
        }

    stat_values: dict[str, int] = {}
    for line in stat_path.read_text().splitlines():
        key, value = line.split()
        stat_values[key] = int(value)

    current = int(current_path.read_text().strip())
    inactive_file = stat_values.get("inactive_file", 0)
    return {
        "cgroup_memory_current_bytes": current,
        "cgroup_working_set_bytes": max(current - inactive_file, 0),
        "cgroup_anon_bytes": stat_values.get("anon"),
        "cgroup_file_bytes": stat_values.get("file"),
        "cgroup_active_file_bytes": stat_values.get("active_file"),
        "cgroup_inactive_file_bytes": inactive_file,
        "cgroup_shmem_bytes": stat_values.get("shmem"),
        "cgroup_slab_bytes": stat_values.get("slab"),
    }


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


class _JemallocMallctl:

    def __init__(self) -> None:
        self._lib = self._load_library()
        self.mallctl = getattr(self._lib, "mallctl", None) if self._lib is not None else None
        if self.mallctl is not None:
            self.mallctl.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            self.mallctl.restype = ctypes.c_int

    @staticmethod
    def _load_library() -> ctypes.CDLL | None:
        with contextlib.suppress(Exception):
            return ctypes.CDLL(None)
        name = ctypes.util.find_library("jemalloc")
        if name:
            with contextlib.suppress(Exception):
                return ctypes.CDLL(name)
        return None

    def available(self) -> bool:
        return self.mallctl is not None

    def _call(
        self,
        name: bytes,
        oldp: ctypes.c_void_p | None = None,
        oldlenp: ctypes.Array[ctypes.c_size_t] | None = None,
        newp: ctypes.c_void_p | None = None,
        newlen: int = 0,
    ) -> int:
        if self.mallctl is None:
            return -1
        return int(self.mallctl(name, oldp, oldlenp, newp, newlen))

    def read_bool(self, name: str) -> bool | None:
        if self.mallctl is None:
            return None
        value = ctypes.c_bool()
        value_len = ctypes.c_size_t(ctypes.sizeof(value))
        rc = self._call(
            name.encode("utf-8") + b"\0",
            ctypes.byref(value),
            ctypes.pointer(value_len),
            None,
            0,
        )
        if rc != 0:
            return None
        return bool(value.value)

    def read_size_t(self, name: str) -> int | None:
        if self.mallctl is None:
            return None
        value = ctypes.c_size_t()
        value_len = ctypes.c_size_t(ctypes.sizeof(value))
        rc = self._call(
            name.encode("utf-8") + b"\0",
            ctypes.byref(value),
            ctypes.pointer(value_len),
            None,
            0,
        )
        if rc != 0:
            return None
        return int(value.value)

    def write_bool(self, name: str, value: bool) -> bool:
        if self.mallctl is None:
            return False
        encoded = ctypes.c_bool(value)
        rc = self._call(
            name.encode("utf-8") + b"\0",
            None,
            None,
            ctypes.byref(encoded),
            ctypes.sizeof(encoded),
        )
        return rc == 0

    def advance_epoch(self) -> None:
        if self.mallctl is None:
            return
        value = ctypes.c_uint64(1)
        self._call(
            b"epoch\0",
            None,
            None,
            ctypes.byref(value),
            ctypes.sizeof(value),
        )

    def dump_profile(self, output_path: str) -> bool:
        if self.mallctl is None:
            return False
        encoded = output_path.encode("utf-8") + b"\0"
        c_path = ctypes.c_char_p(encoded)
        rc = self._call(
            b"prof.dump\0",
            None,
            None,
            ctypes.byref(c_path),
            ctypes.sizeof(c_path),
        )
        return rc == 0


_JEMALLOC = _JemallocMallctl()


def _collect_jemalloc_stats() -> dict[str, int | bool | None]:
    stats: dict[str, int | bool | None] = {
        "mallctl_available": _JEMALLOC.available(),
        "jemalloc_maps_present": "jemalloc" in Path("/proc/self/maps").read_text(),
        "jemalloc_profiling_enabled": None,
        "allocated": None,
        "active": None,
        "resident": None,
        "mapped": None,
        "retained": None,
    }
    if not _JEMALLOC.available():
        return stats

    _JEMALLOC.advance_epoch()
    stats["jemalloc_profiling_enabled"] = _JEMALLOC.read_bool("opt.prof")
    for field in ("allocated", "active", "resident", "mapped", "retained"):
        stats[field] = _JEMALLOC.read_size_t(f"stats.{field}")
    return stats


def _dump_jemalloc_heap_profile(output_dir: Path) -> str | None:
    if not _JEMALLOC.available():
        return None
    if not _JEMALLOC.read_bool("opt.prof"):
        return None
    _JEMALLOC.write_bool("prof.active", True)
    _JEMALLOC.write_bool("prof.thread_active", True)
    _JEMALLOC.advance_epoch()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    path = output_dir / f"heap_profile_{os.getpid()}_{timestamp}.heap"
    if _JEMALLOC.dump_profile(str(path)):
        return str(path)
    return None


def _iter_smaps_regions() -> list[dict[str, int | str]]:
    try:
        lines = Path("/proc/self/smaps").read_text().splitlines()
    except OSError:
        return []

    regions: list[dict[str, int | str]] = []
    current: dict[str, int | str] | None = None
    for line in lines:
        if "-" in line and line.split(" ", 1)[0].count("-") == 1:
            parts = line.split()
            current = {
                "header": line,
                "path": parts[-1] if len(parts) >= 6 else "",
                "size_kb": 0,
                "rss_kb": 0,
                "private_dirty_kb": 0,
                "anonymous_kb": 0,
            }
            regions.append(current)
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        number = value.strip().split()[0]
        if not number.isdigit():
            continue
        parsed = int(number)
        if key == "Size":
            current["size_kb"] = parsed
        elif key == "Rss":
            current["rss_kb"] = parsed
        elif key == "Private_Dirty":
            current["private_dirty_kb"] = parsed
        elif key == "Anonymous":
            current["anonymous_kb"] = parsed
    return regions


def _top_anon_vmas_tsv(limit: int = 20) -> str:
    regions = [region for region in _iter_smaps_regions() if str(region["path"]).startswith("[") or not region["path"]]
    regions.sort(
        key=lambda region: (
            int(region["private_dirty_kb"]),
            int(region["anonymous_kb"]),
            int(region["size_kb"]),
        ),
        reverse=True,
    )
    rows = ["private_dirty_kb\tanonymous_kb\trss_kb\tsize_kb\tpath\theader"]
    for region in regions[:limit]:
        rows.append("\t".join([
            str(region["private_dirty_kb"]),
            str(region["anonymous_kb"]),
            str(region["rss_kb"]),
            str(region["size_kb"]),
            str(region["path"]),
            str(region["header"]),
        ]))
    return "\n".join(rows) + "\n"


def _info_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def _capture_checkpoint(label_dir: Path) -> dict[str, Any]:
    label_dir.mkdir(parents=True, exist_ok=True)

    proc_status_path = label_dir / "proc_status.txt"
    smaps_rollup_path = label_dir / "smaps_rollup.txt"
    top_anon_vmas_path = label_dir / "top_anon_vmas.tsv"
    jemalloc_stats_path = label_dir / "jemalloc_stats.json"

    proc_status_path.write_text(Path("/proc/self/status").read_text())
    smaps_rollup_path.write_text(Path("/proc/self/smaps_rollup").read_text())
    top_anon_vmas_path.write_text(_top_anon_vmas_tsv())
    jemalloc_stats = _collect_jemalloc_stats()
    jemalloc_stats_path.write_bytes(_info_json_bytes(jemalloc_stats))
    heap_profile_path = _dump_jemalloc_heap_profile(label_dir)
    snapshot = {
        "label_dir": str(label_dir),
        "proc_status_path": str(proc_status_path),
        "smaps_rollup_path": str(smaps_rollup_path),
        "top_anon_vmas_path": str(top_anon_vmas_path),
        "jemalloc_stats_path": str(jemalloc_stats_path),
        "heap_profile_path": heap_profile_path,
    }
    (label_dir / "snapshot.json").write_bytes(_info_json_bytes(snapshot))
    return snapshot


def _profile_output_stats(profile_base: Path) -> dict[str, int]:
    parent = profile_base.parent
    if not parent.exists():
        return {"profile_output_bytes": 0, "profile_output_files": 0, "profile_output_part_files": 0}
    prefix = profile_base.name
    files = [candidate for candidate in parent.iterdir() if candidate.is_file() and candidate.name.startswith(prefix)]
    total_bytes = sum(path.stat().st_size for path in files)
    part_files = sum(1 for path in files if ".part_" in path.name)
    return {
        "profile_output_bytes": total_bytes,
        "profile_output_files": len(files),
        "profile_output_part_files": part_files,
    }


def _write_csv(samples: list[dict[str, Any]], path: Path) -> None:
    fieldnames = sorted({key for sample in samples for key in sample})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:

    def _series(name: str) -> list[int]:
        return [sample[name] for sample in samples if sample.get(name) is not None]

    summary: dict[str, Any] = {"num_samples": len(samples)}
    for field in (
            "rss_bytes",
            "cgroup_working_set_bytes",
            "cgroup_anon_bytes",
            "cgroup_file_bytes",
            "allocated",
            "active",
            "resident",
            "mapped",
            "retained",
    ):
        values = _series(field)
        summary[f"{field}_start"] = values[0] if values else None
        summary[f"{field}_end"] = values[-1] if values else None
        summary[f"{field}_max"] = max(values) if values else None
        summary[f"{field}_delta"] = values[-1] - values[0] if len(values) >= 2 else None
    return summary


def _collect_sample(stage: str, elapsed_s: float, step: int, profile_base: Path) -> dict[str, Any]:
    sample = {
        "timestamp_s": time.time(),
        "elapsed_s": elapsed_s,
        "stage": stage,
        "step": step,
        "pid": os.getpid(),
        "rss_bytes": _read_proc_status_bytes("VmRSS"),
        "vms_bytes": _read_proc_status_bytes("VmSize"),
        "nvidia_smi_process_used_memory_mb": _query_nvidia_smi_process_memory_mb(os.getpid()),
    }
    sample.update(_read_cgroup_memory_fields())
    sample.update(_collect_jemalloc_stats())
    sample.update(_profile_output_stats(profile_base))
    return sample


def _configure_cupti_env(cupti_dir: str | None) -> None:
    os.environ.setdefault("PROTON_LAUNCH_METADATA_NOSYNC", "1")
    if cupti_dir is None:
        os.environ.pop("TRITON_CUPTI_LIB_PATH", None)
        os.environ.pop("TRITON_CUPTI_LIB_BLACKWELL_PATH", None)
        return
    os.environ["TRITON_CUPTI_LIB_PATH"] = cupti_dir
    os.environ["TRITON_CUPTI_LIB_BLACKWELL_PATH"] = cupti_dir


def _detect_packaged_cupti_dirs() -> tuple[Path, Path]:
    triton = importlib.import_module("triton")
    root = Path(triton.__file__).resolve().parent / "backends" / "nvidia" / "lib"
    return root / "cupti", root / "cupti-blackwell"


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
    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)

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
    static_x = torch.rand((numel, ), device=device, dtype=torch.float32)
    static_y = torch.rand((numel, ), device=device, dtype=torch.float32)
    static_z = torch.empty_like(static_x)

    def direct_iteration() -> None:
        for _ in range(args.graph_ops):
            static_z.fill_(0.0)
            add_kernel[grid](static_x, static_y, static_z, numel, BLOCK_SIZE=args.block_size)
            static_x.copy_(static_z)

    for _ in range(args.warmup):
        direct_iteration()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    g = torch.cuda.CUDAGraph()
    if args.capture_before_start:
        with torch.cuda.graph(g, stream=stream):
            direct_iteration()
        torch.cuda.synchronize()

    current_phase = 0
    last_cleared_phase = -1
    profile_base = output_dir / f"profile_{args.label}"
    profile_mode = f"periodic_flushing:format={args.data_format}"

    selected_cupti_dir = Path(args.cupti_dir) if args.cupti_dir is not None else None
    if selected_cupti_dir is None:
        generic_dir, blackwell_dir = _detect_packaged_cupti_dirs()
        target = triton.runtime.driver.active.get_current_target()
        selected_cupti_dir = blackwell_dir if target.backend == "cuda" and target.arch >= 100 else generic_dir

    run_metadata = {
        "label": args.label,
        "python_executable": sys.executable,
        "pid": os.getpid(),
        "device": str(device),
        "duration_seconds": args.duration_seconds,
        "t0_seconds": args.t0_seconds,
        "t1_seconds": args.t1_seconds,
        "t3_seconds": args.t3_seconds,
        "sample_every_seconds": args.sample_every_seconds,
        "warmup": args.warmup,
        "phase_every": args.phase_every,
        "numel": args.numel,
        "block_size": args.block_size,
        "graph_ops": args.graph_ops,
        "replays_per_step": args.replays_per_step,
        "data_format": args.data_format,
        "lifecycle": args.lifecycle,
        "clear_completed_phases": bool(args.clear_completed_phases),
        "sleep_ms": args.sleep_ms,
        "capture_before_start": bool(args.capture_before_start),
        "triton_version_file": str(Path(triton.__file__).resolve()),
        "triton_target_backend": triton.runtime.driver.active.get_current_target().backend,
        "triton_target_arch": triton.runtime.driver.active.get_current_target().arch,
        "selected_cupti": _load_cupti_info(selected_cupti_dir),
        "env": {
            "TRITON_CUPTI_LIB_PATH": os.environ.get("TRITON_CUPTI_LIB_PATH"),
            "TRITON_CUPTI_LIB_BLACKWELL_PATH": os.environ.get("TRITON_CUPTI_LIB_BLACKWELL_PATH"),
            "TRITON_PROFILE_BUFFER_SIZE": os.environ.get("TRITON_PROFILE_BUFFER_SIZE"),
            "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
            "MALLOC_CONF": os.environ.get("MALLOC_CONF"),
        },
        "loaded_cupti_libs_before_start": _read_loaded_shared_objects("libcupti.so", exact_basename=True),
        "loaded_libcupti_objects_before_start": _read_loaded_shared_objects("libcupti"),
    }

    samples: list[dict[str, Any]] = []
    checkpoints_root = output_dir / "checkpoints" / args.label
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    checkpoints = [
        ("t0", args.t0_seconds),
        ("t_plus_1h", args.t1_seconds),
        ("t_plus_3h", args.t3_seconds),
    ]
    captured_checkpoints: dict[str, Any] = {}

    def maybe_capture_checkpoints(elapsed_s: float) -> None:
        for label, target_s in checkpoints:
            if label in captured_checkpoints:
                continue
            if elapsed_s < target_s:
                continue
            captured_checkpoints[label] = _capture_checkpoint(checkpoints_root / label)

    samples.append(_collect_sample("pre_start", 0.0, 0, profile_base))
    session = proton.start(str(profile_base), backend="cupti", context="shadow", mode=profile_mode)
    samples.append(_collect_sample("post_start", 0.0, 0, profile_base))
    run_metadata["loaded_cupti_libs_after_start"] = _read_loaded_shared_objects("libcupti.so", exact_basename=True)
    run_metadata["loaded_libcupti_objects_after_start"] = _read_loaded_shared_objects("libcupti")

    if not args.capture_before_start:
        with torch.cuda.graph(g, stream=stream):
            direct_iteration()
        torch.cuda.synchronize()
        samples.append(_collect_sample("post_graph_capture", 0.0, 0, profile_base))

    if args.lifecycle == "step":
        proton.deactivate(session=session)
        samples.append(_collect_sample("post_initial_deactivate", 0.0, 0, profile_base))

    start = time.monotonic()
    last_sample_elapsed = -args.sample_every_seconds
    step = 0
    while True:
        elapsed_s = time.monotonic() - start
        if elapsed_s >= args.duration_seconds:
            break

        if args.lifecycle == "step":
            proton.activate(session=session)

        with proton.scope("graph_replay_step" if args.lifecycle == "step" else "graph_replay"):
            for _ in range(args.replays_per_step):
                g.replay()

        if args.phase_every > 0 and (step + 1) % args.phase_every == 0:
            current_phase = int(proton.data.advance_phase(session))

        if args.lifecycle == "step":
            proton.deactivate(session=session, flushing=False)

        if args.phase_every > 0 and (step + 1) % args.phase_every == 0:
            if args.clear_completed_phases and current_phase > 0:
                clear_phase = current_phase - 1
                if clear_phase > last_cleared_phase and proton.data.is_phase_complete(session, clear_phase):
                    proton.data.clear(session, phase=clear_phase, clear_up_to_phase=True)
                    last_cleared_phase = clear_phase

        elapsed_s = time.monotonic() - start
        maybe_capture_checkpoints(elapsed_s)

        if elapsed_s - last_sample_elapsed >= args.sample_every_seconds:
            samples.append(_collect_sample("loop", elapsed_s, step + 1, profile_base))
            last_sample_elapsed = elapsed_s

        step += 1
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    torch.cuda.synchronize()
    maybe_capture_checkpoints(time.monotonic() - start)
    proton.finalize(output_format=args.data_format)
    samples.append(_collect_sample("post_finalize", time.monotonic() - start, step, profile_base))
    run_metadata["loaded_cupti_libs_after_finalize"] = _read_loaded_shared_objects("libcupti.so", exact_basename=True)
    run_metadata["loaded_libcupti_objects_after_finalize"] = _read_loaded_shared_objects("libcupti")

    sample_path = output_dir / f"samples_{args.label}.csv"
    summary_path = output_dir / f"summary_{args.label}.json"
    _write_csv(samples, sample_path)

    summary = run_metadata | {
        "sample_csv": str(sample_path),
        "profile_base": str(profile_base),
        "summary_json": str(summary_path),
        "samples": _summarize_samples(samples),
        "checkpoints": captured_checkpoints,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary["samples"], sort_keys=True))
    return 0


def _compare_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    comparison: dict[str, Any] = {"runs": summaries}
    if len(summaries) < 2:
        return comparison

    base = summaries[0]
    other = summaries[1]
    for field in (
            "allocated_delta",
            "active_delta",
            "resident_delta",
            "mapped_delta",
            "retained_delta",
            "rss_bytes_delta",
            "cgroup_anon_bytes_delta",
            "cgroup_file_bytes_delta",
    ):
        base_value = base["samples"].get(field)
        other_value = other["samples"].get(field)
        comparison[field] = {
            "base_label": base["label"],
            "other_label": other["label"],
            "base": base_value,
            "other": other_value,
            "difference": None if base_value is None or other_value is None else other_value - base_value,
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
            "--duration-seconds",
            str(args.duration_seconds),
            "--t0-seconds",
            str(args.t0_seconds),
            "--t1-seconds",
            str(args.t1_seconds),
            "--t3-seconds",
            str(args.t3_seconds),
            "--sample-every-seconds",
            str(args.sample_every_seconds),
            "--warmup",
            str(args.warmup),
            "--phase-every",
            str(args.phase_every),
            "--numel",
            str(args.numel),
            "--block-size",
            str(args.block_size),
            "--graph-ops",
            str(args.graph_ops),
            "--replays-per-step",
            str(args.replays_per_step),
            "--device",
            str(args.device),
            "--data-format",
            args.data_format,
            "--lifecycle",
            args.lifecycle,
            "--sleep-ms",
            str(args.sleep_ms),
        ]
        if args.clear_completed_phases:
            cmd.append("--clear-completed-phases")
        if args.capture_before_start:
            cmd.append("--capture-before-start")

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
