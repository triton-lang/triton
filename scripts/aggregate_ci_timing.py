#!/usr/bin/env python3

from __future__ import annotations

import argparse
import heapq
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

try:
    import ijson
except ImportError:
    ijson = None

Interval = tuple[int, int]
CALIBRATION_SCOPE = "__triton_ci_timing_calibration__"


def union_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted((start, end) for start, end in intervals if end > start):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def intersect_intervals(lhs: Iterable[Interval], rhs: Iterable[Interval]) -> list[Interval]:
    left = union_intervals(lhs)
    right = union_intervals(rhs)
    result: list[Interval] = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def duration(intervals: Iterable[Interval]) -> int:
    return sum(end - start for start, end in union_intervals(intervals))


def clip_intervals(intervals: Iterable[Interval], windows: Iterable[Interval]) -> list[Interval]:
    return intersect_intervals(intervals, windows)


def iter_trace_events(trace_path: Path) -> Iterator[dict]:
    if ijson is not None:
        with trace_path.open("rb") as stream:
            yield from ijson.items(stream, "traceEvents.item", use_float=True)
        return
    yield from json.loads(trace_path.read_text()).get("traceEvents", [])


def iter_trace_intervals(
    summary_path: Path,
    summary: dict,
    stats: dict[str, int],
    errors: list[str],
) -> Iterator[Interval]:
    calibration = summary.get("calibration")
    if not calibration:
        errors.append(f"{summary_path}: missing calibration")
        return

    trace_path = summary_path.parent / Path(summary["profile"]).name
    phase_zero_suffix = ".part_0.chrome_trace"
    if not trace_path.exists() and trace_path.name.endswith(phase_zero_suffix):
        profile_base = trace_path.name[:-len(phase_zero_suffix)]
        phase_sidecars = list(summary_path.parent.glob(f"{profile_base}.part_*.profile.json"))
        fallback = summary_path.parent / f"{profile_base}.chrome_trace"
        if len(phase_sidecars) == 1 and fallback.exists():
            trace_path = fallback
    if not trace_path.exists():
        errors.append(f"{trace_path}: missing trace")
        return
    if trace_path.stat().st_size == 0:
        errors.append(f"{trace_path}: empty trace")
        return
    try:
        events = iter_trace_events(trace_path)
        stream_count = 0
        trace_anchor_us: float | None = None
        mono_anchor_ns = (int(calibration["enter_before_ns"]) + int(calibration["enter_after_ns"])) // 2
        buffered: list[Interval] = []
        last_start_ns: int | None = None
        for event in events:
            if event.get("ph") == "M":
                name = event.get("args", {}).get("name", "")
                if isinstance(name, str) and name.startswith("GPU Stream "):
                    stream_count += 1
                continue
            if event.get("name") == CALIBRATION_SCOPE and event.get("cat") == "scope" and event.get("ph") == "X":
                if trace_anchor_us is not None:
                    errors.append(f"{trace_path}: multiple calibration events")
                    return
                trace_anchor_us = float(event["ts"])
                continue
            if event.get("cat") != "kernel" or event.get("ph") != "X":
                continue
            if trace_anchor_us is None:
                errors.append(f"{trace_path}: kernel event precedes calibration")
                return
            start_ns = mono_anchor_ns + round((float(event["ts"]) - trace_anchor_us) * 1000)
            duration_ns = max(0, round(float(event["dur"]) * 1000))
            interval = (start_ns, start_ns + duration_ns)
            stats["raw_gpu_ns"] += duration_ns
            stats["kernel_events"] += 1
            if stream_count > 1:
                buffered.append(interval)
            else:
                if last_start_ns is not None and start_ns < last_start_ns:
                    errors.append(f"{trace_path}: single-stream events are not time ordered")
                    return
                last_start_ns = start_ns
                yield interval
    except Exception as exc:
        errors.append(f"{trace_path}: {exc}")
        return

    if trace_anchor_us is None:
        errors.append(f"{trace_path}: missing calibration event")
        return
    yield from sorted(buffered)


def load_trace(summary_path: Path, summary: dict) -> tuple[list[Interval], int, list[str]]:
    errors: list[str] = []
    stats = {"raw_gpu_ns": 0, "kernel_events": 0}
    intervals = list(iter_trace_intervals(summary_path, summary, stats, errors))
    return intervals, stats["raw_gpu_ns"], errors


def stream_gpu_stats(
    traces: list[tuple[Path, dict]],
    windows: list[Interval],
    compile_intervals: list[Interval],
) -> tuple[int, int, int, int, list[str]]:
    errors: list[str] = []
    stats = {"raw_gpu_ns": 0, "kernel_events": 0}
    streams = [iter_trace_intervals(path, summary, stats, errors) for path, summary in traces]

    gpu_ns = 0
    overlap_ns = 0
    window_index = 0
    compile_index = 0

    def consume(start: int, end: int) -> None:
        nonlocal gpu_ns, overlap_ns, window_index, compile_index
        while window_index < len(windows) and windows[window_index][1] <= start:
            window_index += 1
        index = window_index
        while index < len(windows) and windows[index][0] < end:
            clipped_start = max(start, windows[index][0])
            clipped_end = min(end, windows[index][1])
            if clipped_end > clipped_start:
                gpu_ns += clipped_end - clipped_start
                while compile_index < len(compile_intervals) and compile_intervals[compile_index][1] <= clipped_start:
                    compile_index += 1
                compile_scan = compile_index
                while compile_scan < len(compile_intervals) and compile_intervals[compile_scan][0] < clipped_end:
                    overlap_start = max(clipped_start, compile_intervals[compile_scan][0])
                    overlap_end = min(clipped_end, compile_intervals[compile_scan][1])
                    if overlap_end > overlap_start:
                        overlap_ns += overlap_end - overlap_start
                    if compile_intervals[compile_scan][1] >= clipped_end:
                        break
                    compile_scan += 1
                compile_index = compile_scan
            index += 1

    current: Interval | None = None
    for start, end in heapq.merge(*streams, key=lambda interval: interval[0]):
        if current is None:
            current = (start, end)
        elif start <= current[1]:
            current = (current[0], max(current[1], end))
        else:
            consume(*current)
            current = (start, end)
    if current is not None:
        consume(*current)

    return gpu_ns, overlap_ns, stats["raw_gpu_ns"], stats["kernel_events"], errors


def analyze(root: Path) -> dict:
    summaries: list[tuple[Path, dict]] = []
    profile_phases: list[tuple[Path, dict]] = []
    profile_phases_by_dir: dict[Path, list[tuple[Path, dict]]] = defaultdict(list)
    errors: list[str] = []
    for path in sorted(root.rglob("summary.json")):
        try:
            summary = json.loads(path.read_text())
            summaries.append((path, summary))
            errors.extend(f"{path}: {error}" for error in summary.get("errors", []))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    for path in sorted(root.rglob("*.profile.json")):
        try:
            profile = json.loads(path.read_text())
            if profile.get("kind") != "gpu_profile_phase":
                errors.append(f"{path}: unexpected profile sidecar kind")
                continue
            profile_phases.append((path, profile))
            profile_phases_by_dir[path.parent].append((path, profile))
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    summary_dirs = {path.parent for path, _ in summaries}
    for directory in sorted(set(profile_phases_by_dir) - summary_dirs):
        errors.append(f"{directory}: profile directory missing summary")
    for path, summary in summaries:
        if summary["role"] == "controller" or summary.get("profile_phase_tests") is None or summary.get("errors"):
            continue
        expected_phases = int(summary.get("profile_phases", 0))
        actual_phases = len(profile_phases_by_dir[path.parent])
        if actual_phases != expected_phases:
            errors.append(f"{path.parent}: expected {expected_phases} profile phases, found {actual_phases}")

    by_invocation: dict[str, list[dict]] = defaultdict(list)
    for _, summary in summaries:
        by_invocation[summary["invocation"]].append(summary)

    invocation_windows: dict[str, Interval] = {}
    invocation_test_windows: dict[str, Interval] = {}
    for invocation, group in by_invocation.items():
        owners = [item for item in group if item["role"] in ("controller", "main")]
        candidates = owners or group
        starts = [int(item["mono_start_ns"]) for item in candidates if item.get("mono_start_ns") is not None]
        ends = [int(item["mono_end_ns"]) for item in candidates if item.get("mono_end_ns") is not None]
        if starts and ends:
            invocation_windows[invocation] = (min(starts), max(ends))
            workers = [item for item in group if item["role"] == "worker"]
            test_candidates = workers or candidates
            test_ends = [
                int(item["mono_tests_end_ns"]) for item in test_candidates if item.get("mono_tests_end_ns") is not None
            ]
            invocation_test_windows[invocation] = (min(starts), max(test_ends) if test_ends else max(ends))
        else:
            errors.append(f"{invocation}: missing session window")

    groups: dict[str, dict[str, list | int]] = defaultdict(
        lambda: {
            "windows": [],
            "full_windows": [],
            "compile": [],
            "traces": [],
            "raw_compile_ns": 0,
            "compile_events": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "ir_init_ns": 0,
            "lowering_ns": 0,
            "store_ns": 0,
        })

    seen_windows: set[tuple[str, str]] = set()
    for path, summary in summaries:
        label = summary["label"]
        group = groups[label]
        invocation = summary["invocation"]
        window = invocation_windows.get(invocation)
        if window is not None and (label, invocation) not in seen_windows:
            group["full_windows"].append(window)
            group["windows"].append(invocation_test_windows[invocation])
            seen_windows.add((label, invocation))

        for event in summary.get("compile_events", []):
            start, end, cache_hit, ir_init, lowering, store = map(int, event)
            group["compile"].append((start, end))
            group["raw_compile_ns"] += end - start
            group["compile_events"] += 1
            group["cache_hits"] += cache_hit
            group["cache_misses"] += 1 - cache_hit
            group["ir_init_ns"] += ir_init
            group["lowering_ns"] += lowering
            group["store_ns"] += store

        if summary["role"] != "controller" and not summary.get("errors") and summary.get("profile"):
            group["traces"].append((path, summary))

    for path, profile in profile_phases:
        groups[profile["label"]]["traces"].append((path, profile))

    results: dict[str, dict[str, int | float]] = {}
    for label, group in sorted(groups.items()):
        windows = union_intervals(group["windows"])
        compile_intervals = clip_intervals(group["compile"], windows)
        compile_ns = duration(compile_intervals)
        gpu_ns, overlap_ns, raw_gpu_ns, kernel_events, trace_errors = stream_gpu_stats(
            group["traces"],
            windows,
            compile_intervals,
        )
        errors.extend(trace_errors)
        wall_ns = duration(windows)
        full_wall_ns = duration(group["full_windows"])
        host_ns = max(0, wall_ns - compile_ns - gpu_ns + overlap_ns)
        results[label] = {
            "wall_ns": wall_ns,
            "worker_shutdown_tail_ns": max(0, full_wall_ns - wall_ns),
            "compile_only_ns": compile_ns - overlap_ns,
            "gpu_only_ns": gpu_ns - overlap_ns,
            "compile_gpu_overlap_ns": overlap_ns,
            "host_remainder_ns": host_ns,
            "raw_compile_worker_ns": int(group["raw_compile_ns"]),
            "raw_gpu_kernel_ns": raw_gpu_ns,
            "gpu_kernel_events": kernel_events,
            "compile_events": int(group["compile_events"]),
            "cache_hits": int(group["cache_hits"]),
            "cache_misses": int(group["cache_misses"]),
            "ir_init_worker_ns": int(group["ir_init_ns"]),
            "lowering_worker_ns": int(group["lowering_ns"]),
            "store_worker_ns": int(group["store_ns"]),
        }

    return {
        "schema_version": 1,
        "summaries": len(summaries),
        "profile_phases": len(profile_phases),
        "results": results,
        "errors": errors,
    }


def combine_analyses(compile_analysis: dict, gpu_analysis: dict) -> dict:
    errors = [f"compile: {error}" for error in compile_analysis["errors"]]
    errors.extend(f"gpu: {error}" for error in gpu_analysis["errors"])
    results: dict[str, dict[str, int | float]] = {}
    labels = sorted(set(compile_analysis["results"]) | set(gpu_analysis["results"]))
    for label in labels:
        if label not in compile_analysis["results"]:
            errors.append(f"{label}: missing compile-only result")
            continue
        if label not in gpu_analysis["results"]:
            errors.append(f"{label}: missing GPU-trace result")
            continue
        compile_result = compile_analysis["results"][label]
        gpu_result = gpu_analysis["results"][label]
        compile_ns = compile_result["compile_only_ns"] + compile_result["compile_gpu_overlap_ns"]
        gpu_ns = gpu_result["gpu_only_ns"] + gpu_result["compile_gpu_overlap_ns"]
        overlap_ns = min(gpu_result["compile_gpu_overlap_ns"], compile_ns, gpu_ns)
        wall_ns = compile_result["wall_ns"]
        results[label] = {
            "wall_ns": wall_ns,
            "worker_shutdown_tail_ns": compile_result["worker_shutdown_tail_ns"],
            "compile_only_ns": compile_ns - overlap_ns,
            "gpu_only_ns": gpu_ns - overlap_ns,
            "compile_gpu_overlap_ns": overlap_ns,
            "host_remainder_ns": max(0, wall_ns - compile_ns - gpu_ns + overlap_ns),
            "raw_compile_worker_ns": compile_result["raw_compile_worker_ns"],
            "raw_gpu_kernel_ns": gpu_result["raw_gpu_kernel_ns"],
            "gpu_kernel_events": gpu_result["gpu_kernel_events"],
            "compile_events": compile_result["compile_events"],
            "cache_hits": compile_result["cache_hits"],
            "cache_misses": compile_result["cache_misses"],
            "ir_init_worker_ns": compile_result["ir_init_worker_ns"],
            "lowering_worker_ns": compile_result["lowering_worker_ns"],
            "store_worker_ns": compile_result["store_worker_ns"],
            "gpu_trace_wall_ns": gpu_result["wall_ns"],
            "gpu_trace_compile_active_ns": gpu_result["compile_only_ns"] + gpu_result["compile_gpu_overlap_ns"],
        }
    return {
        "schema_version": 1,
        "combined_from_independent_runs": True,
        "compile_summaries": compile_analysis["summaries"],
        "gpu_summaries": gpu_analysis["summaries"],
        "profile_phases": gpu_analysis["profile_phases"],
        "results": results,
        "errors": errors,
    }


def format_duration(ns: int) -> str:
    seconds = ns / 1_000_000_000
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}:{remainder:05.2f}"


def print_table(analysis: dict) -> None:
    headers = ["Suite", "Test wall", "Compile only", "GPU only", "Overlap", "Host remainder", "Worker shutdown"]
    rows: list[list[str]] = []
    for label, result in analysis["results"].items():
        rows.append([
            label,
            format_duration(result["wall_ns"]),
            format_duration(result["compile_only_ns"]),
            format_duration(result["gpu_only_ns"]),
            format_duration(result["compile_gpu_overlap_ns"]),
            format_duration(result["host_remainder_ns"]),
            format_duration(result["worker_shutdown_tail_ns"]),
        ])

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def fence(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    print(fence("┌", "┬", "┐"))
    print("│ " + " │ ".join(header.ljust(width) for header, width in zip(headers, widths)) + " │")
    print(fence("├", "┼", "┤"))
    for row in rows:
        print("│ " + " │ ".join(
            cell.ljust(width) if index == 0 else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(row, widths))) + " │")
    print(fence("└", "┴", "┘"))


def self_test() -> None:
    assert union_intervals([(0, 10), (5, 12), (20, 30)]) == [(0, 12), (20, 30)]
    assert intersect_intervals([(0, 10), (20, 30)], [(5, 25)]) == [(5, 10), (20, 25)]
    windows = [(0, 100)]
    compile_intervals = clip_intervals([(10, 40), (60, 80)], windows)
    gpu_intervals = clip_intervals([(30, 70)], windows)
    overlap = duration(intersect_intervals(compile_intervals, gpu_intervals))
    assert duration(compile_intervals) == 50
    assert duration(gpu_intervals) == 40
    assert overlap == 20
    assert duration(windows) - duration(compile_intervals) - duration(gpu_intervals) + overlap == 30

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary_path = root / "summary.json"
        trace_path = root / "gpu.chrome_trace"
        trace_path.write_text(
            json.dumps({
                "traceEvents": [
                    {"name": CALIBRATION_SCOPE, "cat": "scope", "ph": "X", "ts": 1000.0, "dur": 10.0},
                    {"name": "kernel", "cat": "kernel", "ph": "X", "ts": 2000.0, "dur": 500.0},
                ]
            }))
        summary = {
            "profile": str(trace_path),
            "calibration": {
                "enter_before_ns": 10_000_000,
                "enter_after_ns": 10_000_100,
            },
        }
        gpu_intervals, raw_ns, errors = load_trace(summary_path, summary)
        assert errors == []
        assert gpu_intervals == [(11_000_050, 11_500_050)]
        assert raw_ns == 500_000

        phase_sidecar = root / "gpu.part_0.profile.json"
        phase_sidecar.write_text("{}")
        summary["kind"] = "gpu_profile_phase"
        summary["phase"] = 0
        summary["profile"] = str(root / "gpu.part_0.chrome_trace")
        gpu_intervals, raw_ns, errors = load_trace(phase_sidecar, summary)
        assert errors == []
        assert gpu_intervals == [(11_000_050, 11_500_050)]
        assert raw_ns == 500_000

    with tempfile.TemporaryDirectory() as tmp:
        orphan = Path(tmp) / "orphan"
        orphan.mkdir()
        (orphan / "gpu.part_0.profile.json").write_text(
            json.dumps({
                "kind": "gpu_profile_phase",
                "label": "test",
                "invocation": "orphan",
                "role": "worker",
                "worker_id": "gw0",
                "phase": 0,
                "calibration": {
                    "enter_before_ns": 0,
                    "enter_after_ns": 0,
                },
                "profile": str(orphan / "gpu.part_0.chrome_trace"),
            }))
        orphan_analysis = analyze(Path(tmp))
        assert f"{orphan}: profile directory missing summary" in orphan_analysis["errors"]

    compile_result = {
        "wall_ns": 100,
        "worker_shutdown_tail_ns": 1,
        "compile_only_ns": 40,
        "compile_gpu_overlap_ns": 0,
        "raw_compile_worker_ns": 50,
        "compile_events": 2,
        "cache_hits": 1,
        "cache_misses": 1,
        "ir_init_worker_ns": 5,
        "lowering_worker_ns": 40,
        "store_worker_ns": 5,
    }
    gpu_result = {
        "wall_ns": 120,
        "compile_only_ns": 50,
        "gpu_only_ns": 10,
        "compile_gpu_overlap_ns": 20,
        "raw_gpu_kernel_ns": 35,
        "gpu_kernel_events": 3,
    }
    combined = combine_analyses(
        {"summaries": 1, "profile_phases": 0, "results": {"test": compile_result}, "errors": []},
        {"summaries": 1, "profile_phases": 2, "results": {"test": gpu_result}, "errors": []},
    )
    assert combined["results"]["test"]["compile_only_ns"] == 20
    assert combined["results"]["test"]["gpu_only_ns"] == 10
    assert combined["results"]["test"]["compile_gpu_overlap_ns"] == 20
    assert combined["results"]["test"]["host_remainder_ns"] == 50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--gpu-root", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.root is None:
        parser.error("root is required unless --self-test is used")
    result = analyze(args.root)
    if args.gpu_root is not None:
        result = combine_analyses(result, analyze(args.gpu_root))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_table(result)
        if result["errors"]:
            print("\nErrors:")
            for error in result["errors"]:
                print(f"- {error}")


if __name__ == "__main__":
    main()
