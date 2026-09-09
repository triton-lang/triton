import json
import os

import triton


class CompilationTrace:

    def __init__(self, directory, phase, test=None):
        self.directory = directory
        self.phase = phase
        self.test = test
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f"{phase}-{os.getpid()}.jsonl")

    def __call__(self, *, src, metadata, metadata_group, times, cache_hit):
        function = getattr(src, "fn", None)
        kernel = getattr(function, "_fn_name", src.name)
        record = {
            "phase": self.phase,
            "test": self.test,
            "kernel": kernel,
            "hash": metadata["hash"],
            "source_hash": src.hash(),
            "cache_hit": cache_hit,
            "duration_us": times.total,
            "worker": os.environ.get("PYTEST_XDIST_WORKER", "main"),
            "gpu":
            os.environ.get("HIP_VISIBLE_DEVICES" if metadata["target"].backend == "hip" else "CUDA_VISIBLE_DEVICES"),
            "compiler_worker": os.environ.get("TRITON_WARMUP_COMPILER_WORKER"),
            "cache_dir": triton.knobs.cache.dir,
        }
        with open(self.path, "a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def summarize_compile_trace(directory):
    records = []
    attempted_tests = {}
    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if name.endswith(".jsonl"):
                with open(path, encoding="utf-8") as source:
                    records.extend(json.loads(line) for line in source if line.strip())
            elif name.endswith(".tests"):
                with open(path, encoding="utf-8") as source:
                    for line in source:
                        if line.strip():
                            attempted = json.loads(line)
                            attempted_tests.setdefault(attempted["phase"], set()).add(attempted["test"])

    records_by_phase = {}
    for record in records:
        records_by_phase.setdefault(record["phase"], []).append(record)

    warmup_records = [record for record in records if record["phase"].startswith("warmup-")]
    all_warmed_hashes = {record["hash"] for record in warmup_records}
    all_warmed_entries = {(record["cache_dir"], record["hash"]) for record in warmup_records}
    all_warmed_tests = {record["test"] for record in warmup_records if record.get("test") is not None}
    for name, tests in attempted_tests.items():
        if name.startswith("warmup-"):
            all_warmed_tests.update(tests)

    runtime_records = [record for record in records if not record["phase"].startswith("warmup-")]
    globally_used_hashes = {
        record["hash"]
        for record in runtime_records
        if record["cache_hit"] and (record["cache_dir"], record["hash"]) in all_warmed_entries
    }
    summaries = {}
    phase_names = set(records_by_phase) | set(attempted_tests)
    phase_names.update(name.removeprefix("warmup-") for name in tuple(phase_names) if name.startswith("warmup-"))
    for name in sorted(phase_names):
        events = records_by_phase.get(name, [])
        misses = [record for record in events if not record["cache_hit"]]
        is_warmup_phase = name.startswith("warmup-")
        matching = [] if is_warmup_phase else records_by_phase.get(f"warmup-{name}", [])
        warmed_hashes = {record["hash"] for record in matching}
        warmed_tests = {record["test"] for record in matching if record.get("test") is not None}
        if not is_warmup_phase:
            warmed_tests.update(attempted_tests.get(f"warmup-{name}", set()))
        warmed_test_events = [] if is_warmup_phase else [
            record for record in events if record.get("test") in warmed_tests
        ]
        warmed_test_hits = [
            record for record in warmed_test_events
            if record["cache_hit"] and (record["cache_dir"], record["hash"]) in all_warmed_entries
        ]
        incomplete_tests = {
            record["test"]
            for record in warmed_test_events
            if not record["cache_hit"] or (record["cache_dir"], record["hash"]) not in all_warmed_entries
        }
        runtime_tests = {record.get("test") for record in events} | attempted_tests.get(name, set())
        missing_tests = warmed_tests - runtime_tests if not is_warmup_phase else set()
        runtime_warmed_entries = set() if is_warmup_phase else all_warmed_entries
        warmed_events = [record for record in events if (record["cache_dir"], record["hash"]) in runtime_warmed_entries]
        warmed_hits = sum(record["cache_hit"] for record in warmed_events)
        summaries[name] = {
            "events": len(events),
            "disk_hits": sum(record["cache_hit"] for record in events),
            "disk_misses": len(misses),
            "warmed_hits": warmed_hits,
            "warmed_misses": len(warmed_events) - warmed_hits,
            "warmed_test_events": len(warmed_test_events),
            "warmed_test_hits": len(warmed_test_hits),
            "warmed_test_misses": len(warmed_test_events) - len(warmed_test_hits),
            "incomplete_warmed_test_count": len(incomplete_tests),
            "incomplete_warmed_tests": sorted(incomplete_tests)[:20],
            "missing_warmed_test_count": len(missing_tests),
            "missing_warmed_tests": sorted(missing_tests)[:20],
            "unused_warmup_hashes": len(warmed_hashes - globally_used_hashes),
            "compile_seconds": round(sum(record["duration_us"] for record in misses) / 1_000_000, 3),
        }

    return {
        "phases": summaries,
        "warmup_hashes": len(all_warmed_hashes),
        "warmup_tests": len(all_warmed_tests),
        "unused_warmup_hashes": len(all_warmed_hashes - globally_used_hashes),
    }


def _require_complete_warmup(report):
    runtime = {name: summary for name, summary in report["phases"].items() if not name.startswith("warmup-")}
    if not runtime or not any(summary["warmed_test_events"] for summary in runtime.values()):
        raise SystemExit("warmup did not produce any compiler events in warmed runtime tests")

    incomplete = {
        name: summary["incomplete_warmed_tests"]
        for name, summary in runtime.items()
        if summary["warmed_test_misses"]
    }
    if incomplete:
        raise SystemExit(f"warmed runtime tests were not complete cache hits: {json.dumps(incomplete, sort_keys=True)}")

    missing = {
        name: summary["missing_warmed_tests"]
        for name, summary in runtime.items()
        if summary["missing_warmed_test_count"]
    }
    if missing:
        raise SystemExit(f"marked warmup tests were not executed: {json.dumps(missing, sort_keys=True)}")

    if report.get("unused_warmup_hashes", 0):
        raise SystemExit(f"warmup produced {report['unused_warmup_hashes']} unused specializations")
