from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton.tools import ir_golden
from triton.tools.ir_golden import (SCHEMA_VERSION, GoldenCorpus, GoldenCorpusError, capture_compilations,
                                    classify_golden_change, compilation_capture_listener,
                                    export_triton_opt_reproducers, freeze_payload, function_names, ir_metrics, main,
                                    normalize_source, normalize_ttgir, replay_reproducer_shard,
                                    triton_opt_reproducer, ttgir_dependency_signature, validate_golden_evidence,
                                    verify_payload, write_reproducer_shard, write_shard)

TARGET = {"backend": "cuda", "arch": 80, "warp_size": 32}
TTIR = 'module { tt.func public @copy(%arg0: !tt.ptr<f32>) { tt.return } }'
GLUON = ('module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, '
         'ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} '
         '{ tt.func public @copy(%arg0: !tt.ptr<f32>) { tt.return } }')
LAYOUT_BEFORE = ("#source = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], "
                 "order = [0]}>\n"
                 "#target = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], "
                 "order = [0]}>\n"
                 "module {\n"
                 "  tt.func public @copy(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {\n"
                 "    %0 = tt.load %arg0 : tensor<32xf32, #source>\n"
                 "    %1 = ttg.convert_layout %0 : tensor<32xf32, #source> -> tensor<32xf32, #target>\n"
                 "    tt.store %arg1, %1 : tensor<32xf32, #target>\n"
                 "    tt.return\n"
                 "  }\n"
                 "}\n")
LAYOUT_AFTER = ("#source = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], "
                "order = [0]}>\n"
                "#target = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], "
                "order = [0]}>\n"
                "module {\n"
                "  tt.func public @copy(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {\n"
                "    %0 = tt.load %arg0 : tensor<32xf32, #target>\n"
                "    tt.store %arg1, %0 : tensor<32xf32, #target>\n"
                "    tt.return\n"
                "  }\n"
                "}\n")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "production_golden_case" not in metafunc.fixturenames:
        return
    root = os.environ.get("TRITON_PRODUCTION_IR_CORPUS")
    if root is None:
        metafunc.parametrize("production_golden_case", [
            pytest.param(None, marks=pytest.mark.skip(reason="External production IR golden corpus is not configured"))
        ])
        return
    corpus = GoldenCorpus(root)
    metafunc.parametrize("production_golden_case", [
        pytest.param(case, marks=pytest.mark.xdist_group(case.shard),
                     id=f"{case.language}-{case.kernel}-{case.case_id[:12]}") for case in corpus.cases
    ])


def make_payload(language: str = "triton") -> dict:
    return {
        "id": f"synthetic-{language}", "kernel": "copy", "family": "synthetic", "language": language, "target": TARGET,
        "source": TTIR if language == "triton" else GLUON, "launch": {"num_warps": 4, "num_ctas": 1, "num_stages": 2},
        "metadata": {"num_warps": 4, "num_ctas": 1, "num_stages": 2, "instrumentation_mode": ""}
    }


def make_corpus(root: Path, languages: tuple[str, ...] = ("triton", "gluon")) -> GoldenCorpus:
    payloads = {}
    cases = []
    for language in languages:
        payload = freeze_payload(make_payload(language))
        member = f"{payload['id']}.json"
        payloads[member] = payload
        cases.append({
            "id": payload["id"], "kernel": payload["kernel"], "family": payload["family"], "language": language, "arch":
            TARGET["arch"], "shard": "synthetic", "member": member
        })
    shard = write_shard(root / "synthetic.tar.zst", payloads)
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "shards": [shard], "cases": cases}))
    return GoldenCorpus(root)


def make_evidence(payload: dict, expected: str, actual: str, *, kind: str = "structural-improvement") -> dict:
    source = payload.get("canonical_source", payload["source"])
    evidence = {
        "case_id": payload["id"], "source_sha256": payload.get("canonical_source_sha256",
                                                                   hashlib.sha256(source.encode()).hexdigest()),
        "previous_global_sha256": hashlib.sha256(expected.encode()).hexdigest(),
        "candidate_global_sha256": hashlib.sha256(actual.encode()).hexdigest(),
        "target_arch": payload["target"]["arch"], "baseline_compiler_commit": "a" * 40,
        "candidate_compiler_commit": "b" * 40, "rationale": "Remove a redundant physical layout conversion",
        "legacy_unchanged": True, "correctness_passed": True, "kind": kind
    }
    if kind == "gpu-benchmark":
        evidence.update({
            "gpu_uuid": "GPU-synthetic", "driver_version": "580.0", "gpu_arch": payload["target"]["arch"],
            "prepared_input_sha256": "c" * 64, "baseline_cubin_sha256": "d" * 64,
            "candidate_cubin_sha256": "e" * 64, "repetitions": 300,
            "baseline_samples_ns": [1050, 1055, 1048, 1052, 1051],
            "candidate_samples_ns": [910, 912, 911, 913, 909]
        })
    return evidence


@pytest.mark.parametrize("language", ("triton", "gluon"))
def test_golden_replays_without_visible_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, language: str) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    corpus = make_corpus(tmp_path)
    case = corpus.select(language=language)[0]
    assert set(verify_payload(corpus.payload(case))) == {"legacy", "global"}


def test_normalization_removes_debug_locations() -> None:
    source = ('#loc = loc("/machine-specific/kernel.py":5:1)\n'
              'module { tt.func public @copy(%arg0: !tt.ptr<f32> loc(#loc)) { tt.return loc(#loc) } loc(#loc) }')
    normalized = normalize_source(source, TARGET, "triton")
    assert "/machine-specific/" not in normalized
    assert function_names(normalized) == ["copy"]


def test_golden_rejects_changed_output() -> None:
    payload = freeze_payload(make_payload())
    payload["goldens"]["global"]["ttgir"] += "\n"
    with pytest.raises(GoldenCorpusError, match="reference is corrupt"):
        verify_payload(payload)


def test_golden_rejects_changed_compiler_metadata() -> None:
    payload = freeze_payload(make_payload())
    payload["launch"]["num_warps"] = 0
    with pytest.raises(GoldenCorpusError, match="compiler options"):
        verify_payload(payload)


def test_golden_rejects_missing_shard(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)
    (tmp_path / "synthetic.tar.zst").unlink()
    with pytest.raises(GoldenCorpusError, match="missing"):
        corpus.payload(corpus.cases[0])


def test_golden_rejects_corrupt_shard(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)
    shard = tmp_path / "synthetic.tar.zst"
    shard.write_bytes(shard.read_bytes() + b"corruption")
    with pytest.raises(GoldenCorpusError, match="SHA-256"):
        corpus.payload(corpus.cases[0])


def test_golden_refresh_requires_explicit_acceptance(tmp_path: Path) -> None:
    make_corpus(tmp_path)
    with pytest.raises(SystemExit):
        main(["refresh", "--corpus", str(tmp_path)])
    with pytest.raises(SystemExit):
        main(["refresh", "--corpus", str(tmp_path), "--accept"])
    with pytest.raises(SystemExit):
        main(["refresh", "--corpus", str(tmp_path), "--accept", "--strategy", "global"])


def test_golden_classifies_fewer_layout_conversions() -> None:
    result = classify_golden_change(LAYOUT_BEFORE, LAYOUT_AFTER)
    assert result.classification == "fewer-conversions"
    assert result.expected_metrics["conversions"] == 1
    assert result.actual_metrics["conversions"] == 0


def test_golden_rejects_changed_observable_operations() -> None:
    result = classify_golden_change(LAYOUT_BEFORE, LAYOUT_AFTER.replace("tt.store", "tt.load"))
    assert result.classification == "regression"
    assert "observable" in result.reason


def test_golden_rejects_changed_computation() -> None:
    result = classify_golden_change(LAYOUT_BEFORE, LAYOUT_AFTER.replace("tensor<32xf32", "tensor<16xf32"))
    assert result.classification == "regression"
    assert "attributes" in result.reason


def test_golden_evidence_requires_exact_case_and_legacy_parity() -> None:
    payload = make_payload()
    evidence = make_evidence(payload, LAYOUT_BEFORE, LAYOUT_AFTER)
    validate_golden_evidence(payload, LAYOUT_BEFORE, LAYOUT_AFTER, evidence)
    evidence["legacy_unchanged"] = False
    with pytest.raises(GoldenCorpusError, match="unchanged legacy"):
        validate_golden_evidence(payload, LAYOUT_BEFORE, LAYOUT_AFTER, evidence)


def test_golden_evidence_rejects_unproven_layout_changes() -> None:
    payload = make_payload()
    candidate = LAYOUT_BEFORE.replace("#source> -> tensor<32xf32, #target>",
                                      "#source> -> tensor<32xf32, #source>")
    candidate = candidate.replace("tt.store %arg1, %1 : tensor<32xf32, #target>",
                                  "tt.store %arg1, %1 : tensor<32xf32, #source>")
    assert classify_golden_change(LAYOUT_BEFORE, candidate).classification == "layout-change"
    evidence = make_evidence(payload, LAYOUT_BEFORE, candidate, kind="gpu-benchmark")
    validate_golden_evidence(payload, LAYOUT_BEFORE, candidate, evidence)
    evidence["repetitions"] = 299
    with pytest.raises(GoldenCorpusError, match="300 CUDA-graph"):
        validate_golden_evidence(payload, LAYOUT_BEFORE, candidate, evidence)


def test_golden_evidence_rejects_insignificant_gpu_speedup() -> None:
    payload = make_payload()
    candidate = LAYOUT_BEFORE.replace("#source> -> tensor<32xf32, #target>",
                                      "#source> -> tensor<32xf32, #source>")
    candidate = candidate.replace("tt.store %arg1, %1 : tensor<32xf32, #target>",
                                  "tt.store %arg1, %1 : tensor<32xf32, #source>")
    evidence = make_evidence(payload, LAYOUT_BEFORE, candidate, kind="gpu-benchmark")
    evidence["candidate_samples_ns"] = [1040, 1080, 1030, 1065, 1055]
    with pytest.raises(GoldenCorpusError, match="statistically significant"):
        validate_golden_evidence(payload, LAYOUT_BEFORE, candidate, evidence)


def test_global_golden_refresh_preserves_legacy_and_records_history(monkeypatch: pytest.MonkeyPatch,
                                                                    tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path, ("triton",))
    case = corpus.cases[0]
    payload = corpus.payload(case)
    legacy = dict(payload["goldens"]["legacy"])
    payload["goldens"]["global"] = {
        "ttgir": LAYOUT_BEFORE, "sha256": hashlib.sha256(LAYOUT_BEFORE.encode()).hexdigest(),
        "metrics": ir_metrics(LAYOUT_BEFORE)
    }
    monkeypatch.setattr(ir_golden, "compile_case", lambda source, strategy: (LAYOUT_AFTER, ir_metrics(LAYOUT_AFTER)))
    evidence = make_evidence(payload, LAYOUT_BEFORE, LAYOUT_AFTER)
    assert corpus.refresh([case], strategy="global", evidence={case.case_id: evidence}) == 1
    refreshed = GoldenCorpus(tmp_path).payload(case)
    assert refreshed["goldens"]["legacy"] == legacy
    assert refreshed["goldens"]["global"]["ttgir"] == LAYOUT_AFTER
    history = json.loads((tmp_path / "global-golden-history.jsonl").read_text())
    assert history["case_id"] == case.case_id
    assert history["evidence"]["candidate_compiler_commit"] == "b" * 40


def test_global_refresh_updates_only_affected_standalone_checksums(monkeypatch: pytest.MonkeyPatch,
                                                                    tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path, ("triton",))
    destination = tmp_path / "triton-opt-reproducers"
    original = export_triton_opt_reproducers(corpus, corpus.cases, destination, workers=1)
    previous_checksum = original["shards"][0]["expected_sha256"]
    case = corpus.cases[0]
    payload = corpus.payload(case)
    payload["goldens"]["global"] = {
        "ttgir": LAYOUT_BEFORE, "sha256": hashlib.sha256(LAYOUT_BEFORE.encode()).hexdigest(),
        "metrics": ir_metrics(LAYOUT_BEFORE)
    }
    monkeypatch.setattr(ir_golden, "compile_case", lambda source, strategy: (LAYOUT_AFTER, ir_metrics(LAYOUT_AFTER)))
    evidence = make_evidence(payload, LAYOUT_BEFORE, LAYOUT_AFTER)
    assert corpus.refresh([case], strategy="global", evidence={case.case_id: evidence}) == 1
    updated = json.loads((destination / "manifest.json").read_text())
    assert updated["shards"][0]["expected_sha256"] != previous_checksum
    assert updated["source_manifest_sha256"] == hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest()
    assert (destination / updated["shards"][0]["checksum"]).read_text().startswith(
        updated["shards"][0]["expected_sha256"])


def test_golden_refresh_rejects_legacy_updates(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path, ("triton",))
    with pytest.raises(GoldenCorpusError, match="legacy is immutable"):
        corpus.refresh(corpus.cases, strategy="legacy", evidence={})


def test_golden_audit_filters_exact_case(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = make_corpus(tmp_path)
    case = corpus.select(language="triton")[0]
    assert main(["audit", "--corpus", str(tmp_path), "--case-id", case.case_id, "--workers", "1"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "cases": 1, "classifications": {"identical": 1}, "changes": []
    }


def test_golden_audit_accepts_multiple_exact_cases(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = make_corpus(tmp_path)
    first, second = corpus.cases
    assert main([
        "audit", "--corpus", str(tmp_path), "--case-id", first.case_id, "--case-id", second.case_id, "--workers", "1"
    ]) == 0
    assert json.loads(capsys.readouterr().out)["classifications"] == {"identical": 2}


def test_golden_filters_and_inventory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_corpus(tmp_path)
    assert main(["inventory", "--corpus", str(tmp_path), "--language", "gluon"]) == 0
    assert json.loads(capsys.readouterr().out) == {"cases": 1, "families": {"synthetic": 1}, "languages": {"gluon": 1}}


@pytest.mark.parametrize("language", ("triton", "gluon"))
@pytest.mark.parametrize("strategy", ("legacy", "global"))
def test_triton_opt_reproducer_captures_exact_compiler_pipeline(monkeypatch: pytest.MonkeyPatch, language: str,
                                                                strategy: str) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    payload = freeze_payload(make_payload(language))
    monkeypatch.setenv("TRITON_REPRODUCER_PATH", "previous-reproducer")
    reproducer, output = triton_opt_reproducer(payload, strategy)
    assert "mlir_reproducer" in reproducer
    assert "pipeline:" in reproducer
    assert ("gluon-inline" if language == "gluon" else "convert-triton-to-tritongpu") in reproducer
    if language == "triton":
        expected_pass = "tritongpu-optimize-layouts" if strategy == "global" else "tritongpu-remove-layout-conversions"
        assert expected_pass in reproducer
    assert normalize_ttgir(output) == payload["goldens"][strategy]["ttgir"]
    assert os.environ["TRITON_REPRODUCER_PATH"] == "previous-reproducer"


def test_triton_opt_reproducer_rejects_changed_golden() -> None:
    payload = freeze_payload(make_payload())
    payload["goldens"]["legacy"]["sha256"] = "incorrect"
    with pytest.raises(GoldenCorpusError, match="reference is corrupt"):
        triton_opt_reproducer(payload, "legacy")


@pytest.mark.parametrize("language", ("triton", "gluon"))
@pytest.mark.parametrize("compressed", (False, True))
def test_triton_opt_reproducer_runs_without_python(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, language: str,
                                                   compressed: bool) -> None:
    executable = os.environ.get("TRITON_OPT") or shutil.which("triton-opt")
    if executable is None:
        pytest.skip("triton-opt is unavailable")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    payload = freeze_payload(make_payload(language))
    reproductions = []
    for strategy in ("legacy", "global"):
        source, expected = triton_opt_reproducer(payload, strategy)
        reproductions.append(
            {"id": payload["id"], "strategy": strategy, "language": language, "source": source, "expected": expected})
    path = tmp_path / ("synthetic.mlir.zst" if compressed else "synthetic.mlir")
    descriptor = write_reproducer_shard(path, reproductions)
    assert descriptor["replays"] == 2
    result = replay_reproducer_shard(path, executable, expected_sha256=descriptor["expected_sha256"],
                                     compressed_sha256=descriptor["sha256"])
    assert result["sha256"] == descriptor["expected_sha256"]
    command = ('zstd -dc "$1" | "$2" --mlir-disable-threading --mlir-diagnostic-verbosity-level=errors '
               '--run-reproducer --split-input-file - | '
               'sha256sum --check "${1%.mlir.zst}.sha256"') if compressed else (
                   '"$2" --mlir-disable-threading --mlir-diagnostic-verbosity-level=errors '
                   '--run-reproducer --split-input-file "$1" | '
                   'sha256sum --check "${1%.mlir}.sha256"')
    pipeline = subprocess.run(["bash", "-o", "pipefail", "-c", command, "_",
                               str(path), executable], capture_output=True, text=True, check=False)
    assert pipeline.returncode == 0, pipeline.stderr
    assert "OK" in pipeline.stdout


def test_triton_opt_reproducer_detects_corrupt_input_and_output(tmp_path: Path) -> None:
    payload = freeze_payload(make_payload())
    source, expected = triton_opt_reproducer(payload, "legacy")
    path = tmp_path / "synthetic.mlir.zst"
    descriptor = write_reproducer_shard(
        path,
        [{"id": payload["id"], "strategy": "legacy", "language": "triton", "source": source, "expected": expected}])
    with pytest.raises(GoldenCorpusError, match="SHA-256"):
        replay_reproducer_shard(path, compressed_sha256="incorrect")
    executable = os.environ.get("TRITON_OPT") or shutil.which("triton-opt")
    if executable is not None:
        with pytest.raises(GoldenCorpusError, match="golden output changed"):
            replay_reproducer_shard(path, executable, expected_sha256="incorrect")
    assert descriptor["checksum"] == "synthetic.sha256"


def test_triton_opt_reproducer_canonicalizes_independent_deallocations(tmp_path: Path) -> None:
    expected = "  ttg.local_dealloc %2 : type\n  ttg.local_dealloc %1 : type\n"
    descriptor = write_reproducer_shard(
        tmp_path / "synthetic.mlir",
        [{"id": "synthetic", "strategy": "legacy", "language": "triton", "source": "module {}\n", "expected": expected}
         ])
    canonical = "  ttg.local_dealloc %1 : type\n  ttg.local_dealloc %2 : type\n\n"
    assert descriptor["expected_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()


def test_triton_opt_reproducer_covers_identical_layout_strategies_once(tmp_path: Path) -> None:
    payload = freeze_payload(make_payload("gluon"))
    source, expected = triton_opt_reproducer(payload, "legacy")
    path = tmp_path / "synthetic.mlir"
    descriptor = write_reproducer_shard(path, [{
        "id": payload["id"], "strategy": "legacy", "strategies": ["legacy", "global"], "language": "gluon", "source":
        source, "expected": expected
    }])
    assert descriptor["executions"] == 1
    assert descriptor["replays"] == 2
    assert json.loads(
        path.read_text().splitlines()[0].removeprefix("// IR-GOLDEN: "))["strategies"] == ["legacy", "global"]


def test_triton_opt_instrumented_gluon_preserves_distinct_layout_pipelines(tmp_path: Path) -> None:
    payload = make_payload("gluon")
    payload["metadata"]["instrumentation_mode"] = "consan"
    payload = freeze_payload(payload)
    reproductions = []
    for strategy in ("legacy", "global"):
        source, expected = triton_opt_reproducer(payload, strategy)
        reproductions.append(
            {"id": payload["id"], "strategy": strategy, "language": "gluon", "source": source, "expected": expected})
    assert "tritongpu-remove-layout-conversions" in reproductions[0]["source"]
    assert "tritongpu-optimize-layouts" in reproductions[1]["source"]
    descriptor = write_reproducer_shard(tmp_path / "instrumented.mlir", reproductions)
    assert descriptor["executions"] == descriptor["replays"] == 2


def test_triton_opt_export_balances_pipeline_batches(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus"
    corpus_path.mkdir()
    corpus = make_corpus(corpus_path)
    manifest = export_triton_opt_reproducers(corpus, corpus.cases, tmp_path / "balanced", max_replays_per_shard=1)
    assert manifest["replays"] == 4
    assert manifest["executions"] == 3
    assert len(manifest["shards"]) == 3
    assert max(shard["executions"] for shard in manifest["shards"]) == 1


def test_triton_opt_export_and_replay_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    executable = os.environ.get("TRITON_OPT") or shutil.which("triton-opt")
    if executable is None:
        pytest.skip("triton-opt is unavailable")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    make_corpus(corpus)
    destination = tmp_path / "reproducers"
    assert main(["export-opt", "--corpus", str(corpus), "--output", str(destination), "--workers", "1"]) == 0
    assert json.loads(capsys.readouterr().out) == {"cases": 2, "replays": 4, "executions": 3, "shards": 1}
    assert (destination / "replay-00000.mlir").is_file()
    assert main([
        "replay-opt", "--corpus",
        str(corpus), "--output",
        str(destination), "--workers", "1", "--triton-opt", executable
    ]) == 0
    assert json.loads(capsys.readouterr().out) == {"cases": 2, "replays": 4, "executions": 3, "shards": 1, "workers": 1}


def test_capture_listener_records_cache_hits_and_misses(tmp_path: Path) -> None:
    source = tmp_path / "copy.ttir"
    source.write_text(TTIR)
    captured = tmp_path / "captured"
    listener = compilation_capture_listener(captured)
    src = SimpleNamespace(language="triton", name="copy")
    for cache_hit in (False, True):
        listener(src=src, metadata={"hash": "kernel-hash", "target": TARGET}, metadata_group={"copy.ttir": source},
                 times={"total": 1}, cache_hit=cache_hit)
        event = json.loads((captured / "kernel-hash" / "capture.json").read_text())
        assert event["cache_hit"] is cache_hit
        assert (captured / "kernel-hash" / "copy.ttir").read_text() == TTIR


def test_capture_context_restores_existing_listener(tmp_path: Path) -> None:
    from triton import knobs

    previous = knobs.compilation.listener
    with capture_compilations(tmp_path) as listener:
        assert knobs.compilation.listener is listener
    assert knobs.compilation.listener is previous


def test_ir_metrics_count_observable_operations() -> None:
    text = 'ttg.convert_layout %0\n"tt.reduce"(%1)\ntt.load %2\ntt.store %3'
    assert ir_metrics(text) == {"conversions": 1, "reductions": 1, "loads": 1, "stores": 1}


def test_golden_normalization_stabilizes_independent_deallocations() -> None:
    first = "  ttg.local_dealloc %2 : type\n  ttg.local_dealloc %1 : type\n"
    second = "  ttg.local_dealloc %1 : type\n  ttg.local_dealloc %2 : type\n"
    assert normalize_ttgir(first) == normalize_ttgir(second)


def test_golden_signature_stabilizes_independent_pure_operations() -> None:
    first = ("  %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>\n"
             "  %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>\n"
             "  %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>\n"
             "  %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>\n"
             "  tt.store %arg0, %2 : tensor<1x8xi32>\n")
    second = ("  %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>\n"
              "  %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>\n"
              "  %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>\n"
              "  %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>\n"
              "  tt.store %arg0, %3 : tensor<1x8xi32>\n")
    assert ttgir_dependency_signature(first) == ttgir_dependency_signature(second)


def test_golden_signature_stabilizes_independent_constants_and_ssa_names() -> None:
    first = ("  %first = arith.constant 1 : i32\n"
             "  %second = arith.constant 2 : i32\n"
             "  %sum = arith.addi %first, %second : i32\n"
             "  tt.store %arg0, %sum : i32\n")
    second = ("  %x = arith.constant 2 : i32\n"
              "  %y = arith.constant 1 : i32\n"
              "  %z = arith.addi %y, %x : i32\n"
              "  tt.store %arg0, %z : i32\n")
    assert classify_golden_change(first, second).classification == "equivalent"


def test_golden_signature_preserves_layout_changes() -> None:
    first = "  %0 = ttg.convert_layout %arg0 : tensor<8xf32, #layout_a> -> tensor<8xf32, #layout_b>\n"
    second = "  %0 = ttg.convert_layout %arg0 : tensor<8xf32, #layout_a> -> tensor<8xf32, #layout_c>\n"
    assert ttgir_dependency_signature(first) != ttgir_dependency_signature(second)


@pytest.fixture(scope="session")
def production_golden_corpus() -> GoldenCorpus | None:
    root = os.environ.get("TRITON_PRODUCTION_IR_CORPUS")
    return GoldenCorpus(root) if root is not None else None


def test_external_production_golden_corpus(production_golden_corpus: GoldenCorpus,
                                           production_golden_case: object) -> None:
    assert production_golden_corpus is not None
    assert production_golden_case is not None
    verify_payload(production_golden_corpus.payload(production_golden_case))
