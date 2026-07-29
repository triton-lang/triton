from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton.tools.ir_golden import (SCHEMA_VERSION, GoldenCorpus, GoldenCorpusError, capture_compilations,
                                    compilation_capture_listener, freeze_payload, function_names, ir_metrics, main,
                                    normalize_source, normalize_ttgir, ttgir_dependency_signature, verify_payload,
                                    write_shard)

TARGET = {"backend": "cuda", "arch": 80, "warp_size": 32}
TTIR = 'module { tt.func public @copy(%arg0: !tt.ptr<f32>) { tt.return } }'
GLUON = ('module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, '
         'ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} '
         '{ tt.func public @copy(%arg0: !tt.ptr<f32>) { tt.return } }')


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
    assert main(["refresh", "--corpus", str(tmp_path), "--accept"]) == 0


def test_golden_filters_and_inventory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_corpus(tmp_path)
    assert main(["inventory", "--corpus", str(tmp_path), "--language", "gluon"]) == 0
    assert json.loads(capsys.readouterr().out) == {"cases": 1, "families": {"synthetic": 1}, "languages": {"gluon": 1}}


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
