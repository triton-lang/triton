import triton
import re
import os


def test_triton_reproducer_path(monkeypatch, tmp_path):
    # If we get a cache hit there will be no reproducer generated
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    @triton.jit
    def triton_():
        return

    # We need an temp empty file for MLIR to write the reproducer to, and then
    # the TRITON_REPRODUCER_PATH env var enables crash the reproduction
    # generation in MLIR.
    repro_path = tmp_path / "repro_prefix"
    monkeypatch.setenv("TRITON_REPRODUCER_PATH", str(repro_path))

    # Run the kernel so MLIR will generate a crash reproducer. It doesn't really
    # matter what the kernel does, just that the PassManager runs its passes.
    triton_[(1, )]()

    stages = {
        'make_ttir': "triton-combine",
        'make_ttgir': "triton.*-coalesce",
        'make_llir': "convert-triton-.*gpu-to-llvm",
    }

    for stage_name, stage_pipeline_check in stages.items():
        assert os.path.exists(str(repro_path) + '.' + stage_name + '.repro.mlir')
        curr_repro_path = tmp_path / ("repro_prefix." + stage_name + ".repro.mlir")
        repro = curr_repro_path.read_text()
        assert "mlir_reproducer" in repro, f"Expected MLIR reproducer in {curr_repro_path}. Got:\n{repro}"
        m = re.search(r"pipeline: \"(.*" + stage_pipeline_check + ".*)\"", repro)
        assert m, "Expected to match pass pipeline after \"pipeline:\" in MLIR reproducer"
        pipeline_str = m.group(1)
        assert pipeline_str, "Expected non-empty pass pipeline in MLIR reproducer"
