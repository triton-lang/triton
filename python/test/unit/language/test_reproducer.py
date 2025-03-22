import triton
import re


def test_triton_reproducer_path(monkeypatch, tmp_path):
    # If we get a cache hit there will be no reproducer generated
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    @triton.jit
    def triton_():
        return

    # We need an temp empty file for MLIR to write the reproducer to, and then
    # the TRITON_REPRODUCER_PATH env var enables crash the reproduction
    # generation in MLIR.
    repro_path = tmp_path / "repro.mlir"
    repro_path.touch()
    monkeypatch.setenv("TRITON_REPRODUCER_PATH", str(repro_path))

    # Run the kernel so MLIR will generate a crash reproducer. It doesn't really
    # matter what the kernel does, just that the PassManager runs its passes.
    triton_[(1, )]()

    repro = repro_path.read_text()
    assert "mlir_reproducer" in repro, f"Expected MLIR reproducer in {repro_path}. Got:\n{repro}"
    m = re.search(r"pipeline: \"(.*)\"", repro)
    assert m, "Expected to match pass pipeline after \"pipeline:\" in MLIR reproducer"
    pipeline_str = m.group(1)
    assert pipeline_str, "Expected non-empty pass pipeline in MLIR reproducer"
