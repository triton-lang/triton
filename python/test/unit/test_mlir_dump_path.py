import subprocess
import tempfile
import pytest
import textwrap
from pathlib import Path
import os


@pytest.mark.forked
def test_basic_mlir_dump(monkeypatch):
    dump_dir = Path(tempfile.mkdtemp())
    monkeypatch.setenv("MLIR_ENABLE_DUMP", "1")
    monkeypatch.setenv("MLIR_DUMP_PATH", str(dump_dir))

    kernel_code = textwrap.dedent("""
        import triton
        import triton.language as tl
        import torch

        @triton.jit
        def dummy_kernel(x_ptr, y_ptr, n_elements: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * 128 + tl.arange(0, 128)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask)
            tl.store(y_ptr + offs, x, mask=mask)

        if __name__ == "__main__":
            n = 1024
            x = torch.arange(n, dtype=torch.float32, device="cuda")
            y = torch.empty_like(x)
            dummy_kernel[(n // 128,)](x, y, n_elements=n)
    """)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(kernel_code)
        script_path = f.name

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(
        ["python3", script_path],
        env=env,
        capture_output=True,
        text=True,
    )

    # Diagnostic printing if failure
    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

    assert result.returncode == 0, "Triton kernel script failed"
    assert any(f.suffix == ".mlir" for f in dump_dir.iterdir()), "No MLIR dump generated"

    os.remove(script_path)
