import importlib.util
from pathlib import Path


def _load_build_helpers():
    module_path = Path(__file__).resolve().parents[2] / "build_helpers.py"
    spec = importlib.util.spec_from_file_location("build_helpers", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_download_and_copy_override_include_does_not_materialize_backend_tree(tmp_path, monkeypatch):
    build_helpers = _load_build_helpers()
    base_dir = tmp_path / "repo"
    include_src = tmp_path / "cuda-include"
    include_src.mkdir()
    (include_src / "cuda_runtime.h").write_text("// header\n")
    (include_src / "device").mkdir()
    (include_src / "device" / "special.h").write_text("// nested\n")

    monkeypatch.setattr(build_helpers, "get_base_dir", lambda: str(base_dir))

    helper_args = build_helpers.BuildHelperArgs(
        cache_path=str(tmp_path / "cache"),
        offline_build=True,
        llvm_system_suffix=None,
        llvm_syspath=None,
        json_syspath=None,
        ptxas_path=None,
        ptxas_blackwell_path=None,
        cuobjdump_path=None,
        nvdisasm_path=None,
        cudacrt_path=None,
        cudart_path=None,
        cupti_include_path=None,
        cupti_lib_path=None,
        cupti_lib_blackwell_path=None,
    )

    build_helpers.download_and_copy(
        name="nvcc",
        src_func=lambda *_args: "unused",
        dst_path="include",
        override_path=str(include_src),
        version="0",
        url_func=lambda *_args: "unused",
        helper_args=helper_args,
    )

    backend_include = base_dir / "third_party" / "nvidia" / "backend" / "include"
    assert not backend_include.exists()
