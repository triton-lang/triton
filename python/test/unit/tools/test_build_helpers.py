import argparse
import importlib.util
import os
from pathlib import Path

BUILD_HELPERS_PATH = Path(__file__).resolve().parents[3] / "build_helpers.py"
SPEC = importlib.util.spec_from_file_location("triton_build_helpers", BUILD_HELPERS_PATH)
assert SPEC is not None
assert SPEC.loader is not None
build_helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_helpers)


def test_normalize_parsed_args_keeps_blackwell_ptxas_override(tmp_path):
    parser = argparse.ArgumentParser()
    build_helpers.add_common_args(parser)

    cache_dir = tmp_path / "cache"
    ptxas_dir = tmp_path / "toolchain"
    ptxas_dir.mkdir()

    parsed_args = parser.parse_args([
        "--triton-cache-path",
        str(cache_dir),
        "--triton-ptxas-path",
        str(ptxas_dir / "ptxas"),
        "--triton-ptxas-blackwell-path",
        str(ptxas_dir / "ptxas-blackwell"),
    ])

    helper_args = build_helpers.normalize_parsed_args(parsed_args)

    assert helper_args.cache_path == os.path.abspath(str(cache_dir))
    assert helper_args.ptxas_path == os.path.abspath(str(ptxas_dir / "ptxas"))
    assert helper_args.ptxas_blackwell_path == os.path.abspath(str(ptxas_dir / "ptxas-blackwell"))
