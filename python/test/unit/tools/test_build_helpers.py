import argparse
import os

from build_helpers import add_common_args
from build_helpers import normalize_parsed_args


def test_normalize_parsed_args_keeps_blackwell_ptxas_override(tmp_path):
    parser = argparse.ArgumentParser()
    add_common_args(parser)

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

    helper_args = normalize_parsed_args(parsed_args)

    assert helper_args.cache_path == os.path.abspath(str(cache_dir))
    assert helper_args.ptxas_path == os.path.abspath(str(ptxas_dir / "ptxas"))
    assert helper_args.ptxas_blackwell_path == os.path.abspath(str(ptxas_dir / "ptxas-blackwell"))
