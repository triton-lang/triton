"""
Script for running one Matrix Multiplication kernel config at a time
"""

import argparse
import re
import sys
import tune_gemm

def parse_args():
    parser = argparse.ArgumentParser(
        prog="check corectness of particular config for tuning gemm script",
        allow_abbrev=False,
    )

    parser.add_argument("-m", type=int, default=0)
    parser.add_argument("-n", type=int, default=0)
    parser.add_argument("-k", type=int, default=0)
    parser.add_argument("--block_m", type=int, default=0)
    parser.add_argument("--block_n", type=int, default=0)
    parser.add_argument("--block_k", type=int, default=0)
    parser.add_argument("--group_m", type=int, default=0)
    parser.add_argument("--split_k", type=int, default=0)
    parser.add_argument("--num_warps", type=int, default=0)
    parser.add_argument("--num_stages", type=int, default=0)
    parser.add_argument("--waves_per_eu", type=int, default=0)

    parser.add_argument("--config_str", type=str, default="", help="can take from gemm_tune.py script output, looks like M16_N8_K128_BM64_BN64_BK64_GM1_SK2_nW2_nS0_EU0")
    args = parser.parse_args()

    return args


def parse_config(cfg_str):
    values = cfg_str.split("_")
    config_name = {"M": "M",
                   "N": "N",
                   "K": "K",
                   "BM": "BLOCK_SIZE_M",
                   "BN": "BLOCK_SIZE_N",
                   "BK": "BLOCK_SIZE_K",
                   "GM": "GROUP_SIZE_M",
                   "SK": "SPLIT_K",
                   "nW": "num_warps",
                   "nS": "num_stages",
                   "EU": "waves_per_eu",
                  }
    config = {}
    for val in values:
        match = re.search("([a-zA-Z]*)([0-9]*)", val)
        if match:
            cfg_field_name = config_name[match.group(1)]
            config[cfg_field_name] = int(match.group(2))
    return config


def main():
    args = parse_args()
    if args.config_str:
        config = parse_config(args.config_str)
    else:
        config = {"M": args.m,
                  "N": args.n,
                  "K": args.k,
                  "BLOCK_SIZE_M": args.block_m,
                  "BLOCK_SIZE_N": args.block_n,
                  "BLOCK_SIZE_K": args.block_k,
                  "GROUP_SIZE_M": args.group_m,
                  "SPLIT_K": args.split_k,
                  "num_warps": args.num_warps,
                  "num_stages": args.num_stages,
                  "waves_per_eu": args.waves_per_eu,
                 }
    tune_gemm.test_correctness(config["M"], config["N"], config["K"], config, verbose=True)


if __name__ == "__main__":
    sys.exit(main())
