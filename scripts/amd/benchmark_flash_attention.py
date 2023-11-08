import argparse
import sys
import git


git_repo = git.Repo('.', search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
sys.path.insert(0, git_root+'/python/perf-kernels')
FA = __import__('06-fused-attention-fwd-transV')

attention = FA._attention.apply

import torch

def benchmark_FA(BATCH, H, N_CTX, D_HEAD, causal, rep, mode, dtype=torch.float16, device="cuda"):
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((BATCH, H, D_HEAD, N_CTX), dtype=dtype, device="cuda", requires_grad=True)
    sm_scale = 1.3
    split_kernel = True
    if mode == "bwd":
        causal=True
    fn = lambda: attention(q, k, v, sm_scale)

    o = fn()

    if mode == "bwd":
        do = torch.randn_like(o)
        o.backward(do, retain_graph=True)

    for i in range(rep):
        if mode == "bwd":
            o = fn()
            o.backward(do, retain_graph=True)
        if mode == "fwd":
            fn()

    torch.cuda.synchronize()


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="FA benchmarking",
        description="benchmark FA fwd and bwd with 2 GPUs",
        allow_abbrev=False,
    )

    parser.add_argument("-bs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-nheads", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-d", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-seqlen", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-rep", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-mode", type=str, default=argparse.SUPPRESS)

    parsed_args = parser.parse_args(args)

    bs = parsed_args.bs
    nheads = parsed_args.nheads
    d = parsed_args.d
    seqlen = parsed_args.seqlen
    rep = parsed_args.rep
    mode = parsed_args.mode

    benchmark_FA(bs, nheads, seqlen, d, False, rep, mode)


if __name__ == '__main__':
    sys.exit(main())
