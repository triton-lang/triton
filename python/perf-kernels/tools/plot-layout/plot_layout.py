import argparse
import os
import sys

from blocked import generate_blocked_tex
from dot import generate_dot_tex
from lds import generate_lds_tex
from utils import run_bash_command, OneLineFormatter
from wmma import generate_wmma_tex


def parse_args():
    top_parser = argparse.ArgumentParser(
        prog="Draw triton layouts",
        allow_abbrev=False,
    )
    top_parser.add_argument("--output", type=str, default="myplot", help='output pdf file name (without surfix)')
    top_parser.add_argument("--keep", action='store_true', default=False, help='If set, keep the generated .tex file')
    top_parser.add_argument("--force", action='store_true', default=False,
                            help='If set, overwrite the pdf file with the same name')
    subparsers = top_parser.add_subparsers(dest="plot_type", metavar="PLOT_TYPE", required=False, title="subcommands",
                                           description="Choose to plot blocked, lds, dot or wmma",
                                           help="Choose one of the four plot mode")
    # blocked layout parameters
    blocked_parser = subparsers.add_parser(
        "blocked",
        allow_abbrev=False,
        help="plot blocked layout for global memory access",
        formatter_class=lambda prog: OneLineFormatter(prog, max_help_position=40),
    )
    # tensor shapes
    blocked_parser.add_argument("-r", "--rowName", type=str, default="M", metavar="ROW", help='tensor dim0 name')
    blocked_parser.add_argument("-c", "--colName", type=str, default="K", metavar="COL", help='tensor dim1 name')
    blocked_parser.add_argument("-B", "--matrixB", action="store_true",
                                help='shortcut to plot operand B with dimension name of (K, N)')
    blocked_parser.add_argument("-s", "--sizePerThread", type=int, nargs=2, default=(1, 4), metavar=("s0", "s1"),
                                help="how many elements each thread holds in the 2D block per CTA")
    blocked_parser.add_argument("-t", "--threadsPerWarp", type=int, nargs=2, default=(16, 4), metavar=("t0", "t1"),
                                help="how thread is partitioned into a 2D grid in a warp with 64 threads")
    blocked_parser.add_argument("-w", "--warpsPerCTA", type=int, nargs=2, default=(1, 4), metavar=("w0", "w1"),
                                help="how warps tile a CTA")
    blocked_parser.add_argument("-o", "--order", type=int, nargs=2, default=(1, 0), metavar=("minor", "major"),
                                help="order from most minor to most major")
    blocked_parser.add_argument(
        "-b", "--blockShape", type=int, nargs=2, metavar=("b0", "b1"),
        help='block size (dim0, dim1) of the tile. If not specified it presumably equals to the shape of CTA')
    ## dot layout parameters
    dot_parser = subparsers.add_parser(
        "dot",
        allow_abbrev=False,
        help="plot dot layout for MFMA",
        formatter_class=lambda prog: OneLineFormatter(prog, max_help_position=50),
    )
    dot_parser.add_argument("--dotShape", type=int, nargs=3, default=(32, 128, 64), metavar=("M", "N", "K"),
                            help='Dot op shape in the form of M, N, K')
    dot_parser.add_argument("--warpsPerCTA", type=int, nargs=2, default=(1, 4), metavar=("w0", "w1"),
                            help="how warps tile the dot result matrix")
    dot_parser.add_argument("--nonKDim", type=int, default=16, choices=[16, 32],
                            help='mfma instruction dimension of M/N')
    dot_parser.add_argument("--kWidth", type=int, default=4, choices=[4, 8, 16, 32],
                            help='number of contiguous elements each thread owns during MFMA')
    dot_parser.add_argument("--kGroup", type=int, default=1, choices=[1, 2],
                            help='total number of elements / kWidth per mfma instruction')
    dot_parser.add_argument("--dtypeA", type=str, default='fp16',
                            choices=['fp16', 'bf16', 'fp8', 'bf8', 'fp6', 'bf6', 'f4',
                                     'i8'], help='element type of operand A')
    dot_parser.add_argument("--dtypeB", type=str, default='fp16',
                            choices=['fp16', 'bf16', 'fp8', 'bf8', 'fp6', 'bf6', 'f4',
                                     'i8'], help='element type of operand B')
    dot_parser.add_argument("--mfmaTrans", action='store_true', default=False,
                            help='If set, then use mfma.trans layout')
    dot_parser.add_argument("--scale", action='store_true', default=False,
                            help='If set, plot the scale tensor for mfma_f8f6f4 instructions')
    ## LDS access parameters
    lds_parser = subparsers.add_parser(
        "lds",
        allow_abbrev=False,
        help="plot LDS (shared memory) layout",
        formatter_class=lambda prog: OneLineFormatter(prog, max_help_position=50),
    )
    lds_parser.add_argument("--tensorShape", type=int, nargs=2, default=(128, 64),
                            help='2D block shape in the form of (dim0, dim1)')
    lds_parser.add_argument("--kWidth", type=int, default=4, choices=[4, 8, 16, 32],
                            help='number of contiguous elements per thread')
    lds_parser.add_argument("--dtype", type=str, default='fp16',
                            choices=['fp16', 'bf16', 'fp8', 'bf8', 'fp6', 'bf6', 'f4',
                                     'i8'], help='element type of tensor to be stored in LDS')
    lds_parser.add_argument("--nonKDim", type=int, default=16, choices=[16, 32], help='mfma instruction dim')
    lds_parser.add_argument("--banks", type=int, default=32, choices=[32, 64], help='choose the number of banks in LDS')
    lds_parser.add_argument("--layout", type=str, default="none", choices=['swizzle', 'padding', 'none'],
                            help='choose the LDS data layout')
    lds_parser.add_argument("--access", type=str, default="none", choices=['read', 'write', 'none'],
                            help='choose LDS access mode')
    lds_parser.add_argument("--mnContig", action='store_true', default=False,
                            help='If set, the tensor is K x N and n-contig')
    lds_parser.add_argument("--mfma-trans-load", action='store_true', default=False,
                            help='If set, use MFMA transpose load instructions')
    lds_parser.add_argument("--swizzleVec", type=int, default=4, choices=[4, 8, 16, 32],
                            help='number of contiguous elements in a vector to swizzle')
    lds_parser.add_argument("--padInterval", type=int, default=1, help='Add padding for every padInterval bytes')
    lds_parser.add_argument("--padAmount", type=int, default=0, help='Pad padAmount bytes for every padInterval bytes')
    ## wmma instruction layout parameter
    wmma_parser = subparsers.add_parser("wmma", help="plot dot layout for wmma")
    wmma_parser.add_argument("--wave-size", type=int, default=32, choices=[32, 64],
                             help='choose the wmma instruction mode')

    # workaround for top-level parser's flag passes in after subparser
    args, remaining = top_parser.parse_known_args()
    args = top_parser.parse_args(remaining, namespace=args)

    return args


def main():
    args = parse_args()
    force = args.force
    ofilename = args.output
    keepSrc = args.keep

    if os.path.exists(f"{ofilename}.pdf"):
        if not force:
            print(f"{ofilename}.pdf exists already. Please use --force to overwrite or change output name")
            sys.exit(0)
        else:
            print(f"{ofilename}.pdf exists but overwritten!")

    match args.plot_type:
        case "blocked":
            generate_blocked_tex(args)
        case "dot":
            generate_dot_tex(args)
        case "lds":
            generate_lds_tex(args)
        case "wmma":
            generate_wmma_tex(args)
        case _:
            raise NotImplementedError(f"Only blocked, dot, lds and wmma supported, you entered {args.plot_type}")

    ret = run_bash_command(f"pdflatex -halt-on-error -jobname {ofilename} myplot.tex")
    if ret == 0:
        print(f"plot saved in {ofilename}.pdf")
        # Remove auxiliary files
        os.remove(f"{ofilename}.aux")
        os.remove(f"{ofilename}.log")
        if not keepSrc:
            os.remove("myplot.tex")
            run_bash_command("rm -rf ./auto")
    else:
        print("pdflatex has problem generating the pdf file. Check myplot.log for the error!!!")
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())
