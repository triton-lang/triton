import argparse
import sys
import os
import subprocess


def draw_preamble_cmd():
    return '''\\documentclass[tikz, border=1mm, dvipsnames]{standalone}
\\usepackage{ifthen}
\\usepackage{tikz}
\\usetikzlibrary{arrows.meta,arrows}
\\usetikzlibrary{intersections}
\\usetikzlibrary{calc, quotes}
\\usetikzlibrary{patterns}
\\usepackage{xparse}

\\ExplSyntaxOn
\\NewExpandableDocumentCommand{\\bitwiseXor}{mm}
 {
  \\recuenco_bitwise_xor:nn { #1 } { #2 }
 }

\\cs_new:Nn \\recuenco_bitwise_xor:nn
 {
  \\int_from_bin:e
   {
    \\__recuenco_bitwise_xor:ee { \\int_to_bin:n { #1 } } { \\int_to_bin:n { #2 } }
   }
 }
\\cs_generate_variant:Nn \\int_from_bin:n { e }

\\cs_new:Nn \\__recuenco_bitwise_xor:nn
 {
  \\__recuenco_bitwise_xor_binary:ee
   {
    \\prg_replicate:nn
     {
      \\int_max:nn { \\tl_count:n { #1 } } { \\tl_count:n { #2 } } - \\tl_count:n { #1 }
     }
     { 0 }
     #1
   }
   {
    \\prg_replicate:nn
     {
      \\int_max:nn { \\tl_count:n { #1 } } { \\tl_count:n { #2 } } - \\tl_count:n { #2 }
     }
     { 0 }
     #2
   }
 }
\\cs_generate_variant:Nn \\__recuenco_bitwise_xor:nn { ee }

\\cs_new:Nn \\__recuenco_bitwise_xor_binary:nn
 {
  \\__recuenco_bitwise_xor_binary:w #1;#2;
 }
\\cs_generate_variant:Nn \\__recuenco_bitwise_xor_binary:nn { ee }

\\cs_new:Npn \\__recuenco_bitwise_xor_binary:w #1#2;#3#4;
 {
  \\int_abs:n { #1-#3 }
  \\tl_if_empty:nF { #2 } { \\__recuenco_bitwise_xor_binary:w #2;#4; }
 }

\\ExplSyntaxOff'''


def draw_dot_layout_cmd(M, N, K, mfmaNonKDim, warpsPerCTA, trans, kpack):
    return f'''\\begin{{document}}
  \\begin{{tikzpicture}}
    \\def\\scale{{1}}
    \\def\\elem{{0.04}}
    \\coordinate (C TL) at (0,0);
    \\def\\opColorAL{{magenta}}
    \\def\\opColorAR{{cyan}}
    \\def\\opColorBL{{Maroon}}
    \\def\\opColorBR{{BlueGreen}}
    \\drawDot{{{M}}}{{{N}}}{{{K}}}{{{mfmaNonKDim}}}{{{warpsPerCTA[0]}}}{{{warpsPerCTA[1]}}}{{{trans}}}{{{kpack}}}

    \\coordinate (C TL) at ($(C TL)+({N}*\elem+32*\elem, 0)$);
    \\def\\mfmaTrans{{{trans}}}

    %% Draw zoomed in view of mfma
    \\def\\elem{{.16}}
    \\pgfmathsetmacro{{\\gap}}{{\\elem*5}}
    \\pgfmathsetmacro{{\\nonTrans}}{{1-\\mfmaTrans}}
    \\pgfmathsetmacro{{\\groups}}{{64/{mfmaNonKDim}}}
    \\coordinate (C TL) at ($(C TL)+(.5*\\gap+1.2*\\nonTrans*\\gap+\\groups*{kpack}*\\elem, 0)$);
    \\drawMFMAInstr{{{mfmaNonKDim}}}{{{kpack}}}{{\\mfmaTrans}}

  \\end{{tikzpicture}}
\\end{{document}}'''


def draw_blocked_layout_cmd(M, K, sizePerThread, threadsPerWarp, warpsPerCTA, order):
    return f'''\\begin{{document}}
  \\begin{{tikzpicture}}
    \\def\\scale{{1}}
    \\def\\elem{{0.06}}
    \\coordinate (TL) at (0,0);
    \\drawBlockedTensor{{{M}}}{{{K}}}{{{sizePerThread[0]}}}{{{sizePerThread[1]}}}{{{threadsPerWarp[0]}}}{{{warpsPerCTA[0]}}}{{{warpsPerCTA[1]}}}{{{order[0]}}}
  \\end{{tikzpicture}}
\\end{{document}}'''


def draw_lds_access_cmd(M, K, kpack, ldsLayout, ldsAccess, sizePerThread, threadsPerWarp):
    if ldsLayout == 'swizzle':
        hasSwizzle = 1
    elif ldsLayout == 'padding':
        hasSwizzle = 2
    else:
        hasSwizzle = 0

    if ldsAccess == 'read':
        accessMode = 1
    elif ldsAccess == 'write':
        accessMode = 2
    else:
        accessMode = 0

    return f'''\\begin{{document}}
  \\begin{{tikzpicture}}
    \\def\\scale{{1}}
    \\def\\M{{{M}}}
    \\def\\K{{{K}}}
    \\def\\vec{{{kpack}}}
    \\def\\hasSwizzle{{{hasSwizzle}}}
    \\def\\accessMode{{{accessMode}}}

    \\def\\sizePerThreadK{{{sizePerThread[1]}}}
    \\def\\sizePerThreadM{{{sizePerThread[0]}}}
    \\def\\threadsPerWarpK{{{threadsPerWarp[1]}}}

    \\def\\elem{{0.18}}
    \\coordinate (TL) at (0,0);
    \\drawTensorLayoutGlobalMem
    \\coordinate (TL) at ($(TL)+(0, -24*\\elem-10*\\elem)$);
    \\drawLDSLayoutTritonSwizzling{{\\hasSwizzle}}{{\\accessMode}}
  \\end{{tikzpicture}}
\\end{{document}}'''


def draw_wmma_instr_cmd(waveSize):
    wmma_mode = 0 if waveSize == 32 else 1
    return f'''\\begin{{document}}
  \\begin{{tikzpicture}}
    \\def\\scale{{1}}
    \\coordinate (C TL) at (0,0);
    \\def\\elem{{0.25}}
    \\drawWMMAInstr{{{wmma_mode}}}{{1}}
  \\end{{tikzpicture}}
\\end{{document}}'''


def run_bash_command(commandstring):
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE)
    return proc.stdout.splitlines()


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Draw triton layouts",
        allow_abbrev=False,
    )
    ## tensor shapes
    parser.add_argument("-shape", type=int, nargs=3, default=(32, 128, 64), help='Tensor shape in the form of M,N,K')
    parser.add_argument("-plot", type=str, default="blocked", choices=['blocked', 'dot', 'wmma', 'lds'],
                        help='choose plot mode')
    parser.add_argument("-nonKDim", type=int, default=32, choices=[16, 32], help='mfma instruction dim')
    ## blocked layout parameters
    parser.add_argument("-sizePerThread", type=int, nargs=2, default=(1, 4))
    parser.add_argument("-threadsPerWarp", type=int, nargs=2, default=(16, 4))
    parser.add_argument("-warpsPerCTA", type=int, nargs=2, default=(1, 4))
    parser.add_argument("-order", type=int, nargs=2, default=(1, 0))
    ## LDS access parameters
    parser.add_argument("-kWidth", type=int, default=4, choices=[4, 8, 16], help='number of elements per thread')
    parser.add_argument("-lds_layout", type=str, default="none", choices=['swizzle', 'padding', 'none'],
                        help='choose the LDS data layout')
    parser.add_argument("-lds_access", type=str, default="none", choices=['read', 'write', 'none'],
                        help='choose LDS access mode')
    ## wmma instruction layout parameter
    parser.add_argument("-wave_size", type=int, default=32, choices=[32, 64], help='choose the wmma instruction mode')

    parser.add_argument("-o", type=str, default="myplot", help='output pdf file name (without surfix)')
    parser.add_argument("-mfmaTrans", action='store_true', default=False, help='If set, then use mfma.trans layout')
    parser.add_argument("-keep", action='store_true', default=False, help='If set, keep the generated .tex file')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    shape = args.shape
    M = shape[0]
    N = shape[1]
    K = shape[2]
    plot_mode = args.plot
    mfmaNonKDim = args.nonKDim
    kpack = args.kWidth
    trans = 1 if args.mfmaTrans else 0
    ofilename = args.o
    keepSrc = args.keep

    ldsLayout = args.lds_layout
    ldsAccess = args.lds_access

    waveSize = args.wave_size

    sizePerThread = args.sizePerThread
    threadsPerWarp = args.threadsPerWarp
    warpsPerCTA = args.warpsPerCTA
    order = args.order

    CTAShape = []
    if plot_mode == 'blocked':
        print(f"Plotting tensor M={M},K={K} with blocked layout:")
        print(f"sizePerThread={sizePerThread}", end=" ")
        print(f"threadsPerWarp={threadsPerWarp}", end=" ")
        print(f"warpsPerCTA={warpsPerCTA}", end=" ")
        print(f"order={order}", end=" ")
        CTAShape.append(sizePerThread[0] * threadsPerWarp[0] * warpsPerCTA[0])
        CTAShape.append(sizePerThread[1] * threadsPerWarp[1] * warpsPerCTA[1])

    if plot_mode == 'dot':
        mfma_inst_str = "mfma_32x32" if mfmaNonKDim == 32 else "mfma_16x16"
        mfma_trans_str = ".trans" if trans else ""
        print(f"Plotting dot operation with shapes M={M},N={N},K={K}")
        print("MFMA: " + mfma_inst_str + mfma_trans_str + f" kWidth = {kpack}", end=" ")
        print(f"warpsPerCTA={warpsPerCTA}", end=" ")
        CTAShape.append(mfmaNonKDim * warpsPerCTA[0])
        CTAShape.append(mfmaNonKDim * warpsPerCTA[1])

    if plot_mode == 'blocked' or plot_mode == 'dot':
        print(f"CTAShape={CTAShape}")
        assert M != 0 and CTAShape[0] <= M and M % CTAShape[0] == 0, "bad tensor dimension M"

    if plot_mode == 'blocked':
        assert K != 0 and CTAShape[1] <= K and K % CTAShape[1] == 0, "bad tensor dimension K"

    if plot_mode == 'dot':
        assert N != 0 and CTAShape[1] <= N and N % CTAShape[1] == 0, "bad tensor dimension N"
        assert K != 0 and K % (2 * kpack) == 0, "bad tensor dimension K"

    if plot_mode == 'lds':
        print(f"Plotting LDS access for tensor M={M},K={K} with vec={kpack}")
        if ldsAccess == 'write':
            print(f"sizePerThread={sizePerThread}, threadsPerWarp={threadsPerWarp}")

    with open("myplot.tex", 'w') as f_plot:
        with open("tikzplot.tex") as file:
            tikz_code = file.read()

        preamble_str = draw_preamble_cmd()

        draw_blockedLayout_str = draw_blocked_layout_cmd(M, K, sizePerThread, threadsPerWarp, warpsPerCTA, order)

        draw_dotLayout_str = draw_dot_layout_cmd(M, N, K, mfmaNonKDim, warpsPerCTA, trans, kpack)

        draw_lds_str = draw_lds_access_cmd(M, K, kpack, ldsLayout, ldsAccess, sizePerThread, threadsPerWarp)

        draw_wmma_str = draw_wmma_instr_cmd(waveSize)

        f_plot.write(preamble_str + "\n")
        f_plot.write(tikz_code)
        if plot_mode == 'blocked':
            f_plot.write(draw_blockedLayout_str)
        elif plot_mode == 'dot':
            f_plot.write(draw_dotLayout_str)
        elif plot_mode == 'lds':
            f_plot.write(draw_lds_str)
        elif plot_mode == 'wmma':
            f_plot.write(draw_wmma_str)

    run_bash_command(f"pdflatex -jobname {ofilename} myplot.tex")
    print(f"plot saved in {ofilename}.pdf")

    ## Remove au files
    os.remove(f"{ofilename}.aux")
    os.remove(f"{ofilename}.log")
    if not keepSrc:
        os.remove("myplot.tex")
        run_bash_command("rm -rf ./auto")


if __name__ == '__main__':
    sys.exit(main())
