from dataclasses import dataclass
from pathlib import Path


@dataclass
class DotConfig:
    mfmaNonKDim: int
    kWidth: int
    kGroup: int
    trans: int
    warpsPerCTA: tuple


matrixFormatTable = {'fp8': 0, 'bf8': 1, 'fp6': 2, 'bf6': 3, 'f4': 4}


def matrixFormat(dtypeA, dtypeB):
    """
    return CBSZ and BLGP according to data types
    b000: E4M3(FP8)
    b001: E5M2(BF8)
    b010: E2M3(FP6)
    b011: E3M2(BF6)
    b100: E2M1(FP4)
    """
    return matrixFormatTable[dtypeA], matrixFormatTable[dtypeB]


def isType4Or6Bit(dtype):
    return dtype == 'fp6' or dtype == 'bf6' or dtype == 'f4'


def isType8BitFloat(dtype):
    return dtype == 'fp8' or dtype == 'bf8'


def isType16Bit(dtype):
    return dtype == 'bf16' or dtype == 'fp16'


def isMixedPrecType(dtype):
    return isType8BitFloat(dtype) or isType4Or6Bit(dtype)


def isMixedPrecBtwF8AndF4OrF6(dtypeA, dtypeB):
    return (isType8BitFloat(dtypeA) and isType4Or6Bit(dtypeB)) or \
           (isType8BitFloat(dtypeB) and isType4Or6Bit(dtypeA))


def draw_dot_layout_cmd(M, N, K, dtypeA, dtypeB, mfma_inst_str, isMixed864, plot_scale, dotConfig):
    mfmaNonKDim = dotConfig.mfmaNonKDim
    warpsPerCTA = dotConfig.warpsPerCTA
    trans = 1 if dotConfig.trans else 0
    kWidth = dotConfig.kWidth
    kGroup = dotConfig.kGroup
    scaleLabel = 0.7 if (kWidth == 4 or (kWidth == 8 and mfmaNonKDim == 32)) else 1

    outType = 'i32' if dtypeA == 'i8' else 'f32'
    kWidth_a = kWidth_b = kWidth
    kGroup_a = kGroup_b = kGroup
    if isMixed864:
        if isType8BitFloat(dtypeA):
            kWidth_a = 16
            kGroup_a = 2
            kWidth_b = 32
            kGroup_b = 1
        else:
            kWidth_a = 32
            kGroup_a = 1
            kWidth_b = 16
            kGroup_b = 2
    kWidth_left = kWidth_b if trans else kWidth_a
    kGroup_left = kGroup_b if trans else kGroup_a

    elemSmall = 0.04
    elemLarge = 0.16
    elemPerThread = kWidth_a * kGroup_a
    if elemPerThread == 16:
        ratio = 0.8
    elif elemPerThread == 32:
        ratio = 0.6
    else:
        ratio = 1
    elemWidth = elemLarge * ratio

    scaling = 1 if plot_scale else 0

    return f"""\\begin{{document}}
               \\begin{{tikzpicture}}
               \\def\\scale{{1}}
               \\def\\elem{{{elemSmall}}}
               \\def\\elemW{{\\elem}}
               \\def\\kWidthA{{{kWidth_a}}}
               \\def\\kWidthB{{{kWidth_b}}}
               \\def\\kGroupA{{{kGroup_a}}}
               \\def\\kGroupB{{{kGroup_b}}}
               \\coordinate (C TL) at (0,0);
               \\drawDot{{{M}}}{{{N}}}{{{K}}}{{{mfmaNonKDim}}}{{{warpsPerCTA[0]}}}{{{warpsPerCTA[1]}}}{{{trans}}}

               \\coordinate (C TL) at ($(C TL)+({N}*\elem+32*\elem, 0)$);
               \\def\\mfmaTrans{{{trans}}}

               %% Draw zoomed in view of mfma
               \\def\\scaleLabel{{{scaleLabel}}}
               \\pgfmathsetmacro{{\\oldElem}}{{\\elem}}
               \\def\\elem{{{elemLarge}}}
               \\def\\elemW{{{elemWidth}}}
               \\pgfmathsetmacro{{\\gap}}{{\\elem*5}}
               \\pgfmathsetmacro{{\\nonTrans}}{{1-\\mfmaTrans}}
               \\pgfmathsetmacro{{\\groups}}{{64/{mfmaNonKDim}}}
               \\coordinate (C TL) at ($(C TL)+({scaling}*0.3*\\gap+{scaling}*\\groups*4*\elemW+.5*\\gap+1.2*\\nonTrans*\\gap+\\groups*{kWidth_left}*{kGroup_left}*\\elemW, -{M}*\\oldElem+{mfmaNonKDim}*\\elem)$);
               \\coordinate (mfma instr) at ($(C TL)+(-.5*\\gap-0.6*\\nonTrans*\\gap-0.4*\\mfmaTrans*\\gap, 1.5*\\gap+.5*\\mfmaTrans*\\gap)$);
               \\node [scale=\scaleLabel, above left, align=left, draw=black, fill=white] at (mfma instr) {{{mfma_inst_str}}};
               \\drawMFMAInstr{{{mfmaNonKDim}}}{{\\mfmaTrans}}{{{dtypeA}}}{{{dtypeB}}}{{{outType}}}{{{scaling}}}

               \\end{{tikzpicture}}
               \\end{{document}}"""


def checkMfmaValidity(mfmaNonKDim, kWidth, kGroup, dtypeA, dtypeB, trans, scale):
    # Check input types
    # Mixed precision is only allowed within f8, f6 and f4
    assert (isMixedPrecType(dtypeA) and isMixedPrecType(dtypeB)) or \
           (dtypeA == dtypeB), \
           f"Cannot do mixed precision mfma with {dtypeA} and {dtypeB}"
    """
    Check mfma size according to data types
    * refers to newly added instructions on gfx950
    Both dtyes are f4 or fp6 or bf6
      *mfma_f32_16x16x128_f8f6f4: kWidth = 32, kGroup = 1
      *mfma_f32_32x32x64_f8f6f4: kWidth = 32, kGroup = 1
    One dtype is fp8 or bf8
      When the other operand is f4, fp6, or bf6
        *mfma_f32_16x16x128_f8f6f4: kWidth = 16, kGroup = 2
        *mfma_f32_32x32x64_f8f6f4: kWidth = 16, kGroup = 2
      When the other operand is fp8 or bf8
        *mfma_f32_16x16x128_f8f6f4: kWidth = 16, kGroup = 2
        mfma_f32_16x16x32_fp8/bf8_fp8/bf8: kWidth = 16, kGroup = 1, kpack=2
        mfma_f32_16x16x32_fp8/bf8_fp8/bf8: kWidth = 8, kGroup = 1
        *mfma_f32_32x32x64_f8f6f4: kWidth = 16, kGroup = 2
        mfma_f32_32x32x16_fp8/bf8_fp8/bf8: kWidth = 16, kGroup = 1, kpack=2
        mfma_f32_32x32x16_fp8/bf8_fp8/bf8: kWidth = 8, kGroup = 1
    Both dtypes are bf16 or bf16
        *mfma_f32_16x16x32_f16/bf16: kWidth = 8, kGroup = 1
        mfma_f32_16x16x16_f16/bf16: kWidth = 4, kGroup = 1
        *mfma_f32_32x32x16_f16/bf16: kWidth = 8, kGroup = 1
        mfma_f32_32x32x8_f16/bf16: kWidth = 4, kGroup = 1
    Both types are i8
        *mfma_i32_16x16x64_i8: kWidth = 16, kGroup = 1
        mfma_i32_16x16x32_i8: kWidth = 8, kGroup = 1
        *mfma_i32_32x32x32_i8: kWidth = 16, kGroup = 1
        mfma_i32_32x32x16_i8: kWidth = 8, kGroup = 1

    Return mfma instruction name and kpack
    """
    kDim = 64 / mfmaNonKDim * kWidth * kGroup
    # Both dtyes are f4 or fp6 or bf6
    if isType4Or6Bit(dtypeA) and isType4Or6Bit(dtypeB):
        assert kWidth == 32 and kGroup == 1, f"Only kWidth=32 and kGroup=1 is supported for {dtypeA} x {dtypeB}"
        kpack = 1
        CBSZ = matrixFormatTable[dtypeB] if trans else matrixFormatTable[dtypeA]
        BLGP = matrixFormatTable[dtypeA] if trans else matrixFormatTable[dtypeB]
        scale_str = 'scale_' if scale else ''
        return f"mfma_{scale_str}f32_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_f8f6f4", kpack, CBSZ, BLGP, scale

    # Both dtypes are fp8 or bf8
    if isType8BitFloat(dtypeA) and isType8BitFloat(dtypeB):
        assert (kWidth == 8 and kGroup == 1) or (
            kWidth == 16), f"Not a valid mfma instruction for {dtypeA} x {dtypeB} with {kWidth=} and {kGroup=}"
        kpack = 2 if (kWidth == 16 and kGroup == 1) else 1
        if kGroup == 2:
            suffix = "f8f6f4"
            CBSZ = matrixFormatTable[dtypeB] if trans else matrixFormatTable[dtypeA]
            BLGP = matrixFormatTable[dtypeA] if trans else matrixFormatTable[dtypeB]
            plot_scale = scale
            scale_str = 'scale_' if scale else ''
        else:
            suffix = f"{dtypeB}_{dtypeA}" if trans else f"{dtypeA}_{dtypeB}"
            CBSZ = -1
            BLGP = -1
            plot_scale = False
            scale_str = ''
        kDim = kDim / 2 if kpack == 2 else kDim
        return f"mfma_{scale_str}f32_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_{suffix}", kpack, CBSZ, BLGP, plot_scale

    # Both types are fp16 or bf16
    if isType16Bit(dtypeA) and isType16Bit(dtypeB):
        assert (kWidth == 8 or kWidth == 4) and kGroup == 1, \
            f"Not a valid mfma instruction for {dtypeA} x {dtypeB} with {kWidth=} and {kGroup=}"
        kpack = 1
        CBSZ = -1
        BLGP = -1
        return f"mfma_f32_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_{dtypeA}", kpack, CBSZ, BLGP, False

    # Both types are i8
    if dtypeA == 'i8' and dtypeB == 'i8':
        assert (kWidth == 16 or kWidth == 8) and kGroup == 1, \
            f"Not a valid mfma instruction for {dtypeA} x {dtypeB} with {kWidth=} and {kGroup=}"
        kpack = 1
        CBSZ = -1
        BLGP = -1
        return f"mfma_i32_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_{dtypeA}", kpack, CBSZ, BLGP, False

    assert False, "Mixed precision between fp8/bf8 and fp6/bf6/f4 not supported in this mode"


def generate_dot_tex(args):
    assert args.plot_type == "dot", \
        f"parsing the wrong arguments. Want dot but have {args.plot_type}"
    # preprocess the args
    dotShape = args.dotShape
    M = dotShape[0]
    N = dotShape[1]
    K = dotShape[2]
    warpsPerCTA = args.warpsPerCTA
    mfmaNonKDim = args.nonKDim
    dtypeA = args.dtypeA
    dtypeB = args.dtypeB
    kWidth = args.kWidth
    kGroup = args.kGroup
    trans = args.mfmaTrans
    scale = args.scale
    # TODO: some of the checking can be done inside this dataclass as well but plot_dot requires quite some refactoring on this
    dotConfig = DotConfig(mfmaNonKDim, kWidth, kGroup, trans, warpsPerCTA)

    # checks and logging
    CTAShape = [
        mfmaNonKDim * warpsPerCTA[0],
        mfmaNonKDim * warpsPerCTA[1],
    ]
    print(f"Plotting dot operation with shapes {(M, N, K)=}, {kWidth=}, {kGroup=}, {warpsPerCTA=}, {CTAShape=}")
    assert M != 0 and CTAShape[0] <= M and M % CTAShape[0] == 0 and \
        N != 0 and CTAShape[1] <= N and N % CTAShape[1] == 0, \
        f"block size ({M}, {N}) should equal to or be multiple of CTA shape ({CTAShape[0]}, {CTAShape[1]})"
    if isMixedPrecBtwF8AndF4OrF6(dtypeA, dtypeB):
        # In the case of mixed precision between 8-bit and 4 or 6-bit,
        # ignore kWidth and kGroup since inA and inB have different kWidth and kGroup values
        if mfmaNonKDim == 16:
            kDim = 128
        elif mfmaNonKDim == 32:
            kDim = 64
        else:
            raise NotImplementedError("scaled dot only supports 32x32x64 or 16x16x128 for now")
        assert K != 0 and K % kDim == 0, \
            f"BLOCK_K = {K} should be spanned by one or multiple of MFMA instructions with KDim = {kDim}"
        kpack = 1
        CBSZ = matrixFormatTable[dtypeB] if trans else matrixFormatTable[dtypeA]
        BLGP = matrixFormatTable[dtypeA] if trans else matrixFormatTable[dtypeB]
        scale_str = 'scale_' if scale else ''
        mfma_inst_str = f"mfma_{scale_str}f32_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_f8f6f4"
        isMixed864 = True
        plot_scale = scale
    else:
        kDim = kWidth * kGroup * 64 // mfmaNonKDim
        assert K % kDim == 0, f"one mfma instruction requires multiple of {kDim} elements along k dim but BLOCK_K = {K}"
        mfma_inst_str, kpack, CBSZ, BLGP, plot_scale = checkMfmaValidity(mfmaNonKDim, kWidth, kGroup, dtypeA, dtypeB,
                                                                         trans, scale)
        isMixed864 = False
    flag = '' if CBSZ == -1 else f" with {CBSZ=},{BLGP=}"
    scale_info = " (scale is not supported hence ignored)" if (scale and not plot_scale) else ''
    print(f"MFMA: {mfma_inst_str} x {kpack}{flag}{scale_info}", end="")
    mfma_inst_str = mfma_inst_str.replace("_", "\\_")
    mfma_inst_str = mfma_inst_str + flag
    if kpack == 2:
        mfma_inst_str = mfma_inst_str + " $\\times$ 2"
    if ((dtypeA == 'fp16' or dtypeA == 'bf16') and kWidth == 8) or (dtypeA == 'i8' and kWidth == 16):
        kDim = 64 / mfmaNonKDim * kWidth / 2
        outType = "i32" if dtypeA == 'i8' else "f32"
        old_instr = f"mfma_{outType}_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_{dtypeA}"
        print(f" or {old_instr} x 2")
        old_instr = old_instr.replace("_", "\\_")
        mfma_inst_str = mfma_inst_str + " or\\\\" + old_instr + "$\\times$2"
    else:
        print("")

    # write the tex file
    curr_dir = Path(__file__).resolve().parent
    with open("myplot.tex", 'w') as f_plot:
        with open(curr_dir / "../utils/preamble.tex") as file:
            preamble = file.read()

        f_plot.write(preamble)
        draw_dotLayout_str = draw_dot_layout_cmd(M, N, K, dtypeA, dtypeB, mfma_inst_str, isMixed864, plot_scale,
                                                 dotConfig)
        func_ref = str(curr_dir / "dotLayout")
        f_plot.write(f"\input{{ {func_ref} }}\n")
        f_plot.write(draw_dotLayout_str)
