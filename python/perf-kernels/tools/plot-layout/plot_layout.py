import argparse
import sys
import os
import subprocess
from dataclasses import dataclass


def draw_dot_layout_cmd(M, N, K, dtype_a, dtype_b, mfma_inst_str, isMixed864, plot_scale, dotConfig):
    mfmaNonKDim = dotConfig.mfmaNonKDim
    warpsPerCTA = dotConfig.warpsPerCTA
    trans = dotConfig.trans
    kWidth = dotConfig.kWidth
    kGroup = dotConfig.kGroup
    scaleLabel = 0.7 if (kWidth == 4 or (kWidth == 8 and mfmaNonKDim == 32)) else 1

    outType = 'i32' if dtype_a == 'i8' else 'f32'
    kWidth_a = kWidth_b = kWidth
    kGroup_a = kGroup_b = kGroup
    if isMixed864:
        if isType8BitFloat(dtype_a):
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

    return f'''\\begin{{document}}
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
    \\drawMFMAInstr{{{mfmaNonKDim}}}{{\\mfmaTrans}}{{{dtype_a}}}{{{dtype_b}}}{{{outType}}}{{{scaling}}}

  \\end{{tikzpicture}}
\\end{{document}}'''


def draw_blocked_layout_cmd(dim0, dim1, dim0Name, dim1Name, blockedConfig):
    return f'''\\begin{{document}}
  \\begin{{tikzpicture}}
    \\def\\scale{{1}}
    \\def\\elem{{0.06}}
    \\coordinate (TL) at (0,0);
    \\def\\dimColName{{{dim0Name}}}
    \\def\\dimRowName{{{dim1Name}}}
    \\drawBlockedTensor{{{dim0}}}{{{dim1}}}{{{blockedConfig.sizePerThread[0]}}}{{{blockedConfig.sizePerThread[1]}}}{{{blockedConfig.threadsPerWarp[0]}}}{{{blockedConfig.warpsPerCTA[0]}}}{{{blockedConfig.warpsPerCTA[1]}}}{{{blockedConfig.order[0]}}}
  \\end{{tikzpicture}}
\\end{{document}}'''


def typeToBytes(dtype):
    if dtype == 'bf16' or dtype == 'fp16':
        return 2
    if dtype == 'bf8' or dtype == 'fp8' or dtype == 'i8':
        return 1
    if dtype == 'f4':
        return 0.5
    if dtype == 'fp6' or dtype == 'bf6':
        return 0.75


def maxKDimInBytes(dtype, mfmaNonKDim, kWidth):
    groups = 64 / mfmaNonKDim
    if (dtype == 'bf8' or dtype == 'fp8') and kWidth == 16:
        groups *= 2
    return groups * kWidth * typeToBytes(dtype)


def calcPerPhase(banks, dtype, K):
    bytesPerBank = 4
    return max(banks * bytesPerBank / (K * typeToBytes(dtype)), 1)


def draw_lds_access_cmd(dim0, dim1, dtype, mfmaNonKDim, ldsConfig):
    if ldsConfig.ldsLayout == 'swizzle':
        hasSwizzle = 1
    elif ldsConfig.ldsLayout == 'padding':
        hasSwizzle = 2
    else:
        hasSwizzle = 0

    if ldsConfig.ldsAccess == 'read':
        accessMode = 1
    elif ldsConfig.ldsAccess == 'write':
        accessMode = 2
    else:
        accessMode = 0

    trans = 1 if ldsConfig.mnContig else 0
    useMfmaTransLD = 1 if ldsConfig.mfmaTransLD else 0
    banks = ldsConfig.banks
    padInterval = ldsConfig.padInterval
    padAmount = ldsConfig.padAmount

    if trans:
        dim0Name = 'k'
        dim1Name = 'n'
    else:
        dim0Name = 'm'
        dim1Name = 'k'
    dim0Size = dim0
    dim1Size = dim1
    '''
    Definitions of different vector size

    swizzleVec: Number of elements that are grouped together when swizzling is enabled.
                Note that this is all about LDS layout without considering LDS read
                or write patterns. And this is un-related to K- or MN-contig settings.
    accessVec:  When reading from or writing to LDS, accessVec is the number of contiguous
                elements each thread read or write as a vector.
                This is un-related to K- or MN-contig settings.
                Note that accessVec <= swizzleVec. accessVec for read and write are not
                required to be the same.
    kWidth:     Number of contiguous elements along the k dim that each thread holds
                right before invoking mfma instruction(s). kWidth can be larger than
                the required number of contiguous elements along the k dim for a single
                mfma instruction.
                Note that kWidth is un-related to swizzleVec or accessVec. kWidth should
                be set according to datatype and mfmaNonKDim.

    We need to handle the following cases of LDS layout and access patterns:

    case 1: K-contig in both HBM and LDS (default)
      In most cases, we can set swizzleVec = accessVec = kWidth according to the dtype.
      However, for mfmaNonKDim = 16, banks = 64, and kWidth = 8B, 32 threads will
      access LDS at the same cycle. In this case, we need to double swizzleVec = 16B.

      Swizzling: works as suggested above.
      Padding:   will have bank conflicts for ds_read_b128 due to non-linear thread ids
                 are accessing LDS at the same cycle

    case 2: MN-contig in both HBM and LDS without using mfma_transpose_ld instructions (-mnContig)
      In this case, ds_read can only read one element at a time (i.e. accessVec is always 1).
      Therefore, we can always choose swizzleVec = 16B. kWidth does not matter. accessVec is always 1.
      Note that in this case, only swizzling is supported and can help resolve bank conflicts.
      But the performance bottleneck is scalar ds_read rather than bank conflicts.

    case 3: MN-contig in both HBM and LDS using mfma_transpose_ld instructions (-mnContig -mfma_trans_load)
      In this case, ds_read is done in a special pattern so that the ds_read_b64_tr_bx instructions
      can be used. Each thread will read 8B data, which corresponds to kWidth = 8B/elemInBytes.
      The swizzleVec needs to be set to mfmaNonKDim.

      Swizzling: currently, it leads to bank conflicts for nonKDim = 16 and
                 if the read pattern follows the spec.
                 For nonKDim = 32, swizzling does not have bank conflicts.
      Padding:   It can help resolve bank conflicts for both nonKDim = 16 and 32.
                 However, it leads to a lot of waste of LDS space.

    case 4: MN-contig in HBM and k-Contig in LDS (-inThreadTrans)
      Not supported yet
    '''

    elemTypeInBytes = typeToBytes(dtype)

    bankLabelScale = 0.8
    bsize = 0.15

    if trans == 0:
        # case 1
        swizzleVec = ldsConfig.swizzleVec
        accessVec = ldsConfig.accessVec
        vec = ldsConfig.kWidth
    elif useMfmaTransLD == 0:
        # case 2
        swizzleVec = 16 / elemTypeInBytes
        accessVec = 1
        vec = swizzleVec
    else:
        # case 3
        vec = 8 / elemTypeInBytes
        swizzleVec = mfmaNonKDim
        accessVec = ldsConfig.accessVec

    kWidth = ldsConfig.kWidth
    vecInBytes = vec * elemTypeInBytes

    return f'''\\begin{{document}}
  \\begin{{tikzpicture}}
    \\def\\scale{{1}}
    \\def\\M{{{dim0}}}
    \\def\\K{{{dim1}}}
    \\def\\mfmaKWidth{{{kWidth}}}
    \\def\\vec{{{vec}}}
    \\def\\swizzleVec{{{swizzleVec}}}
    \\def\\accessVec{{{accessVec}}}
    \\def\\vecInBytes{{{vecInBytes}}}
    \\def\\bytesPerElem{{{elemTypeInBytes}}}
    \\def\\hasSwizzle{{{hasSwizzle}}}
    \\def\\accessMode{{{accessMode}}}
    \\def\\mfmaNonKDim{{{mfmaNonKDim}}}
    \\def\\dtype{{{dtype}}}
    \\def\\trans{{{trans}}}
    \\def\\useMfmaTransLD{{{useMfmaTransLD}}}
    \\def\\padInterval{{{padInterval}}}
    \\def\\padAmount{{{padAmount}}}

    \\def\\elemH{{0.18}}
    \\def\\elem{{0.18}}
    \\def\\bsize{{{bsize}}}
    \\def\\bankLabelScale{{{bankLabelScale}}}
    \\coordinate (tile TL) at (0,0);
    \\coordinate (TL) at (tile TL);
    \\drawTensorLayoutGlobalMem{{{dim0Name}}}{{{dim1Name}}}{{{dim0Size}}}{{{dim1Size}}}
    \\coordinate (TL) at ($(TL)+(0, -\drawRow-8*\\elemH)$);
    \\drawLDSLayoutAndAccess{{\\hasSwizzle}}{{\\accessMode}}{{{banks}}}{{{dim0Name}}}{{{dim1Name}}}{{{dim1Size}}}
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


matrixFormatTable = {'fp8': 0, 'bf8': 1, 'fp6': 2, 'bf6': 3, 'f4': 4}


def matrixFormat(dtype_a, dtype_b):
    '''
    return CBSZ and BLGP according to data types
    b000: E4M3(FP8)
    b001: E5M2(BF8)
    b010: E2M3(FP6)
    b011: E3M2(BF6)
    b100: E2M1(FP4)
    '''
    return matrixFormatTable[dtype_a], matrixFormatTable[dtype_b]


def isType4Or6Bit(dtype):
    return dtype == 'fp6' or dtype == 'bf6' or dtype == 'f4'


def isType8BitFloat(dtype):
    return dtype == 'fp8' or dtype == 'bf8'


def isType16Bit(dtype):
    return dtype == 'bf16' or dtype == 'fp16'


def isMixedPrecType(dtype):
    return isType8BitFloat(dtype) or isType4Or6Bit(dtype)


def isMixedPrecBtwF8AndF4OrF6(dtype_a, dtype_b):
    return (isType8BitFloat(dtype_a) and isType4Or6Bit(dtype_b)) or (isType8BitFloat(dtype_b)
                                                                     and isType4Or6Bit(dtype_a))


def checkMfmaValidity(mfmaNonKDim, kWidth, kGroup, dtype_a, dtype_b, trans, scale):
    ## Check input types
    ## Mixed precision is only allowed within f8, f6 and f4
    assert (isMixedPrecType(dtype_a) and isMixedPrecType(dtype_b)) or (
        dtype_a == dtype_b), f"Cannot do mixed precision mfma with {dtype_a} and {dtype_b}"
    '''
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
    '''
    kDim = 64 / mfmaNonKDim * kWidth * kGroup
    ## Both dtyes are f4 or fp6 or bf6
    if isType4Or6Bit(dtype_a) and isType4Or6Bit(dtype_b):
        assert kWidth == 32 and kGroup == 1, f"Only kWidth=32 and kGroup=1 is supported for {dtype_a} x {dtype_b}"
        kpack = 1
        CBSZ = matrixFormatTable[dtype_b] if trans else matrixFormatTable[dtype_a]
        BLGP = matrixFormatTable[dtype_a] if trans else matrixFormatTable[dtype_b]
        scale_str = 'scale_' if scale else ''
        return f"mfma_{scale_str}f32_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_f8f6f4", kpack, CBSZ, BLGP, scale

    ## Both dtypes are fp8 or bf8
    if isType8BitFloat(dtype_a) and isType8BitFloat(dtype_b):
        assert (kWidth == 8 and kGroup == 1) or (
            kWidth == 16), f"Not a valid mfma instruction for {dtype_a} x {dtype_b} with {kWidth=} and {kGroup=}"
        kpack = 2 if (kWidth == 16 and kGroup == 1) else 1
        if kGroup == 2:
            suffix = "f8f6f4"
            CBSZ = matrixFormatTable[dtype_b] if trans else matrixFormatTable[dtype_a]
            BLGP = matrixFormatTable[dtype_a] if trans else matrixFormatTable[dtype_b]
            plot_scale = scale
            scale_str = 'scale_' if scale else ''
        else:
            suffix = f"{dtype_b}_{dtype_a}" if trans else f"{dtype_a}_{dtype_b}"
            CBSZ = -1
            BLGP = -1
            plot_scale = False
            scale_str = ''
        kDim = kDim / 2 if kpack == 2 else kDim
        return f"mfma_{scale_str}f32_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_{suffix}", kpack, CBSZ, BLGP, plot_scale

    ## Both types are fp16 or bf16
    if isType16Bit(dtype_a) and isType16Bit(dtype_b):
        assert (
            kWidth == 8 or kWidth == 4
        ) and kGroup == 1, f"Not a valid mfma instruction for {dtype_a} x {dtype_b} with {kWidth=} and {kGroup=}"
        kpack = 1
        CBSZ = -1
        BLGP = -1
        return f"mfma_f32_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_{dtype_a}", kpack, CBSZ, BLGP, False

    ## Both types are i8
    if dtype_a == 'i8' and dtype_b == 'i8':
        assert (
            kWidth == 16 or kWidth == 8
        ) and kGroup == 1, f"Not a valid mfma instruction for {dtype_a} x {dtype_b} with {kWidth=} and {kGroup=}"
        kpack = 1
        CBSZ = -1
        BLGP = -1
        return f"mfma_i32_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_{dtype_a}", kpack, CBSZ, BLGP, False

    assert False, "Mixed precision between fp8/bf8 and fp6/bf6/f4 not supported in this mode"


def run_bash_command(commandstring):
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE)
    return proc.stdout.splitlines()


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Draw triton layouts",
        allow_abbrev=False,
    )
    ## tensor shapes
    parser.add_argument("-tensorShape", type=int, nargs=2, default=(128, 64),
                        help='2D tensor shape in the form of dim0,dim1')
    parser.add_argument("-dotShape", type=int, nargs=3, default=(32, 128, 64), help='Dot op shape in the form of M,N,K')
    parser.add_argument("-plot", type=str, default="blocked", choices=['blocked', 'dot', 'wmma', 'lds'],
                        help='choose plot mode')
    parser.add_argument("-dim0", type=str, default="M", help='tensor dim0 name')
    parser.add_argument("-dim1", type=str, default="K", help='tensor dim1 name')
    ## blocked layout parameters
    parser.add_argument("-sizePerThread", type=int, nargs=2, default=(1, 4))
    parser.add_argument("-threadsPerWarp", type=int, nargs=2, default=(16, 4))
    parser.add_argument("-warpsPerCTA", type=int, nargs=2, default=(1, 4))
    parser.add_argument("-order", type=int, nargs=2, default=(1, 0))
    ## dot layout parameters
    parser.add_argument("-nonKDim", type=int, default=16, choices=[16, 32], help='mfma instruction dim')
    parser.add_argument("-kWidth", type=int, default=4, choices=[4, 8, 16, 32],
                        help='number of contiguous elements per thread')
    parser.add_argument("-kGroup", type=int, default=1, choices=[1, 2],
                        help='total number of elements / kWidth per mfma instruction')
    parser.add_argument("-dtype_a", type=str, default='fp16',
                        choices=['fp16', 'bf16', 'fp8', 'bf8', 'fp6', 'bf6', 'f4',
                                 'i8'], help='element type of operand A')
    parser.add_argument("-dtype_b", type=str, default='fp16',
                        choices=['fp16', 'bf16', 'fp8', 'bf8', 'fp6', 'bf6', 'f4',
                                 'i8'], help='element type of operand B')
    parser.add_argument("-mfmaTrans", action='store_true', default=False, help='If set, then use mfma.trans layout')
    parser.add_argument("-scale", action='store_true', default=False,
                        help='If set, plot the scale tensor for mfma_f8f6f4 instructions')
    ## LDS access parameters
    parser.add_argument("-banks", type=int, default=32, choices=[32, 64], help='choose the number of banks in LDS')
    parser.add_argument("-lds_layout", type=str, default="none", choices=['swizzle', 'padding', 'none'],
                        help='choose the LDS data layout')
    parser.add_argument("-lds_access", type=str, default="none", choices=['read', 'write', 'none'],
                        help='choose LDS access mode')
    parser.add_argument("-mnContig", action='store_true', default=False,
                        help='If set, the tensor is K x N and n-contig')
    parser.add_argument("-mfma_trans_load", action='store_true', default=False,
                        help='If set, use MFMA transpose load instructions')
    parser.add_argument("-swizzleVec", type=int, default=4, choices=[4, 8, 16, 32],
                        help='number of contiguous elements in a vector to swizzle')
    parser.add_argument("-padInterval", type=int, default=1, help='Add padding for every padInterval bytes')
    parser.add_argument("-padAmount", type=int, default=0, help='Pad padAmount bytes for every padInterval bytes')
    ## wmma instruction layout parameter
    parser.add_argument("-wave_size", type=int, default=32, choices=[32, 64], help='choose the wmma instruction mode')
    parser.add_argument("-o", type=str, default="myplot", help='output pdf file name (without surfix)')
    parser.add_argument("-keep", action='store_true', default=False, help='If set, keep the generated .tex file')

    args = parser.parse_args()

    return args


@dataclass
class BlockedConfig:
    sizePerThread: tuple
    threadsPerWarp: tuple
    warpsPerCTA: tuple
    order: tuple


@dataclass
class DotConfig:
    mfmaNonKDim: int
    kWidth: int
    kGroup: int
    trans: int
    warpsPerCTA: tuple


@dataclass
class LDSConfig:
    banks: int
    ldsLayout: str
    ldsAccess: str
    mnContig: bool
    mfmaTransLD: bool
    swizzleVec: int
    accessVec: int
    kWidth: int
    padInterval: int
    padAmount: int

    def __init__(self, banks, ldsLayout, ldsAccess, mnContig, mfmaTransLD, swizzleVec, accessVec, kWidth, padInterval,
                 padAmount):
        self.banks = banks
        self.ldsLayout = ldsLayout
        self.ldsAccess = ldsAccess
        self.mnContig = mnContig
        self.mfmaTransLD = mfmaTransLD
        self.swizzleVec = swizzleVec
        self.accessVec = accessVec
        self.kWidth = kWidth
        self.padInterval = padInterval
        self.padAmount = padAmount
        if self.swizzleVec < self.kWidth:
            self.swizzleVec = self.kWidth

    def print(self):
        print(
            f"{self.banks=} {self.ldsLayout=} {self.ldsAccess=} {self.mnContig=} {self.mfmaTransLD=} {self.swizzleVec=} {self.accessVec=} {self.kWidth=} {self.padInterval} {self.padAmount}"
        )


def main():
    args = parse_args()

    dotShape = args.dotShape
    M = dotShape[0]
    N = dotShape[1]
    K = dotShape[2]
    tShape = args.tensorShape
    dim0 = tShape[0]
    dim1 = tShape[1]
    dim0Name = args.dim0
    dim1Name = args.dim1
    plot_mode = args.plot
    mfmaNonKDim = args.nonKDim
    kWidth = args.kWidth
    kGroup = args.kGroup
    dtype_a = args.dtype_a
    dtype_b = args.dtype_b
    trans = 1 if args.mfmaTrans else 0
    scale = 1 if args.scale else 0
    ofilename = args.o
    keepSrc = args.keep

    ldsLayout = args.lds_layout
    ldsAccess = args.lds_access
    banks = args.banks
    mnContig = args.mnContig
    mfmaTransLD = args.mfma_trans_load
    swizzleVec = args.swizzleVec
    padInterval = args.padInterval
    padAmount = args.padAmount

    waveSize = args.wave_size

    sizePerThread = args.sizePerThread
    threadsPerWarp = args.threadsPerWarp
    warpsPerCTA = args.warpsPerCTA
    order = args.order

    blockedConfig = BlockedConfig(sizePerThread, threadsPerWarp, warpsPerCTA, order)
    dotConfig = DotConfig(mfmaNonKDim, kWidth, kGroup, trans, warpsPerCTA)
    ldsConfig = LDSConfig(banks, ldsLayout, ldsAccess, mnContig, mfmaTransLD, swizzleVec, kWidth, kWidth, padInterval,
                          padAmount)

    CTAShape = []
    if plot_mode == 'blocked':
        print(f"Plotting tensor {dim0Name}={dim0},{dim1Name}={dim1} with blocked layout:")
        print(f"{sizePerThread=}", end=" ")
        print(f"{threadsPerWarp=}", end=" ")
        print(f"{warpsPerCTA=}", end=" ")
        print(f"{order=}", end=" ")
        CTAShape.append(sizePerThread[0] * threadsPerWarp[0] * warpsPerCTA[0])
        CTAShape.append(sizePerThread[1] * threadsPerWarp[1] * warpsPerCTA[1])
        print(f"CTAShape={CTAShape}")
        assert dim0 != 0 and CTAShape[0] <= dim0 and dim0 % CTAShape[0] == 0, "bad tensor dimension " + dim0Name
        assert dim1 != 0 and CTAShape[1] <= dim1 and dim1 % CTAShape[1] == 0, "bad tensor dimension " + dim1Name

    if plot_mode == 'dot':
        CTAShape.append(mfmaNonKDim * warpsPerCTA[0])
        CTAShape.append(mfmaNonKDim * warpsPerCTA[1])
        print(f"Plotting dot operation with shapes=M{M}-N{N}-K{K},{kWidth=},{kGroup=},{warpsPerCTA=},{CTAShape=}")
        assert M != 0 and CTAShape[0] <= M and M % CTAShape[0] == 0, "bad tensor dimension M"
        assert N != 0 and CTAShape[1] <= N and N % CTAShape[1] == 0, "bad tensor dimension N"
        if isMixedPrecBtwF8AndF4OrF6(dtype_a, dtype_b):
            ## In the case of mixed precision between 8-bit and 4 or 6-bit,
            ## ignore kWidth and kGroup since inA and inB have different kWidth and kGroup values
            kDim = 128
            assert K != 0 and K % kDim == 0, f"one mfma instruction requires {kDim:.0f} elements along k dim but BLOCK_K = {K}"
            kpack = 1
            CBSZ = matrixFormatTable[dtype_b] if trans else matrixFormatTable[dtype_a]
            BLGP = matrixFormatTable[dtype_a] if trans else matrixFormatTable[dtype_b]
            scale_str = 'scale_' if scale else ''
            mfma_inst_str = f"mfma_{scale_str}f32_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_f8f6f4"
            isMixed864 = True
            plot_scale = scale
        else:
            kDim = kWidth * kGroup * 64 / mfmaNonKDim
            assert K != 0 and K % kDim == 0, f"one mfma instruction requires {kDim:.0f} elements along k dim but BLOCK_K = {K}"
            mfma_inst_str, kpack, CBSZ, BLGP, plot_scale = checkMfmaValidity(mfmaNonKDim, kWidth, kGroup, dtype_a,
                                                                             dtype_b, trans, scale)
            isMixed864 = False
        flag = '' if CBSZ == -1 else f" with {CBSZ=},{BLGP=}"
        scale_info = " (scale is not supported hence ignored)" if (scale and not plot_scale) else ''
        print(f"MFMA: {mfma_inst_str} x {kpack}{flag}{scale_info}", end="")
        mfma_inst_str = mfma_inst_str.replace("_", "\\_")
        mfma_inst_str = mfma_inst_str + flag
        if kpack == 2:
            mfma_inst_str = mfma_inst_str + " $\\times$ 2"
        if ((dtype_a == 'fp16' or dtype_a == 'bf16') and kWidth == 8) or (dtype_a == 'i8' and kWidth == 16):
            kDim = 64 / mfmaNonKDim * kWidth / 2
            outType = "i32" if dtype_a == 'i8' else "f32"
            old_instr = f"mfma_{outType}_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_{dtype_a}"
            print(f" or {old_instr} x 2")
            old_instr = old_instr.replace("_", "\\_")
            mfma_inst_str = mfma_inst_str + " or\\\\" + old_instr + "$\\times$2"
        else:
            print("")

    if plot_mode == 'lds':
        print(f"Plotting LDS access for tensor {dim0}x{dim1} with vec={kWidth}")

    with open("myplot.tex", 'w') as f_plot:
        with open("preamble.tex") as file:
            preamble = file.read()

        f_plot.write(preamble)
        if plot_mode == 'blocked':
            draw_blockedLayout_str = draw_blocked_layout_cmd(dim0, dim1, dim0Name, dim1Name, blockedConfig)
            f_plot.write("\input{blockedLayout}\n")
            f_plot.write(draw_blockedLayout_str)
        elif plot_mode == 'dot':
            draw_dotLayout_str = draw_dot_layout_cmd(M, N, K, dtype_a, dtype_b, mfma_inst_str, isMixed864, plot_scale,
                                                     dotConfig)
            f_plot.write("\input{dotLayout}\n")
            f_plot.write(draw_dotLayout_str)
        elif plot_mode == 'lds':
            draw_lds_str = draw_lds_access_cmd(dim0, dim1, dtype_a, mfmaNonKDim, ldsConfig)
            f_plot.write("\input{ldsLayout}\n")
            f_plot.write(draw_lds_str)
        elif plot_mode == 'wmma':
            draw_wmma_str = draw_wmma_instr_cmd(waveSize)
            f_plot.write("\input{wmmaLayout}\n")
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
