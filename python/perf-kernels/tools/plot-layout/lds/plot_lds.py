from dataclasses import dataclass
from pathlib import Path


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
    """
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
    """

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

    return f"""\\begin{{document}}
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
               \\end{{document}}"""


def generate_lds_tex(args):
    assert args.plot_type == "lds", \
        f"parsing the wrong arguments. Want lds but have {args.plot_type}"
    # preprocess the args
    tShape = args.tensorShape
    dim0 = tShape[0]
    dim1 = tShape[1]
    accessVec = kWidth = args.kWidth
    dtype = args.dtype
    mfmaNonKDim = args.nonKDim
    ldsLayout = args.layout
    ldsAccess = args.access
    banks = args.banks
    mnContig = args.mnContig
    mfmaTransLD = args.mfma_trans_load
    swizzleVec = args.swizzleVec
    padInterval = args.padInterval
    padAmount = args.padAmount

    ldsConfig = LDSConfig(banks, ldsLayout, ldsAccess, mnContig, mfmaTransLD, swizzleVec, accessVec, kWidth,
                          padInterval, padAmount)

    # checks and logging
    print(f"Plotting LDS access for tensor {dim0}x{dim1} with vec={kWidth}")
    # write the tex file
    curr_dir = Path(__file__).resolve().parent
    with open("myplot.tex", 'w') as f_plot:
        with open(curr_dir / "../utils/preamble.tex") as file:
            preamble = file.read()

        f_plot.write(preamble)
        draw_lds_str = draw_lds_access_cmd(dim0, dim1, dtype, mfmaNonKDim, ldsConfig)
        func_ref = str(curr_dir / "ldsLayout")
        f_plot.write(f"\input{{ {func_ref} }}\n")
        f_plot.write(draw_lds_str)
