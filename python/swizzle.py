# raw parameters
import argparse


def linearize_index(r, c):
    return r*strideRow + c*strideCol

def swizzle(r, c, swizzling=False, verbose=False):
    rowOff = r * strideRow
    phase = int((r / perPhase) % maxPhase)
    colOffSwizzled = ((c // outVec) ^ phase) * outVec if swizzling \
        else 0
    colOffOrdered = int(c % outVec) // minVec * minVec
    colOff = (colOffSwizzled + colOffOrdered) * strideCol
    offset = rowOff + colOff
    if verbose:
        print(f"rowOff = r * strideRow = {r} * {strideRow} = {rowOff}")
        if swizzling:
            print(f"phase = (r / perPhase) % maxPhase = ({r} / {perPhase}) % {maxPhase} = {phase}")
            print(f"colOffSwizzled = ((c // outVec) ^ phase) * outVec * strideCol = (({c} // {outVec}) ^ {phase}) * {outVec} * {strideCol} = {colOffSwizzled*strideCol}")
        print(f"colOffOrdered = (c % outVec) // minVec * minVec * strideCol = ({c} % {outVec}) // {minVec} * {minVec} * {strideCol} = {colOffOrdered*strideCol}")
    return offset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--swizzle", action="store_true",
                        help="enable swizzling for shared pointer computation")
    parser.add_argument("--swap", action="store_true", help="swap r and c")
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_single = subparsers.add_parser('single', help='enter single pair of (r,c)')
    parser_single.add_argument("r", type=int, help="row index of input")
    parser_single.add_argument("c", type=int, help="col index of input")
    parser_single.add_argument("-v", "--verbose", action="store_true",
                        help="print pointer calculation")

    parser_group = subparsers.add_parser('group', help='compute a matrix of indices')
    parser_group.add_argument("nRows", type=int, help="number of rows from blocked")
    parser_group.add_argument("nCols", type=int, help="number of cols from blocked")

    args = parser.parse_args()
    return args

# shared memory info
nBanks = 32
bankByteWidth = 32
dtypeByteWidth = 16
bankWidth = nBanks * bankByteWidth // dtypeByteWidth

# matrix A info
strideRow = 64
strideCol = 1
inVec = 8
outVec = 4
perPhase = 1
maxPhase = 16

# # matrix B info
# strideRow = 32
# strideCol = 1
# inVec = 2
# outVec = 2
# perPhase = 1
# maxPhase = 4

# derived parameters
minVec = min(inVec, outVec)


if __name__ == "__main__":
    args = parse_args()
    if 'c' in args:
        r, c = args.r, args.c
        if args.swap:
            r, c = c, r
        print(f"r = {r}, c = {c}")
        print("linearized input:", linearize_index(r, c))
        print("swizzled  output:", swizzle(r, c, swizzling=args.swizzle, verbose=args.verbose))
    else:
        nRows, nCols = args.nRows, args.nCols
        shared = []
        print("Blocked:")
        for r in range(nRows):
            for c in range(nCols):
                idx = r*nCols + c
                # print(r, c, idx)
                print(f"{idx:>3d} ", end="")
                if args.swap:
                    rIn, cIn = c, r
                else:
                    rIn, cIn = r, c
                shared.append(swizzle(rIn, cIn, swizzling=args.swizzle))
            print()
        print("Shared:")
        for s in shared:
            print(f"{s:>3d} ", end="")
            if (s+1) % bankWidth == 0 and s > 1:
                print()
        print()
