import argparse
import sys

import triton
import triton._C.libtriton.triton as libtriton

if __name__ == '__main__':

    # valid source and target formats
    VALID_FORMATS = ['triton-ir', 'triton-gpu-ir', 'llvm-ir', 'ptx']

    # set up the argument parser
    # TODO: conditional requirements
    parser = argparse.ArgumentParser()
    parser.add_argument('src', help="Source file to compile")
    parser.add_argument('--target', required=True,
                        help="Target format, one of: " + ', '.join(VALID_FORMATS))
    parser.add_argument('--sm', type=int, help="Compute capability to compile for")
    parser.add_argument('--ptx-version', type=int, help="PTX version to compile for")

    # parse the args
    args = parser.parse_args()

    # TODO: clean-up and re-use triton.compiler primitive functions
    # check for validity of format arguments
    if args.target not in VALID_FORMATS:
        print("Invalid target format: " + args.target)
        sys.exit(0)

    # parse source file to MLIR module
    context = libtriton.ir.context()
    module = libtriton.ir.parse_mlir_module(args.src, context)
    module.context = context

    # optimizer triton-ir
    module = triton.compiler.optimize_triton_ir(module)
    if args.target == 'triton-ir':
        print(module.str())
        sys.exit(0)

    if not args.sm:
        raise argparse.ArgumentError(None, "Must specify --sm for PTX compilation")

    # triton-ir -> triton-gpu-ir
    module = triton.compiler.ttir_to_ttgir(module, num_warps=4, num_stages=3, compute_capability=args.sm)
    if args.target == 'triton-gpu-ir':
        print(module.str())
        sys.exit(0)

    # triton-gpu-ir -> llvm-ir
    module = triton.compiler.ttgir_to_llir(module, extern_libs=None, compute_capability=args.sm)
    if args.target == 'llvm-ir':
        print(module)
        sys.exit(0)

    if not args.ptx_version:
        raise argparse.ArgumentError(None, "Must specify --ptx-version for PTX compilation")

    # llvm-ir -> ptx
    module = triton.compiler.llir_to_ptx(module, compute_capability=args.sm, ptx_version=args.ptx_version)
    assert args.target == 'ptx'
    print(module)
