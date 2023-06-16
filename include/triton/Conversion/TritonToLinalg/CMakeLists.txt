#===------------------------------------------------------------------------===#
#
# Copyright (c) Triton Project Contributors.
#
#===------------------------------------------------------------------------===#

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonToLinalg)
add_public_tablegen_target(TritonToLinalgConversionPassIncGen)
