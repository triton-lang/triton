set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(TritonCombine.inc -gen-rewriters)
add_public_tablegen_target(TritonCombineIncGen)

add_mlir_dialect_library(TritonTransforms
  Combine.cpp
  RewriteTensorPointer.cpp

  DEPENDS
  TritonTransformsIncGen
  TritonCombineIncGen
)
