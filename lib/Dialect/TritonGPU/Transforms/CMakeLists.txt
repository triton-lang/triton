set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(TritonGPUCombine.inc -gen-rewriters)
add_public_tablegen_target(TritonGPUCombineIncGen)

add_mlir_dialect_library(TritonGPUTransforms
  Combine.cpp
  Pipeline.cpp
  TritonGPUConversion.cpp

  DEPENDS
  TritonGPUTransformsIncGen
  TritonGPUCombineIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  MLIRTransformUtils
)
