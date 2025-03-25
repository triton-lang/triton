add_triton_library(MLIRLLVMXPUDialect
  Dialect.cpp

#   ADDITIONAL_HEADER_DIRS
#   ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR
#   ${PROJECT_SOURCE_DIR}/third_party/triton/Dialect/LLVMXPU/IR/

  DEPENDS
  MLIRXPUConversionsIncGen

#   LINK_LIBS PUBLIC
#   MLIRIR
#   MLIRLLVMDialect
#   MLIRSideEffectInterfaces
)
