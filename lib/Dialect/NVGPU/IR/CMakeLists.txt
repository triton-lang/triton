add_mlir_dialect_library(NVGPUIR
  Dialect.cpp

  DEPENDS
  NVGPUTableGen
  NVGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
)
