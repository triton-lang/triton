add_triton_library(NVGIR
  Dialect.cpp

  DEPENDS
  NVGTableGen
  NVGAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
)
