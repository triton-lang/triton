add_triton_library(ProtonIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  ProtonTableGen
  ProtonAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRFunctionInterfaces
  MLIRTransformUtils
)
