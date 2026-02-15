add_triton_library(ProtonIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  ProtonTableGen
  ProtonAttrDefsIncGen
)
