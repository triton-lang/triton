add_triton_library(TritonXPUIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  TritonXPUTableGen
  TritonXPUAttrDefsIncGen
)
