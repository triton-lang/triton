add_triton_library(ProtonGPUIR
  Dialect.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
  ProtonGPUTableGen
  ProtonGPUAttrDefsIncGen
  ProtonGPUTypesIncGen
)
