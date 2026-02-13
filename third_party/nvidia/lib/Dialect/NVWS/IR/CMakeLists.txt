add_triton_library(NVWSIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  NVWSTableGen
  NVWSAttrDefsIncGen

)
