add_triton_library(NVGPUIR
  Dialect.cpp

  DEPENDS
  NVGPUTableGen
  NVGPUAttrDefsIncGen

)
