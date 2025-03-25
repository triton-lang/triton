add_triton_library(TritonToTritonXPU
  TritonXPUConversion.cpp
  TritonToTritonXPUPass.cpp

  DEPENDS
  TT2TTXConversionPassIncGen
)
