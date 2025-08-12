add_triton_ut(
  NAME TestOptimizeLDS
  SRCS OptimizeLDSTest.cpp
  LIBS
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonAMDGPUToLLVM
    MLIRUBToLLVM
    TritonAMDUtils
    TritonAMDAnalysis
    TritonAMDGPUTransforms
    TritonAMDGPUDialectToLLVM
)
