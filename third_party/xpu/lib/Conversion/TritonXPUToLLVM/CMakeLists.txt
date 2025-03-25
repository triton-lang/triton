add_triton_library(TritonXPUToLLVM
  ConvertLayoutOpToLLVM.cpp
  ElementwiseOpToLLVM.cpp
  FuncOpToLLVM.cpp
  GPUOpToLLVMXPU.cpp
  LoadStoreOpToLLVM.cpp
  MakeRangeOpToLLVM.cpp
  ReduceOpToLLVM.cpp
  SPMDOpToLLVM.cpp
  TritonXPUToLLVM.cpp
  VectorizedOpToLLVM.cpp
  ViewOpToLLVM.cpp
  XPUUtilityOpToLLVM.cpp

  TargetInfo.cpp
  TypeConverter.cpp
  Utility.cpp

  DEPENDS
  TTX2LLVMConversionPassIncGen

  LINK_LIBS PUBLIC
  TritonGPUToLLVM
)
