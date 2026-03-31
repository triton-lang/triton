add_triton_library(TritonInstrumentTransforms
  GlobalSanitizer.cpp
  ConcurrencySanitizer.cpp
  FpSanitizer.cpp

  DEPENDS
  TritonInstrumentTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  TritonToTritonGPU
  TritonInstrumentIR
  MLIRTransformUtils
)
