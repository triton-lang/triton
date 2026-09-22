add_triton_library(TritonInstrumentTransforms
  PrepareConSanCaptures.cpp
  GlobalSanitizer.cpp
  ConcurrencySanitizer.cpp
  FpSanitizer.cpp

  DEPENDS
  TritonInstrumentTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  TritonToTritonGPU
  TritonInstrumentIR
  MLIRTransformUtils
)
