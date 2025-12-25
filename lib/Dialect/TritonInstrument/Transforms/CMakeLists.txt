add_triton_library(TritonInstrumentTransforms
  ConcurrencySanitizer.cpp
  IllegalInstructionSanitizer.cpp

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
