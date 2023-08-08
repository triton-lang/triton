add_mlir_translation_library(NVGPUToLLVMIR
  NVGPUToLLVMIR.cpp

  DEPENDS
  NVGPUTableGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  NVGPUIR
  )
