add_triton_library(MLIRLLVMXPUToLLVMIRTranslation
  LLVMXPUToLLVMIRTranslation.cpp

#   DEPENDS
#   MLIRXPUConversionsIncGen

#   LINK_COMPONENTS
#   Core

#   LINK_LIBS PUBLIC
#   MLIRLLVMXPUDialect
#   MLIRIR
#   MLIRLLVMDialect
#   MLIRSupport
#   MLIRTargetLLVMIRExport
)
