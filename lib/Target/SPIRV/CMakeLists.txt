# SPIRV Tools is required to disassemble the spirv module
find_package(SPIRVTools)

add_mlir_translation_library(TritonSPIRV
        SPIRVTranslation.cpp

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRLLVMIRTransforms
        MLIRSupport
        MLIRTargetLLVMIRExport
        # spirv tools
        SPIRV-Tools
        )
