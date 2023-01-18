add_mlir_translation_library(TritonLLVMIR
        LLVMIRTranslation.cpp

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRLLVMIR
        MLIRSCFToStandard
        MLIRSupport
        MLIRTargetLLVMIRExport
        )
