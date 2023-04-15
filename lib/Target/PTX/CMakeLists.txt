add_mlir_translation_library(TritonPTX
        PTXTranslation.cpp

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        TritonLLVMIR
        )
