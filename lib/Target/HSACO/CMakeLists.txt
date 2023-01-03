add_mlir_translation_library(TritonHSACO
        HSACOTranslation.cpp

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        TritonLLVMIR
        )
