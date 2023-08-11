add_mlir_translation_library(TritonPTX
        PTXTranslation.cpp

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        TritonLLVMIR
        )

set_source_files_properties(
        PTXTranslation.cpp
        PROPERTIES
        COMPILE_FLAGS "-D__BUILD_DIR__=\\\"${CMAKE_BINARY_DIR}\\\"")
