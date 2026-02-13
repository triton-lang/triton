add_triton_library(TritonLLVMIR
        LLVMDIScope.cpp
        LLVMDILocalVariable.cpp
        LLVMIRBreakPhiStruct.cpp
        LLVMDIUtils.cpp

        DEPENDS
        LLVMIRIncGen

        )

set_source_files_properties(
        LLVMIRTranslation.cpp
        PROPERTIES
        COMPILE_FLAGS "-D__BUILD_DIR__=\\\"${CMAKE_BINARY_DIR}\\\"")
