list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ROCM_LIBRARIES
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib/rocm/libhsa-runtime64.so
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files

add_subdirectory(include)
add_subdirectory(lib)


# shows dependecy of targets
set_property(GLOBAL PROPERTY GLOBAL_DEPENDS_DEBUG_MODE 1)

# Python module
if(TRITON_BUILD_PYTHON_MODULE)
    message(STATUS "Adding ROCM Backend Python module")
    set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR})
    set(PYTHON_SRC ${PYTHON_SRC_PATH}/rocm_backend_for_triton.cc)
    include_directories("." ${PYTHON_SRC_PATH})
    include_directories(../include)

    if(PYTHON_INCLUDE_DIRS)
        include_directories(${PYTHON_INCLUDE_DIRS})
    else()
        find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
        include_directories(${Python3_INCLUDE_DIRS})
        link_directories(${Python3_LIBRARY_DIRS})
        link_libraries(${Python3_LIBRARIES})
        add_link_options(${Python3_LINK_OPTIONS})
    endif()

    add_library(rocm_backend_for_triton SHARED ${PYTHON_SRC})

    # get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
    # get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

    set(ROCM_EXTENSION_LIBRARIES
            TritonAnalysisROCM
            TritonTransforms
            TritonHSACO

            # ${dialect_libs}
            TritonGPUROCMIR
            TritonGPUROCMTransforms

            # ${conversion_libs}
            TritonToTritonGPUROCM
            TritonGPUROCMToLLVM

            # tests
            # TritonTestAnalysis
   
            # llvm
            LLVMCore
            LLVMSupport
            LLVMOption
            LLVMCodeGen
            LLVMAsmParser
   
            # MLIR core
            MLIROptLib
            MLIRIR
            MLIRLLVMDialect
            MLIRPass
            MLIRSupport
            MLIRTransforms
            MLIRExecutionEngine
            MLIRMathToLLVM
            MLIRTransformUtils
            MLIRLLVMToLLVMIRTranslation
            # MLIRNVVMToLLVMIRTranslation
            MLIRROCDLToLLVMIRTranslation
            )
    target_link_libraries(rocm_backend_for_triton PRIVATE ${ROCM_EXTENSION_LIBRARIES})
    target_link_libraries(rocm_backend_for_triton PRIVATE ${LLVM_LIBRARIES})
    # TODO: Figure out which target is sufficient to fix errors; rocm_backend_for_triton is
    # apparently not enough
    link_libraries(stdc++fs)

    target_link_options(rocm_backend_for_triton PRIVATE ${LLVM_LDFLAGS})



    file(INSTALL
         ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
         DESTINATION ${PYTHON_THIRD_PARTY_PATH}/hip/)

    file(INSTALL
         ${CMAKE_CURRENT_SOURCE_DIR}/../python/triton/third_party/rocm/lib/bitcode
         DESTINATION ${PYTHON_THIRD_PARTY_PATH}/hip/lib/)
endif()