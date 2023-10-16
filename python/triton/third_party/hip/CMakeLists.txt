list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# FLAGS
message(STATUS "HIP_BACKEND_MODE = ${HIP_BACKEND_MODE}")
set(ROCM_DEFAULT_DIR "/opt/rocm")
add_definitions( -DROCM_DEFAULT_DIR="${ROCM_DEFAULT_DIR}")
set(ROCM_LIBRARIES
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/hsa/libhsa-runtime64.so
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files

add_subdirectory(include)
add_subdirectory(lib)


# shows dependecy of targets
set_property(GLOBAL PROPERTY GLOBAL_DEPENDS_DEBUG_MODE 1)

# Python module
if(TRITON_BUILD_PYTHON_MODULE)
    message(STATUS "Adding HIP Backend Python module")
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
            MLIRROCDLToLLVMIRTranslation
            # MLIRNVVMToLLVMIRTranslation
            )
    target_link_libraries(rocm_backend_for_triton PRIVATE ${ROCM_EXTENSION_LIBRARIES})
    target_link_libraries(rocm_backend_for_triton PRIVATE ${LLVM_LIBRARIES})
    link_libraries(stdc++fs)
    target_link_options(rocm_backend_for_triton PRIVATE ${LLVM_LDFLAGS})

    # copy to upstream third_party dir
    file(REMOVE_RECURSE ${PYTHON_THIRD_PARTY_PATH}/hip)
    file(INSTALL
         ${CMAKE_CURRENT_SOURCE_DIR}/
         DESTINATION ${PYTHON_THIRD_PARTY_PATH}/hip)

    # set HIP_BACKEND_MODE to true
    set(HIP_BACKEND_PY "${PYTHON_THIRD_PARTY_PATH}/hip/hip_backend.py")
    set(HIP_BACKEND_PY_STAMP "${PYTHON_THIRD_PARTY_PATH}/hip/hip_backend.py.stamp")

    add_custom_command(
        OUTPUT ${HIP_BACKEND_PY_STAMP}
        COMMAND 
            sed -i'' -e 's/HIP_BACKEND_MODE[[:space:]]*=[[:space:]]*False/HIP_BACKEND_MODE = True/' ${HIP_BACKEND_PY}
            COMMAND 
            touch ${HIP_BACKEND_PY_STAMP}
        DEPENDS ${HIP_BACKEND_PY}
        COMMENT "Modifying hip_backend.py to enable HIP_BACKEND_MODE."
    )

    add_custom_target(
        modify_file 
        ALL 
        DEPENDS ${HIP_BACKEND_PY_STAMP}
        COMMENT "Checking and applying modifications to hip_backend.py"
    )
endif()