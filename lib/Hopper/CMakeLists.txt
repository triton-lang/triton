# TODO: re-enable generation of the hopper helper.
# For now we just commit a pre-built bc file.
#cmake_path(GET LLVM_LIBRARY_DIR PARENT_PATH LLVM_DIR)
#set(CUDA_PATH "/usr/local/cuda")
#if(DEFINED ENV{CUDA_PATH})
#        set(CUDA_PATH $ENV{CUDA_PATH})
#endif()
#
#set(outfile "libhopper_helpers.bc")
#add_custom_target(HopperHelpers ALL
#        COMMAND ${LLVM_DIR}/bin/clang -c
#        -O3 -emit-llvm
#        -fno-unwind-tables -fno-exceptions
#        --cuda-gpu-arch=sm_90a
#        --cuda-device-only
#        -D__CUDACC__ -D__CUDABE__ -D__CUDA_LIBDEVICE__
#        -I${CUDA_PATH}/include
#        -target nvptx64-nvidia-gpulibs
#        ${CMAKE_CURRENT_SOURCE_DIR}/HopperHelpers.c -o ${outfile}
#        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/HopperHelpers.c
#        BYPRODUCTS ${outfile}
#        COMMENT "Building LLVM bitcode ${outfile}"
#        VERBATIM
#)
