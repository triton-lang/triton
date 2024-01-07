add_library(triton_nvidia SHARED ${CMAKE_CURRENT_SOURCE_DIR}/triton_nvidia.cc)
set(TRITON_NVIDIA_LIBRARIES
NVGPUIR
  NVGPUToLLVM
  TritonNvidiaGPUTransforms
  TritonGPUToLLVM
  MLIRNVVMDialect
  ${LLVM_LIBRARY_DIR}/libMLIRNVVMToLLVMIRTranslation.a
  ${LLVM_LIBRARY_DIR}/libMLIRGPUToNVVMTransforms.a
  ${LLVM_LIBRARY_DIR}/libMLIRGPUToGPURuntimeTransforms.a
  ${LLVM_LIBRARY_DIR}/libMLIRGPUTransforms.a
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRArithTransforms
  MLIRVectorDialect
  MLIRVectorToLLVM
  MLIRFuncToLLVM
)
target_link_libraries(triton_nvidia PRIVATE ${TRITON_NVIDIA_LIBRARIES} "-Wl,--no-as-needed" triton)
target_link_libraries(triton_nvidia ${PYTHON_LDFLAGS})
