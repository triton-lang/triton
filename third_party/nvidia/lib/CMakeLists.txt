add_subdirectory(Dialect)
# TritonNVIDIAGPUToLLVM is always entered: when the NVIDIA backend is disabled
# it only builds the backend-agnostic utility library (see its CMakeLists).
add_subdirectory(TritonNVIDIAGPUToLLVM)
if(TRITON_BUILD_NVIDIA_BACKEND)
  add_subdirectory(NVGPUToLLVM)
endif()
