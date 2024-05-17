if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonCPU ${CMAKE_CURRENT_SOURCE_DIR}/triton_cpu.cc LINK_LIBS TritonCPUToLLVM)
endif()
