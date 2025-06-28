include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonAMD ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_amd.cc LINK_LIBS TritonAMDGPUToLLVM TritonAMDGPUTransforms TritonAMDGPUDialectToLLVM)
  target_link_libraries(TritonAMD PRIVATE Python3::Module pybind11::headers)

  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/third_party/amd
    FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.td"
      PATTERN "*.h.inc"
  )
endif()
if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
add_subdirectory(test)
