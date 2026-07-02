if(NOT APPLE)
  message(STATUS "Metal backend is only supported on macOS/Apple platforms")
  return()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_triton_plugin(TritonMetal
  triton_metal.cc

  LINK_LIBS
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
  TritonTools
)
