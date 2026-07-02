if(NOT APPLE)
  message(STATUS "Metal backend is only supported on macOS/Apple platforms")
  return()
endif()

# Require macOS 15+ for Metal 3.2, prefer macOS 26+ for Metal 4.0
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS version")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/TritonMetalToLLVM)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(lib)

add_triton_plugin(TritonMetal
  triton_metal.cc

  LINK_LIBS
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonGPUToLLVM
  TritonGPUTransforms
  TritonMetalToLLVM
  TritonTools
  MLIRLLVMDialect
  MLIRPass
)

# Compile Objective-C Metal runtime
if(APPLE)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
    target_link_libraries(TritonMetal PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
  endif()
endif()
