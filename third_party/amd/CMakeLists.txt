include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonAMD
    ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_amd.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/python/dylib_utils.cc
    LINK_LIBS MLIRROCDLToLLVMIRTranslation TritonAMDGPUToLLVM TritonAMDGPUTransforms TritonAMDGPUDialectToLLVM)
  target_link_libraries(TritonAMD PRIVATE Python3::Module nanobind-static)
endif()
add_subdirectory(test)
