include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include) # For #include "Dialect/TritonXPU/IR/Dialect.h"
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(device)
add_subdirectory(include)
add_subdirectory(lib)

if(TRITON_BUILD_PYTHON_MODULE)
    add_triton_plugin(TritonXPU ${CMAKE_CURRENT_SOURCE_DIR}/triton_xpu.cc) #LINK_LIBS TritonToTritonXPU TritonXPUToLLVM)
endif()
