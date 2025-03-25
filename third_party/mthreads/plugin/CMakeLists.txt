include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)

if(TRITON_BUILD_PYTHON_MODULE)
  find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
  add_library(mthreadsTritonPlugin SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_mthreads.cc
  )
  set_target_properties(mthreadsTritonPlugin PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    POSITION_INDEPENDENT_CODE ON
  )
  target_link_libraries(mthreadsTritonPlugin PRIVATE
    TritonMTGPUToLLVM
    # Py
    ${Python3_LIBRARIES}
    ${PYTHON_LDFLAGS}
  )
endif()
