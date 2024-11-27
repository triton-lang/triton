project(Proton CXX)

set(PROTON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
set(PROTON_EXTERN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern)
file(GLOB_RECURSE PROTON_SRC ${PROTON_SRC_DIR}/lib/*.cpp)
add_library(proton SHARED ${PROTON_SRC} ${PROTON_SRC_DIR}/${PROJECT_NAME}.cpp)

if(NOT CUPTI_INCLUDE_DIR)
  message(FATAL_ERROR "CUPTI include directory not defined")
endif()
if(NOT ROCTRACER_INCLUDE_DIR)
  message(FATAL_ERROR "ROCTRACER include directory not defined")
endif()
if (NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "JSON include directory not defined")
endif()

include_directories(${JSON_INCLUDE_DIR})
include_directories(${PROTON_SRC_DIR}/include)
include_directories(${PROTON_EXTERN_DIR})

find_package(Python3 REQUIRED Interpreter Development)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

# Check if the platform is MacOS
if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  # Other platforms build with -flto, but we found that this adds significant overhead to our macos CI without providing a major benefit.
  set(PROTON_PYTHON_LDFLAGS "-undefined dynamic_lookup")
endif()

if(DEFINED CUPTI_LIB_DIR)
  message(STATUS "CUPTI lib directory: ${CUPTI_LIB_DIR}")
  add_compile_definitions(CUPTI_LIB_DIR=${CUPTI_LIB_DIR})
endif()

include_directories(${CUPTI_INCLUDE_DIR})
include_directories(SYSTEM ${ROCTRACER_INCLUDE_DIR})
target_compile_definitions(proton PRIVATE __HIP_PLATFORM_AMD__)

target_link_libraries(proton PRIVATE Python3::Module pybind11::headers)
target_link_options(proton PRIVATE ${PROTON_PYTHON_LDFLAGS})
