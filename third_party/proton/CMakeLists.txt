project(Proton CXX)

set(PROTON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
set(PROTON_EXTERN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern)
file(GLOB_RECURSE PROTON_SRC ${PROTON_SRC_DIR}/lib/*.cpp)
add_library(proton SHARED ${PROTON_SRC} ${PROTON_SRC_DIR}/${PROJECT_NAME}.cpp)

find_package(CUDAToolkit)

if(${CUDAToolkit_VERSION_MAJOR} VERSION_LESS 11)
  message(FATAL_ERROR "CUDA 11 or higher is required")
endif()

# Try to find CUPTI from the default include dir, if not found, search the
# EXTRAS dir
find_path(
  CUPTI_INCLUDE_DIR
  NAMES cupti.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS}
  PATH_SUFFIXES include)

if(NOT CUPTI_INCLUDE_DIR)
  find_path(
    CUPTI_INCLUDE_DIR
    NAMES cupti.h
    HINTS ${CUDAToolkit_ROOT}/extras/CUPTI
    PATH_SUFFIXES include)
endif()

# Check if CUPTI was found
if(NOT CUPTI_INCLUDE_DIR)
  message(FATAL_ERROR "CUPTI include directory not found: CUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
else()
  message(STATUS "Found CUPTI include directory: ${CUPTI_INCLUDE_DIR}")
endif()

include_directories(${PYBIND11_INCLUDE_DIR})
include_directories(${JSON_INCLUDE_DIR})
include_directories(${PROTON_SRC_DIR}/include)
include_directories(${PROTON_EXTERN_DIR})

if(PYTHON_INCLUDE_DIRS)
  include_directories(${PYTHON_INCLUDE_DIRS})
else()
  find_package(Python3 REQUIRED Interpreter Development)
  include_directories(${Python3_INCLUDE_DIRS})
endif()

include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${CUPTI_INCLUDE_DIR})
target_link_libraries(proton ${Python_LIBRARIES})
