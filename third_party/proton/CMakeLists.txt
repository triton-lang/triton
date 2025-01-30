project(Proton CXX)

set(PROTON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/csrc)

if(NOT CUPTI_INCLUDE_DIR)
  message(FATAL_ERROR "CUPTI include directory not defined")
endif()

if(NOT ROCTRACER_INCLUDE_DIR)
  message(FATAL_ERROR "ROCTRACER include directory not defined")
endif()

if(NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "JSON include directory not defined")
endif()

set_property(GLOBAL PROPERTY PROTON_LIBS "")

function(add_proton_library name)
  set_property(GLOBAL APPEND PROPERTY PROTON_LIBS ${name})
  add_library(${name} OBJECT)
  target_sources(${name}
    PRIVATE ${ARGN}
    INTERFACE $<TARGET_OBJECTS:${name}>
  )
endfunction()

find_package(Python3 REQUIRED Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${CUPTI_INCLUDE_DIR})
include_directories(SYSTEM ${ROCTRACER_INCLUDE_DIR})
include_directories(${JSON_INCLUDE_DIR})
include_directories(${PROTON_SRC_DIR}/include)
add_compile_options(-D__HIP_PLATFORM_AMD__)

add_subdirectory(${PROTON_SRC_DIR})

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

# Collect object libraries and make a shared lib
get_property(PROTON_OBJ_LIBS GLOBAL PROPERTY PROTON_LIBS)

if(PROTON_OBJ_LIBS)
  foreach(_lib IN LISTS PROTON_OBJ_LIBS)
    message(STATUS "Found proton object library: ${_lib}")
  endforeach()
else()
  message(WARNING "No object libraries were defined. PROTON_LIBS is empty.")
endif()

# Build a single "proton" shared library from all object libraries
set(PROTON_OBJ_SOURCES "")

foreach(_lib IN LISTS PROTON_OBJ_LIBS)
  list(APPEND PROTON_OBJ_SOURCES $<TARGET_OBJECTS:${_lib}>)
endforeach()

add_library(proton SHARED ${PROTON_OBJ_SOURCES})

target_link_libraries(proton PRIVATE Python3::Module pybind11::headers)
target_link_options(proton PRIVATE ${PROTON_PYTHON_LDFLAGS})
