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

if(PYTHON_INCLUDE_DIRS)
  # We have PYTHON_INCLUDE_DIRS set--this is what we expect when building
  # using pip install.
  include_directories(${PYTHON_INCLUDE_DIRS})
  include_directories(${PYBIND11_INCLUDE_DIR})
else()
  # Otherwise, we might be building from top CMakeLists.txt directly.
  # Try to find Python and pybind11 packages.
  find_package(Python3 REQUIRED Interpreter Development)
  find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")
  include_directories(${Python3_INCLUDE_DIRS})
  include_directories(${pybind11_INCLUDE_DIR})
endif()

# Check if the platform is MacOS
if(APPLE)
  set(PROTON_PYTHON_LDFLAGS "-undefined dynamic_lookup -flto")
endif()

include_directories(${CUPTI_INCLUDE_DIR})
include_directories(${ROCTRACER_INCLUDE_DIR})
target_compile_definitions(proton PRIVATE __HIP_PLATFORM_AMD__)

target_link_libraries(proton PRIVATE ${Python_LIBRARIES} ${PROTON_PYTHON_LDFLAGS})
