project(Proton CXX)

set(PROTON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
set(PROTON_EXTERN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern)
file(GLOB_RECURSE PROTON_SRC ${PROTON_SRC_DIR}/lib/*.cpp)
add_library(proton SHARED ${PROTON_SRC} ${PROTON_SRC_DIR}/${PROJECT_NAME}.cpp)

include_directories(${CUPTI_INCLUDE_DIR})
include_directories(${ROCTRACER_INCLUDE_DIR})
target_compile_definitions(proton PUBLIC __HIP_PLATFORM_AMD__)

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

# Check if the platform is MacOS
if(APPLE)
  set(PROTON_PYTHON_LDFLAGS "-undefined dynamic_lookup -flto")
endif()

target_link_libraries(proton PRIVATE ${Python_LIBRARIES} ${PROTON_PYTHON_LDFLAGS})
