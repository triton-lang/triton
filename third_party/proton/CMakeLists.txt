project(Proton LANGUAGES CXX)

include(Backend.cmake)
include(CheckCXXSourceCompiles)

set(PROTON_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
set(PROTON_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")

# ============ Check for includes =============
if(NOT CUPTI_INCLUDE_DIR)
  message(FATAL_ERROR "CUPTI include directory not defined")
endif()
if(NOT ROCM_INCLUDE_DIR)
  message(FATAL_ERROR "ROCM include directory not defined")
endif()
if(NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "JSON include directory not defined")
endif()

set(_proton_saved_cmake_required_includes "${CMAKE_REQUIRED_INCLUDES}")
set(_proton_saved_cmake_required_definitions "${CMAKE_REQUIRED_DEFINITIONS}")
set(CMAKE_REQUIRED_INCLUDES "${ROCPROFILER_SDK_INCLUDE_DIR}")
set(CMAKE_REQUIRED_DEFINITIONS -D__HIP_PLATFORM_AMD__)
unset(PROTON_ROCPROFILER_SDK_HAS_HIP_GRAPH CACHE)
check_cxx_source_compiles([=[
  #include <rocprofiler-sdk/callback_tracing.h>
  #include <rocprofiler-sdk/external_correlation.h>
  int main() {
    auto kind = ROCPROFILER_CALLBACK_TRACING_HIP_GRAPH;
    auto op = ROCPROFILER_HIP_GRAPH_OPERATION_HIP_GRAPH_LAUNCH;
    rocprofiler_callback_tracing_hip_graph_data_t* data = nullptr;
    auto request = ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH;
    rocprofiler_external_correlation_id_request_cb_t cb = nullptr;
    return static_cast<int>(kind) + static_cast<int>(op) +
           static_cast<int>(request) + (data != nullptr) + (cb != nullptr);
  }
]=] PROTON_ROCPROFILER_SDK_HAS_HIP_GRAPH)
set(CMAKE_REQUIRED_INCLUDES "${_proton_saved_cmake_required_includes}")
set(CMAKE_REQUIRED_DEFINITIONS "${_proton_saved_cmake_required_definitions}")

if(PROTON_ROCPROFILER_SDK_HAS_HIP_GRAPH)
  message(STATUS "Proton ROCProfiler-SDK HIP graph tracing: enabled")
else()
  message(STATUS "Proton ROCProfiler-SDK HIP graph tracing: disabled")
endif()

# ============ Dependencies =============
find_package(Python3 REQUIRED Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

# ============ Define a GLOBAL property to store object-libraries ============
set_property(GLOBAL PROPERTY PROTON_LIBS "")

# ============ Define a function to create object libraries ============
function(add_proton_library name)
  add_library(${name} OBJECT ${ARGN})

  target_link_libraries(${name} PRIVATE Python3::Module pybind11::headers)

  # Use system to skip warnings caused by legacy clang compilers
  target_include_directories(${name}
    SYSTEM PRIVATE
      ${ROCPROFILER_SDK_INCLUDE_DIR}
      "${ROCM_INCLUDE_DIR}"
  )

  target_include_directories(${name}
    PRIVATE
      "${CUPTI_INCLUDE_DIR}"
      "${JSON_INCLUDE_DIR}"
      "${PROTON_COMMON_DIR}/include"
      "${PROTON_SRC_DIR}/include"
  )

  # If HIP is AMD-based
  target_compile_definitions(${name} PRIVATE __HIP_PLATFORM_AMD__)
  if(PROTON_ROCPROFILER_SDK_HAS_HIP_GRAPH)
    target_compile_definitions(${name} PRIVATE
      PROTON_ROCPROFILER_SDK_HAS_HIP_GRAPH=1)
  else()
    target_compile_definitions(${name} PRIVATE
      PROTON_ROCPROFILER_SDK_HAS_HIP_GRAPH=0)
  endif()

  # Append this library name to the GLOBAL property "PROTON_LIBS"
  set_property(GLOBAL APPEND PROPERTY PROTON_LIBS ${name})
endfunction()

# ============ Add subdirectory with actual code that calls add_proton_library ============
add_subdirectory("${PROTON_COMMON_DIR}")
add_subdirectory("${PROTON_SRC_DIR}")

# ============ Add subdirectory with proton tests ============
add_subdirectory(test)

# ============ Discover any Triton backends registering Proton backends ============
discover_proton_backends()

# ============ Codegen the templates allowing for Proton backend registration ============
codegen_proton_backend_templates()

# ============ Possibly handle macOS specifics ============
if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  # Other platforms build with -flto, but we found that this adds significant overhead to our macos CI without providing a major benefit.
  set(PROTON_PYTHON_LDFLAGS "-undefined dynamic_lookup")
endif()

# ============ Collect all object libraries from property and build final shared lib ============
get_property(_proton_obj_libs GLOBAL PROPERTY PROTON_LIBS)

if(NOT _proton_obj_libs)
  message(WARNING "No object libraries were defined in 'PROTON_LIBS'!")
endif()

set(_proton_obj_sources "")
foreach(_lib IN LISTS _proton_obj_libs)
  list(APPEND _proton_obj_sources $<TARGET_OBJECTS:${_lib}>)
  message(STATUS "Collecting object files from ${_lib}")
endforeach()

add_library(proton SHARED ${_proton_obj_sources})

target_link_libraries(proton PRIVATE Python3::Module)
# Apply any macOS linker flags or extra link options
if(PROTON_PYTHON_LDFLAGS)
  target_link_options(proton PRIVATE ${PROTON_PYTHON_LDFLAGS})
endif()

# ============ Link any external libs required by Proton backends ============
link_proton_backend_external_libs()
