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

unset(PROTON_ROCPROFILER_SDK_HAS_HIP_GRAPH CACHE)
set(_proton_saved_cmake_required_includes "${CMAKE_REQUIRED_INCLUDES}")
set(CMAKE_REQUIRED_INCLUDES "${ROCPROFILER_SDK_INCLUDE_DIR};${ROCM_INCLUDE_DIR}")
check_cxx_source_compiles([=[
  #include <rocprofiler-sdk/fwd.h>
  int main() {
    auto op = ROCPROFILER_HIP_GRAPH_OPERATION_EXEC_LAUNCH;
    return static_cast<int>(op);
  }
]=] PROTON_ROCPROFILER_SDK_HAS_HIP_GRAPH)
set(CMAKE_REQUIRED_INCLUDES "${_proton_saved_cmake_required_includes}")

if(PROTON_ROCPROFILER_SDK_HAS_HIP_GRAPH)
  message(STATUS "Proton ROCProfiler-SDK HIP graph tracing: enabled")
else()
  message(STATUS "Proton ROCProfiler-SDK HIP graph tracing: disabled")
endif()

unset(PROTON_ROCPROFILER_SDK_HAS_PC_SAMPLING CACHE)
set(CMAKE_REQUIRED_INCLUDES "${ROCPROFILER_SDK_INCLUDE_DIR}")
check_cxx_source_compiles([=[
  #include <rocprofiler-sdk/pc_sampling.h>
  int main() {
    rocprofiler_pc_sampling_configuration_t config{};
    rocprofiler_pc_sampling_record_host_trap_v0_t hostTrap{};
    rocprofiler_pc_sampling_record_stochastic_v0_t stochastic{};
    rocprofiler_available_pc_sampling_configurations_cb_t cb = nullptr;
    auto method = ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC;
    auto unit = ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS;
    auto reason = ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT;
    (void) config;
    (void) hostTrap;
    (void) stochastic;
    (void) cb;
    (void) method;
    (void) unit;
    (void) reason;
    return 0;
  }
]=] PROTON_ROCPROFILER_SDK_HAS_PC_SAMPLING)

if(PROTON_ROCPROFILER_SDK_HAS_PC_SAMPLING)
  message(STATUS "Proton ROCProfiler-SDK PC sampling: enabled")
else()
  message(STATUS "Proton ROCProfiler-SDK PC sampling: disabled")
endif()

set(_PROTON_ROCPROFILER_SDK_LINK_PATHS "")
foreach(_path IN ITEMS
    "${TRITON_ROCPROFILER_SDK_DEVEL_LIB_PATH}")
  if(_path)
    list(APPEND _PROTON_ROCPROFILER_SDK_LINK_PATHS "${_path}")
  endif()
endforeach()

find_library(PROTON_AMD_COMGR_LIBRARY
  NAMES amd_comgr
  HINTS ${_PROTON_ROCPROFILER_SDK_LINK_PATHS}
  NO_DEFAULT_PATH)
find_library(PROTON_DW_LIBRARY NAMES dw)

unset(PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE CACHE)
if(PROTON_ROCPROFILER_SDK_HAS_PC_SAMPLING AND PROTON_AMD_COMGR_LIBRARY AND
   PROTON_DW_LIBRARY)
  set(_PROTON_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
  set(_PROTON_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES}")
  set(CMAKE_REQUIRED_INCLUDES
      "${ROCPROFILER_SDK_INCLUDE_DIR}"
      "${ROCM_INCLUDE_DIR}")
  set(CMAKE_REQUIRED_LIBRARIES
      "${PROTON_AMD_COMGR_LIBRARY}"
      "${PROTON_DW_LIBRARY}")
  check_cxx_source_compiles([=[
    #include <rocprofiler-sdk/pc_sampling.h>
    #include <rocprofiler-sdk/cxx/codeobj/code_printing.hpp>
    int main() {
      rocprofiler_pc_sampling_record_stochastic_v0_t sample{};
      using rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate;
      using rocprofiler::sdk::codeobj::disassembly::Instruction;
      CodeobjAddressTranslate translator;
      Instruction inst;
      (void) sample;
      (void) inst.comment;
      const void *data = nullptr;
      translator.addDecoder(data, 0, 1, 0, 0);
      (void) translator.get(1, 0);
      (void) translator.removeDecoder(1, 0);
      return 0;
    }
  ]=] PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE)
  set(CMAKE_REQUIRED_INCLUDES "${_PROTON_REQUIRED_INCLUDES}")
  set(CMAKE_REQUIRED_LIBRARIES "${_PROTON_REQUIRED_LIBRARIES}")
endif()

if(PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE)
  message(STATUS "Proton ROCProfiler-SDK code object source locations: enabled")
else()
  message(STATUS "Proton ROCProfiler-SDK code object source locations: disabled")
endif()

# ============ Dependencies =============
find_package(Python3 REQUIRED Interpreter Development.Module)
if(NOT TARGET nanobind-static)
  if(NOT TARGET Python::Module)
    add_library(Python::Module ALIAS Python3::Module)
    set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
    set(Python_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
    set(Python_SOABI "${Python3_SOABI}")
    set(Python_VERSION "${Python3_VERSION}")
    set(Python_INTERPRETER_ID "${Python3_INTERPRETER_ID}")
  endif()

  find_package(nanobind CONFIG REQUIRED HINTS "${Python3_SITELIB}")
  nanobind_build_library(nanobind-static)
  target_link_libraries(nanobind-static PRIVATE Python3::Module)
endif()

# ============ Define a GLOBAL property to store object-libraries ============
set_property(GLOBAL PROPERTY PROTON_LIBS "")

# ============ Define a function to create object libraries ============
function(add_proton_library name)
  add_library(${name} OBJECT ${ARGN})

  target_link_libraries(${name} PRIVATE Python3::Module nanobind-static)

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
  if(PROTON_ROCPROFILER_SDK_HAS_PC_SAMPLING)
    target_compile_definitions(${name} PRIVATE
      PROTON_ROCPROFILER_SDK_HAS_PC_SAMPLING=1)
  else()
    target_compile_definitions(${name} PRIVATE
      PROTON_ROCPROFILER_SDK_HAS_PC_SAMPLING=0)
  endif()
  if(PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE)
    target_compile_definitions(${name} PRIVATE
      PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE=1)
  else()
    target_compile_definitions(${name} PRIVATE
      PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE=0)
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

target_link_libraries(proton PRIVATE Python3::Module nanobind-static)
if(PROTON_ROCPROFILER_SDK_HAS_CODEOBJ_ADDRESS_TRANSLATE)
  target_link_libraries(proton PRIVATE
    "${PROTON_AMD_COMGR_LIBRARY}"
    "${PROTON_DW_LIBRARY}")
endif()
# Apply any macOS linker flags or extra link options
if(PROTON_PYTHON_LDFLAGS)
  target_link_options(proton PRIVATE ${PROTON_PYTHON_LDFLAGS})
endif()

# ============ Link any external libs required by Proton backends ============
link_proton_backend_external_libs()
