cmake_minimum_required(VERSION 3.20)

project(triton_amd_codegen LANGUAGES C CXX)

find_package(LLVM REQUIRED CONFIG)

set(TRITON_AMD_LLVM_REVISION "unknown" CACHE STRING
    "Independently pinned AMD LLVM revision and build number")

# Compile the option guard against this backend's LLVM so its registry remains
# independent of the copy linked into Triton's core library.
set(TRITON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
add_library(triton_amd_codegen SHARED amdgpu_codegen.cc
            "${TRITON_SOURCE_DIR}/lib/Tools/LLVMOptions.cpp")
target_compile_features(triton_amd_codegen PRIVATE cxx_std_20)
target_include_directories(triton_amd_codegen PRIVATE
                           "${TRITON_SOURCE_DIR}/include")
target_include_directories(triton_amd_codegen SYSTEM PRIVATE
                           ${LLVM_INCLUDE_DIRS})
target_compile_definitions(triton_amd_codegen PRIVATE
    TRITON_AMD_LLVM_REVISION="${TRITON_AMD_LLVM_REVISION}")

# Keep this library's NDEBUG state identical to that of the LLVM it links.
# LLVM's headers gate data members on NDEBUG, so a mismatch means the two sides
# disagree about object layout; see LLVMABIGuard.h for what diverges and what
# it costs. This library is configured independently of LLVM, so neither
# direction is automatic. CMake's build-type flags define NDEBUG for Release,
# and target_compile_options land after CMAKE_CXX_FLAGS_<CONFIG>, so -UNDEBUG
# wins. LLVMABIGuard.h re-checks the result at compile time rather than
# trusting this to be right.
if(LLVM_ENABLE_ASSERTIONS)
  target_compile_options(triton_amd_codegen PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
else()
  target_compile_definitions(triton_amd_codegen PRIVATE NDEBUG)
endif()

if(NOT LLVM_ENABLE_EH)
  target_compile_options(triton_amd_codegen PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/EHs-c->
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fno-exceptions>)
endif()
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(triton_amd_codegen PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/GR->
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fno-rtti>)
endif()

llvm_map_components_to_libnames(TRITON_AMD_LLVM_LIBRARIES
    AMDGPUCodeGen AMDGPUDesc AMDGPUInfo AMDGPUAsmParser AMDGPUUtils
    IRReader IPO ScalarOpts TransformUtils Analysis CodeGen Target Support)
target_link_libraries(triton_amd_codegen PRIVATE
                      ${TRITON_AMD_LLVM_LIBRARIES})

foreach(llvm_library IN LISTS TRITON_AMD_LLVM_LIBRARIES)
  get_target_property(llvm_library_type ${llvm_library} TYPE)
  if(NOT llvm_library_type STREQUAL "STATIC_LIBRARY")
    message(FATAL_ERROR "Standalone AMD code generation requires static LLVM libraries: ${llvm_library}")
  endif()
endforeach()

set_target_properties(triton_amd_codegen PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    POSITION_INDEPENDENT_CODE ON)

if(APPLE)
  set(TRITON_AMD_EXPORTS "${CMAKE_CURRENT_BINARY_DIR}/amd-codegen.exports")
  file(GENERATE OUTPUT "${TRITON_AMD_EXPORTS}" CONTENT
      "_triton_amdgpu_assemble\n_triton_amdgpu_compile\n_triton_amdgpu_free\n_triton_amdgpu_revision\n")
  target_link_options(triton_amd_codegen PRIVATE
      "LINKER:-exported_symbols_list,${TRITON_AMD_EXPORTS}")
elseif(UNIX)
  set(TRITON_AMD_EXPORTS "${CMAKE_CURRENT_BINARY_DIR}/amd-codegen.exports")
  file(GENERATE OUTPUT "${TRITON_AMD_EXPORTS}" CONTENT
      "{\n  global:\n    triton_amdgpu_assemble;\n    triton_amdgpu_compile;\n    triton_amdgpu_free;\n    triton_amdgpu_revision;\n  local:\n    *;\n};\n")
  target_link_options(triton_amd_codegen PRIVATE
      "LINKER:--version-script,${TRITON_AMD_EXPORTS}"
      "LINKER:--exclude-libs,ALL"
      "LINKER:-z,defs")
endif()

install(TARGETS triton_amd_codegen
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION lib)
