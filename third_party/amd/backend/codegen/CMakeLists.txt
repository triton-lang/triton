cmake_minimum_required(VERSION 3.20)

project(triton_amd_codegen LANGUAGES C CXX)

find_package(LLVM REQUIRED CONFIG)

set(TRITON_AMD_LLVM_REVISION "unknown" CACHE STRING
    "Independently pinned AMD LLVM revision and build number")

add_library(triton_amd_codegen SHARED amdgpu_codegen.cc)
target_compile_features(triton_amd_codegen PRIVATE cxx_std_17)
target_include_directories(triton_amd_codegen SYSTEM PRIVATE
                           ${LLVM_INCLUDE_DIRS})
target_compile_definitions(triton_amd_codegen PRIVATE
    TRITON_AMD_LLVM_REVISION="${TRITON_AMD_LLVM_REVISION}")

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
      "LINKER:--exclude-libs,ALL")
endif()

install(TARGETS triton_amd_codegen
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION lib)
