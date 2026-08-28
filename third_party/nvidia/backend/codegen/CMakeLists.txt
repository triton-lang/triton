cmake_minimum_required(VERSION 3.20)

project(triton_nvidia_codegen LANGUAGES C CXX)

find_package(LLVM REQUIRED CONFIG)

set(TRITON_NVIDIA_LLVM_REVISION "unknown" CACHE STRING
    "Independently pinned NVIDIA LLVM revision and build number")

add_library(triton_nvidia_codegen SHARED nvptx_codegen.cc)
target_compile_features(triton_nvidia_codegen PRIVATE cxx_std_17)
target_include_directories(triton_nvidia_codegen SYSTEM PRIVATE
                           ${LLVM_INCLUDE_DIRS})
target_compile_definitions(triton_nvidia_codegen PRIVATE
    TRITON_NVIDIA_LLVM_REVISION="${TRITON_NVIDIA_LLVM_REVISION}")

if(NOT LLVM_ENABLE_EH)
  target_compile_options(triton_nvidia_codegen PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/EHs-c->
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fno-exceptions>)
endif()
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(triton_nvidia_codegen PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/GR->
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fno-rtti>)
endif()

llvm_map_components_to_libnames(TRITON_NVIDIA_LLVM_LIBRARIES
    NVPTXCodeGen NVPTXDesc NVPTXInfo
    IRReader IPO ScalarOpts TransformUtils Analysis CodeGen Target Support)
target_link_libraries(triton_nvidia_codegen PRIVATE
                      ${TRITON_NVIDIA_LLVM_LIBRARIES})

foreach(llvm_library IN LISTS TRITON_NVIDIA_LLVM_LIBRARIES)
  get_target_property(llvm_library_type ${llvm_library} TYPE)
  if(NOT llvm_library_type STREQUAL "STATIC_LIBRARY")
    message(FATAL_ERROR "Standalone NVIDIA code generation requires static LLVM libraries: ${llvm_library}")
  endif()
endforeach()

set_target_properties(triton_nvidia_codegen PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    POSITION_INDEPENDENT_CODE ON)

if(APPLE)
  set(TRITON_NVIDIA_EXPORTS "${CMAKE_CURRENT_BINARY_DIR}/nvidia-codegen.exports")
  file(GENERATE OUTPUT "${TRITON_NVIDIA_EXPORTS}" CONTENT
      "_triton_nvptx_compile\n_triton_nvptx_free\n_triton_nvptx_revision\n")
  target_link_options(triton_nvidia_codegen PRIVATE
      "LINKER:-exported_symbols_list,${TRITON_NVIDIA_EXPORTS}")
elseif(UNIX)
  set(TRITON_NVIDIA_EXPORTS "${CMAKE_CURRENT_BINARY_DIR}/nvidia-codegen.exports")
  file(GENERATE OUTPUT "${TRITON_NVIDIA_EXPORTS}" CONTENT
      "{\n  global:\n    triton_nvptx_compile;\n    triton_nvptx_free;\n    triton_nvptx_revision;\n  local:\n    *;\n};\n")
  target_link_options(triton_nvidia_codegen PRIVATE
      "LINKER:--version-script,${TRITON_NVIDIA_EXPORTS}"
      "LINKER:--exclude-libs,ALL"
      "LINKER:-z,defs")
endif()

install(TARGETS triton_nvidia_codegen
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION lib)
