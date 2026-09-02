# SPDX-License-Identifier: Apache-2.0

# The compiler frontend is Python-first.  The generic perf library is a small,
# stable ABI shim that Graph Compiler loads once and uses for all generated TPC
# ELFs.  Keep compile-only Triton builds working on hosts without SynapseAI.
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonGaudi ${CMAKE_CURRENT_SOURCE_DIR}/triton_gaudi.cc)
  target_link_libraries(TritonGaudi PRIVATE Python3::Module nanobind-static)
endif()

find_path(
  TRITON_GAUDI_TPC_INCLUDE_DIR
  NAMES habanalabs/tpc_kernel_lib_interface.h
  PATHS /usr/include)

if(TRITON_GAUDI_TPC_INCLUDE_DIR)
  add_library(triton_gaudi_perf SHARED runtime/perf_lib.cpp)
  target_include_directories(
    triton_gaudi_perf
    PRIVATE ${TRITON_GAUDI_TPC_INCLUDE_DIR} runtime)
  target_compile_features(triton_gaudi_perf PRIVATE cxx_std_17)
  target_compile_options(triton_gaudi_perf PRIVATE -Wall -Wextra -Werror)
  set_target_properties(
    triton_gaudi_perf
    PROPERTIES OUTPUT_NAME triton_gaudi_perf)
  if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(
      triton_gaudi_perf
      PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                 "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../backends/gaudi/lib")
  endif()
endif()
