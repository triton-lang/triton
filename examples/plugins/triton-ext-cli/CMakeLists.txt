get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTRITON_EXT_CLI")
add_executable(triton-ext-cli triton-ext-cli.cpp)

target_include_directories(triton-ext-cli PRIVATE ${PROJECT_SOURCE_DIR}/bin)
target_compile_options(triton-ext-cli PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-ext-cli PRIVATE
  ${triton_libs}
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-ext-cli)
set_target_properties(triton-ext-cli PROPERTIES ENABLE_EXPORTS 1)

if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(triton-ext-cli PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY
  "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../plugins")
endif(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
