# Test-analysis helper lib only exists when the AMD backend is built.
if(TRITON_BUILD_AMD_BACKEND)
  set(TRITON_AMD_TEST_LIBS TritonAMDGPUTestAnalysis)
else()
  set(TRITON_AMD_TEST_LIBS "")
endif()

get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_executable(triton-opt triton-opt.cpp)

target_compile_options(triton-opt PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-opt PRIVATE
  ${triton_libs}
  # tests
  TritonTestAnalysis
  TritonTestDialect
  ${TRITON_AMD_TEST_LIBS}
  TritonTestProton
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-opt)

add_executable(triton-reduce triton-reduce.cpp)
mlir_check_all_link_libraries(triton-reduce)
target_compile_options(triton-reduce PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})

target_link_libraries(triton-reduce PRIVATE
  ${triton_libs}
  # tests
  TritonTestAnalysis
  TritonTestDialect
  ${TRITON_AMD_TEST_LIBS}
  TritonTestProton
  # MLIR core
  MLIRReduceLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-reduce)

add_executable(triton-lsp triton-lsp.cpp)

target_compile_options(triton-lsp PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-lsp PRIVATE
  ${triton_libs}
  # tests
  TritonTestAnalysis
  TritonTestDialect
  ${TRITON_AMD_TEST_LIBS}
  TritonTestProton
  # MLIR core
  MLIRLspServerLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-lsp)


add_executable(triton-llvm-opt triton-llvm-opt.cpp)
add_dependencies(triton-llvm-opt intrinsics_gen)
target_compile_options(triton-llvm-opt PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-llvm-opt PRIVATE
  TritonLLVMIR

  LLVMAnalysis
  LLVMCore
  LLVMSupport
  LLVMOption
  LLVMCodeGen
)
export_executable_symbols_for_plugins(triton-llvm-opt)


add_executable(triton-tensor-layout triton-tensor-layout.cpp)
target_compile_options(triton-tensor-layout PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-tensor-layout PRIVATE
  ${triton_libs}
  TritonTestAnalysis
  TritonTestDialect
  TritonTestProton
  ${TRITON_AMD_TEST_LIBS}
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

# When a backend is disabled, only its tablegen headers are generated (no
# libraries), so consumers that include backend dialect headers need explicit
# order-only dependencies on the generation targets.
foreach(_tg
    TritonAMDGPUTableGen TritonAMDGPUAttrDefsIncGen TritonAMDGPUTransformsIncGen
    TritonAMDGPUConversionPassIncGen
    NVGPUTableGen NVGPUAttrDefsIncGen
    NVWSTableGen NVWSAttrDefsIncGen NVWSTransformsIncGen
    NVGPUConversionPassIncGen TritonNVIDIAGPUConversionPassIncGen)
  if(TARGET ${_tg})
    foreach(_tool triton-opt triton-reduce triton-lsp triton-tensor-layout)
      if(TARGET ${_tool})
        add_dependencies(${_tool} ${_tg})
      endif()
    endforeach()
  endif()
endforeach()
