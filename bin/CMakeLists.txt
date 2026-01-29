get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_executable(triton-opt triton-opt.cpp)

target_compile_options(triton-opt PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-opt PRIVATE
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

mlir_check_all_link_libraries(triton-opt)

add_executable(triton-reduce triton-reduce.cpp)
mlir_check_all_link_libraries(triton-reduce)
target_compile_options(triton-reduce PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})

target_link_libraries(triton-reduce PRIVATE
  ${triton_libs}
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
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
  TritonAMDGPUTestAnalysis
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
  TritonAMDGPUTestAnalysis
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
  )
