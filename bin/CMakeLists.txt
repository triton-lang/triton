get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_llvm_executable(triton-opt triton-opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(triton-opt)
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

add_llvm_executable(triton-reduce triton-reduce.cpp PARTIAL_SOURCES_INTENDED)
mlir_check_all_link_libraries(triton-reduce)

llvm_update_compile_flags(triton-reduce)
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

add_llvm_executable(triton-lsp triton-lsp.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(triton-lsp)
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


add_llvm_executable(triton-llvm-opt
  triton-llvm-opt.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  intrinsics_gen
  SUPPORT_PLUGINS
  )
target_link_libraries(triton-llvm-opt PRIVATE
  TritonLLVMIR

  LLVMAnalysis
  LLVMCore
  LLVMSupport
  LLVMOption
  LLVMCodeGen
  )
export_executable_symbols_for_plugins(triton-llvm-opt)


add_llvm_executable(triton-tensor-layout triton-tensor-layout.cpp PARTIAL_SOURCES_INTENDED)
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
