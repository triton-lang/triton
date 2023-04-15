get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(triton-opt triton-opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(triton-opt)
target_link_libraries(triton-opt PRIVATE
  TritonAnalysis
  TritonTransforms
  TritonGPUTransforms
  ${dialect_libs}
  ${conversion_libs}
  # tests
  TritonTestAnalysis
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-opt)

add_llvm_executable(triton-reduce triton-reduce.cpp PARTIAL_SOURCES_INTENDED)
mlir_check_all_link_libraries(triton-reduce)

llvm_update_compile_flags(triton-reduce)
target_link_libraries(triton-reduce PRIVATE
  TritonAnalysis
  TritonTransforms
  TritonGPUTransforms
  ${dialect_libs}
  ${conversion_libs}
  # tests
  TritonTestAnalysis
  # MLIR core
  MLIRReduceLib
  MLIRPass
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-reduce)


add_llvm_executable(triton-translate triton-translate.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(triton-translate)
 target_link_libraries(triton-translate PRIVATE
         TritonAnalysis
         TritonTransforms
         TritonGPUTransforms
         TritonLLVMIR
         TritonPTX
         TritonHSACO
         ${dialect_libs}
         ${conversion_libs}
         # tests
         TritonTestAnalysis

         LLVMCore
         LLVMSupport
         LLVMOption
         LLVMCodeGen
         LLVMAsmParser

         # MLIR core
         MLIROptLib
         MLIRIR
         MLIRLLVMDialect
         MLIRPass
         MLIRSupport
         MLIRTransforms
         MLIRExecutionEngine
         MLIRMathToLLVM
         MLIRTransformUtils
         MLIRLLVMToLLVMIRTranslation
         MLIRNVVMToLLVMIRTranslation
         MLIRROCDLToLLVMIRTranslation
         )
mlir_check_all_link_libraries(triton-translate)
