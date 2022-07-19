add_subdirectory(FileCheck)
# add_llvm_executable(FileCheck FileCheck/FileCheck.cpp)
# target_link_libraries(FileCheck PRIVATE LLVMFileCheck LLVMSupport)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(triton-opt triton-opt.cpp)

# TODO: what's this?
# llvm_update_compile_flags(triton-opt)
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