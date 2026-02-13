# When TRITON_BUILD_PYTHON_MODULE=ON, libtriton.so contains pybind11 binding
# code whose static constructors reference Python C API symbols (e.g.
# _Py_NoneStruct). Standalone tools must link against libpython so the dynamic
# linker can resolve these symbols at runtime.
if(TRITON_BUILD_PYTHON_MODULE AND TARGET Python3::Python)
  set(TRITON_TOOLS_PYTHON_LIBS Python3::Python)
endif()

add_executable(triton-opt triton-opt.cpp)

add_dependencies(triton-opt triton)
target_compile_options(triton-opt PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-opt PRIVATE
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  triton
  ${TRITON_TOOLS_PYTHON_LIBS}
)


add_executable(triton-reduce triton-reduce.cpp)
add_dependencies(triton-reduce triton)
target_compile_options(triton-reduce PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})

target_link_libraries(triton-reduce PRIVATE
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  # MLIR core (not in libtriton.so)
  MLIRReduceLib
  triton
  ${TRITON_TOOLS_PYTHON_LIBS}
)


add_executable(triton-lsp triton-lsp.cpp)

add_dependencies(triton-lsp triton)
target_compile_options(triton-lsp PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-lsp PRIVATE
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  # MLIR core (not in libtriton.so)
  MLIRLspServerLib
  triton
  ${TRITON_TOOLS_PYTHON_LIBS}
)



add_executable(triton-llvm-opt triton-llvm-opt.cpp)
add_dependencies(triton-llvm-opt triton intrinsics_gen)
target_compile_options(triton-llvm-opt PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-llvm-opt PRIVATE
  LLVMOption
  triton
  ${TRITON_TOOLS_PYTHON_LIBS}
  )
export_executable_symbols_for_plugins(triton-llvm-opt)


add_executable(triton-tensor-layout triton-tensor-layout.cpp)
add_dependencies(triton-tensor-layout triton)
target_compile_options(triton-tensor-layout PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(triton-tensor-layout PRIVATE
  TritonTestAnalysis
  TritonTestDialect
  TritonTestProton
  TritonAMDGPUTestAnalysis
  triton
  ${TRITON_TOOLS_PYTHON_LIBS}
  )
