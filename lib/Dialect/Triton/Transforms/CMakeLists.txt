set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(TritonCombine.inc -gen-rewriters)
add_public_tablegen_target(TritonCombineIncGen)

# Optional Z3 dependency for IndexSimplification pass
find_library(Z3_LIBRARY NAMES z3 HINTS ENV Z3_DIR)
find_path(Z3_INCLUDE_DIR NAMES z3++.h HINTS ENV Z3_DIR)
if(Z3_LIBRARY AND Z3_INCLUDE_DIR)
  message(STATUS "Triton: Z3 found: ${Z3_LIBRARY} — enabling Z3-backed index simplification")
  set(TRITON_HAVE_Z3 TRUE)
else()
  message(WARNING "Triton: Z3 not found — IndexSimplification pass will run algebraic-only mode")
  set(TRITON_HAVE_Z3 FALSE)
endif()

add_triton_library(TritonTransforms
  Combine.cpp
  IndexSimplification.cpp
  LoopAwareCSE.cpp
  LoopInvariantCodeMotion.cpp
  LoopPeeling.cpp
  LoopUnroll.cpp
  ReorderBroadcast.cpp
  RewriteTensorDescriptorToPointer.cpp
  ArithTypeConversion.cpp
  FunctionTypeConversion.cpp

  DEPENDS
  TritonTransformsIncGen
  TritonCombineIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  MLIRTransforms
  MLIRSCFToControlFlow
  TritonIR
)

if(TRITON_HAVE_Z3)
  target_compile_definitions(TritonTransforms PRIVATE TRITON_HAVE_Z3)
  target_include_directories(TritonTransforms PRIVATE ${Z3_INCLUDE_DIR})
  target_link_libraries(TritonTransforms PRIVATE ${Z3_LIBRARY})
endif()
