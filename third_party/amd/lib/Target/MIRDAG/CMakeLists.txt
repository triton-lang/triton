# The DAG builder lives next to the standalone code generator it belongs to.
# Triton's core library still compiles it here, against its own LLVM, for as
# long as python/src/llvm.cc keeps its MIR entry points.
add_triton_library(TritonAMDMIRDAG
  ../../../backend/codegen/DAGBuilder.cpp

  LINK_LIBS PUBLIC
  LLVMCodeGen
  LLVMTarget
)
