add_triton_library(TritonAMDMIRDAG
  DAGBuilder.cpp

  LINK_LIBS PUBLIC
  LLVMCodeGen
)
