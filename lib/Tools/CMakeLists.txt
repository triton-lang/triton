add_triton_library(TritonTools
  LinearLayout.cpp
  Utils.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  f2reduce
)
