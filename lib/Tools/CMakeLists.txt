add_triton_library(TritonTools
  GF2LinearAlgebra.cpp
  GenericSwizzling.cpp
  LayoutUtils.cpp
  LinearLayout.cpp
  PluginUtils.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  f2reduce
)
