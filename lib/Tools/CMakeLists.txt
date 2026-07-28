add_triton_library(TritonTools
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

if(TRITON_BUILD_PYTHON_MODULE)
  target_link_libraries(TritonTools PUBLIC nanobind-static)
endif()
