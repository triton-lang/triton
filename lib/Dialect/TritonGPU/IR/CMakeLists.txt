add_triton_library(TritonGPUIR
  Dialect.cpp
  LinearLayoutConversions.cpp
  LayoutUtility.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  TritonIR
  TritonTools
)
