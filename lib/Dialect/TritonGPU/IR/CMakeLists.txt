add_custom_target(TritonGPUIR.Interface
  DEPENDS
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen
)

add_triton_library(TritonGPUIR
  Dialect.cpp
  LinearLayoutConversions.cpp
  LayoutUtility.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
  TritonGPUIR.Interface

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  TritonIR
  TritonTools
)
