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
  # header only dependency on TritonNvidiaGPU
  TritonNvidiaGPUTableGen
  TritonNvidiaGPUAttrDefsIncGen
  TritonNvidiaGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  TritonIR
  TritonTools
)
