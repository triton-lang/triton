file(REMOVE_RECURSE
  "CMakeFiles/TritonNvidiaGPUOpInterfacesIncGen"
  "Dialect.cpp.inc"
  "Dialect.h.inc"
  "Ops.cpp.inc"
  "Ops.h.inc"
  "OpsEnums.cpp.inc"
  "OpsEnums.h.inc"
  "TritonNvidiaGPUAttrDefs.cpp.inc"
  "TritonNvidiaGPUAttrDefs.h.inc"
  "TritonNvidiaGPUDialect.md"
  "TritonNvidiaGPUOpInterfaces.cpp.inc"
  "TritonNvidiaGPUOpInterfaces.h.inc"
  "TritonNvidiaGPUOps.md"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/TritonNvidiaGPUOpInterfacesIncGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
