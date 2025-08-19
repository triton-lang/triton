file(REMOVE_RECURSE
  "AttrDefs.cpp.inc"
  "AttrDefs.h.inc"
  "AttrInterfaces.cpp.inc"
  "AttrInterfaces.h.inc"
  "CMakeFiles/TritonGPUTableGen"
  "Dialect.cpp.inc"
  "Dialect.h.inc"
  "Ops.cpp.inc"
  "Ops.h.inc"
  "OpsEnums.cpp.inc"
  "OpsEnums.h.inc"
  "TritonGPUDialect.md"
  "TritonGPUOps.md"
  "TypeInterfaces.cpp.inc"
  "TypeInterfaces.h.inc"
  "Types.cpp.inc"
  "Types.h.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/TritonGPUTableGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
