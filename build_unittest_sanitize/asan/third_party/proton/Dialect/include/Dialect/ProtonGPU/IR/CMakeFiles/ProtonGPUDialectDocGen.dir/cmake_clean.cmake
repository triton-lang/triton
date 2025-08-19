file(REMOVE_RECURSE
  "../../../../../../../docs/dialects/ProtonGPUDialect.md"
  "AttrDefs.cpp.inc"
  "AttrDefs.h.inc"
  "CMakeFiles/ProtonGPUDialectDocGen"
  "Dialect.cpp.inc"
  "Dialect.h.inc"
  "Ops.cpp.inc"
  "Ops.h.inc"
  "OpsEnums.cpp.inc"
  "OpsEnums.h.inc"
  "ProtonGPUAttrDefs.md"
  "ProtonGPUDialect.md"
  "ProtonGPUOps.md"
  "Types.cpp.inc"
  "Types.h.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/ProtonGPUDialectDocGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
