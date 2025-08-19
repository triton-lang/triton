file(REMOVE_RECURSE
  "../../../../../../../docs/dialects/ProtonOps.md"
  "AttrDefs.cpp.inc"
  "AttrDefs.h.inc"
  "CMakeFiles/ProtonOpsDocGen"
  "Dialect.cpp.inc"
  "Dialect.h.inc"
  "Ops.cpp.inc"
  "Ops.h.inc"
  "OpsEnums.cpp.inc"
  "OpsEnums.h.inc"
  "ProtonAttrDefs.md"
  "ProtonDialect.md"
  "ProtonOps.md"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/ProtonOpsDocGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
