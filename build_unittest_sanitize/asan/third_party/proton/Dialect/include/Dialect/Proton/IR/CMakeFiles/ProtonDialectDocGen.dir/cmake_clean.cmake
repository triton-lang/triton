file(REMOVE_RECURSE
  "../../../../../../../docs/dialects/ProtonDialect.md"
  "AttrDefs.cpp.inc"
  "AttrDefs.h.inc"
  "CMakeFiles/ProtonDialectDocGen"
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
  include(CMakeFiles/ProtonDialectDocGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
