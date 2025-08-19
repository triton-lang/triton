file(REMOVE_RECURSE
  "../../../../../docs/dialects/GluonOps.md"
  "CMakeFiles/GluonOpsDocGen"
  "Dialect.cpp.inc"
  "Dialect.h.inc"
  "GluonAttrDefs.cpp.inc"
  "GluonAttrDefs.h.inc"
  "GluonDialect.md"
  "GluonOps.md"
  "Ops.cpp.inc"
  "Ops.h.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/GluonOpsDocGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
