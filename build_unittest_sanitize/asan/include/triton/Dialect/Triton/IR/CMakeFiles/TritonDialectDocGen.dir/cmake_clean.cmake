file(REMOVE_RECURSE
  "../../../../../docs/dialects/TritonDialect.md"
  "AttrInterfaces.cpp.inc"
  "AttrInterfaces.h.inc"
  "CMakeFiles/TritonDialectDocGen"
  "Dialect.cpp.inc"
  "Dialect.h.inc"
  "OpInterfaces.cpp.inc"
  "OpInterfaces.h.inc"
  "Ops.cpp.inc"
  "Ops.h.inc"
  "OpsEnums.cpp.inc"
  "OpsEnums.h.inc"
  "TritonDialect.md"
  "TritonOps.md"
  "Types.cpp.inc"
  "Types.h.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/TritonDialectDocGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
