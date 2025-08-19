file(REMOVE_RECURSE
  "../../../../../../docs/dialects/NVWSOps.md"
  "CMakeFiles/NVWSOpsDocGen"
  "Dialect.cpp.inc"
  "Dialect.h.inc"
  "NVWSAttrDefs.cpp.inc"
  "NVWSAttrDefs.h.inc"
  "NVWSAttrEnums.cpp.inc"
  "NVWSAttrEnums.h.inc"
  "NVWSDialect.md"
  "NVWSOpInterfaces.cpp.inc"
  "NVWSOpInterfaces.h.inc"
  "NVWSOps.md"
  "Ops.cpp.inc"
  "Ops.h.inc"
  "Types.cpp.inc"
  "Types.h.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/NVWSOpsDocGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
