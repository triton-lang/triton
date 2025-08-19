file(REMOVE_RECURSE
  "libTritonTestDialect.a"
  "libTritonTestDialect.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TritonTestDialect.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
