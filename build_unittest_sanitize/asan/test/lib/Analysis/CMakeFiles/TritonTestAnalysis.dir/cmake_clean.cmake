file(REMOVE_RECURSE
  "libTritonTestAnalysis.a"
  "libTritonTestAnalysis.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TritonTestAnalysis.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
