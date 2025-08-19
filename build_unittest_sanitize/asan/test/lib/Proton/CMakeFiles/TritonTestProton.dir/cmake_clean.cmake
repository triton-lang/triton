file(REMOVE_RECURSE
  "libTritonTestProton.a"
  "libTritonTestProton.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TritonTestProton.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
