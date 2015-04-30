FILE(REMOVE_RECURSE
  "CMakeFiles/python"
  "build/timestamp"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/python.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
