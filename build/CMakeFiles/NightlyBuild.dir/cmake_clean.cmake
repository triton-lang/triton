FILE(REMOVE_RECURSE
  "CMakeFiles/NightlyBuild"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/NightlyBuild.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
