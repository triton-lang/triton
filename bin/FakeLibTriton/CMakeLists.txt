add_library(faketriton SHARED empty.c)

set_target_properties(faketriton PROPERTIES
  OUTPUT_NAME "triton"
)

if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(faketriton PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY
  "${CMAKE_BINARY_DIR}/bin/faketriton")
endif(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
