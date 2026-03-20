file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/empty.c "")
add_library(faketriton SHARED ${CMAKE_CURRENT_BINARY_DIR}/empty.c)

set_target_properties(faketriton PROPERTIES
  OUTPUT_NAME "triton"
)

if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(faketriton PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY
  "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../plugins")
endif(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
