add_triton_library(f2reduce
  f2reduce.cpp
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/third_party/f2reduce
  FILES_MATCHING
    PATTERN "*.h" 
    PATTERN "*.td"
    PATTERN "*.h.inc"
)