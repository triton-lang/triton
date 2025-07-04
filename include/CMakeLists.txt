add_subdirectory(triton)

if(TRITON_BUILD_PYTHON_MODULE)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton
    FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.td"
      PATTERN "*.h.inc"
  )
endif()
