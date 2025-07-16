add_subdirectory(Dialect)
add_subdirectory(TritonNVIDIAGPUToLLVM)
add_subdirectory(NVGPUToLLVM)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton
    FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.td"
      PATTERN "*.h.inc"
)
