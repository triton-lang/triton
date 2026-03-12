if("nvidia" IN_LIST TRITON_CODEGEN_BACKENDS)
  add_subdirectory(plugins)
endif()
