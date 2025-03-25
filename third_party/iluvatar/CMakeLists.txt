add_subdirectory(include)
add_subdirectory(lib)

if(TRITON_BUILD_PYTHON_MODULE)
  if(FLAGTREE_PLUGIN)
    add_subdirectory(plugin)
    add_triton_plugin(TritonILUVATAR
      SHARED_LIB iluvatarTritonPlugin
    )
  else()
    find_library(iluvatarTritonPluginLib
      NAMES
        iluvatarTritonPlugin.so
      PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}
      REQUIRED
    )
    add_triton_plugin(TritonILUVATAR
      SHARED_LIB ${iluvatarTritonPluginLib}
    )
  endif()
endif()

add_subdirectory(bin)
