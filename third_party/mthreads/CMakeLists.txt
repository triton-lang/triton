add_subdirectory(include)
add_subdirectory(lib)

if(TRITON_BUILD_PYTHON_MODULE)
  if(FLAGTREE_PLUGIN)
    add_subdirectory(plugin)
    add_triton_plugin(TritonMTHREADS
      SHARED_LIB mthreadsTritonPlugin
    )
  else()
    find_library(mthreadsTritonPluginLib
      NAMES
        mthreadsTritonPlugin.so
      PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}
      REQUIRED
    )
    add_triton_plugin(TritonMTHREADS
      SHARED_LIB ${mthreadsTritonPluginLib}
    )
  endif()
endif()

add_subdirectory(bin)
