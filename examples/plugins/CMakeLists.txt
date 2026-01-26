set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Plugins)
add_public_tablegen_target(TritonPluginsIncGen)

llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

set(TRITON_PLUGIN_PASSES
    TritonPluginsTestLib
    )

set(TritonPluginsTestLib_SOURCES
    TritonPlugin.cpp
    )

# Dialect Plugins
add_subdirectory(DialectPlugins)

foreach( plugin ${TRITON_PLUGIN_PASSES} )
    add_library(${plugin} SHARED ${${plugin}_SOURCES})
    add_dependencies(${plugin}
      TritonTableGen
      TritonCanonicalizeIncGen
      TritonPluginsIncGen
    )
    target_link_libraries(${plugin} PRIVATE MLIRPass)

    # CMAKE_LIBRARY_OUTPUT_DIRECTORY is only set during the Python
    # build. It is empty if building directly from the root
    # CMakeLists.txt file. Therefore if not building from Python just
    # use the default CMake shared lib path otherwise this causes a hard
    # build error
    if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
      set_target_properties(${plugin} PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY
      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../plugins")
    endif(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)

    target_compile_options(${plugin} PRIVATE -fvisibility=hidden ${TRITON_DISABLE_EH_RTTI_FLAGS})
endforeach()
