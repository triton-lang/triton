add_llvm_library(PluginLoweringPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        pluginlowering-plugin.cpp

        DEPENDS
        MLIRPluginLowering
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRPluginLowering
        )
target_include_directories(
        PluginLoweringPlugin
        PRIVATE
        ${PROJECT_SOURCE_DIR}/test/lib/DialectPlugins/include
        ${PROJECT_BINARY_DIR}/test/lib/DialectPlugins/include
)

set_target_properties(PluginLoweringPlugin PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY
                      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../dialect-plugins")
target_compile_options(PluginLoweringPlugin PRIVATE -fvisibility=default)
