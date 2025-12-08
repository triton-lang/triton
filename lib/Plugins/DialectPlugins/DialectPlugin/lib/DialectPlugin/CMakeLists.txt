include_directories(${PROJECT_SOURCE_DIR}/lib/Plugins/DialectPlugins/DialectPlugin/include)
include_directories(${PROJECT_BINARY_DIR}/lib/Plugins/DialectPlugins/DialectPlugin/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/nvidia/include)

add_mlir_dialect_library(MLIRDialectPlugin
        DialectPluginTypes.cpp
        DialectPluginDialect.cpp
        DialectPluginOps.cpp
        DialectPluginPasses.cpp
        SHARED

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/lib/Plugins/include/

        DEPENDS
        TritonIR
        TritonTableGen
        TritonCanonicalizeIncGen
        TritonNVIDIAGPUConversionPassIncGen
        MLIRDialectPluginPassesIncGen

        LINK_LIBS PUBLIC
        MLIRPass
        LLVMSupport
        MLIRSupport
        TritonNVIDIAGPUToLLVM
        "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
        )

target_compile_options(MLIRDialectPlugin PRIVATE -fvisibility=hidden)
if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(MLIRDialectPlugin PROPERTIES
                          LIBRARY_OUTPUT_DIRECTORY
                          "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../plugins")
endif(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
