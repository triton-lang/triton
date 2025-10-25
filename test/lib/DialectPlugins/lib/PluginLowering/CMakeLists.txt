
add_mlir_dialect_library(MLIRPluginLowering
        PluginLoweringTypes.cpp
        PluginLoweringDialect.cpp
        PluginLoweringOps.cpp
        PluginLoweringPasses.cpp
        SHARED

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/test/lib/DialectPlugins/include/

        DEPENDS
        TritonTableGen
        TritonCanonicalizeIncGen
        MLIRPluginLoweringPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFunctionInterfaces
        MLIRFuncDialect
        MLIRPass
        LLVMSupport
        MLIRSupport
        TritonGPUIR
        TritonNvidiaGPUIR
        TritonInstrumentIR
        MLIROptLib
        MLIRPass
        MLIRRegisterAllDialects
        MLIRRegisterAllPasses
        MLIRTransforms
        "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
        )

include_directories(${PROJECT_SOURCE_DIR}/test/lib/DialectPlugins/include)
include_directories(${PROJECT_BINARY_DIR}/test/lib/DialectPlugins/include)
set_target_properties(MLIRPluginLowering PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY
                      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../dialect-plugins")
target_compile_options(MLIRPluginLowering PRIVATE -fvisibility=default)
