include_directories(${PROJECT_SOURCE_DIR}/lib/Plugins/DialectPlugins/LoweringDialectPlugin/include)
include_directories(${PROJECT_BINARY_DIR}/lib/Plugins/DialectPlugins/LoweringDialectPlugin/include)

add_mlir_dialect_library(MLIRLoweringDialectPlugin
        LoweringDialectPluginTypes.cpp
        LoweringDialectPluginDialect.cpp
        LoweringDialectPluginOps.cpp
        LoweringDialectPluginPasses.cpp
        SHARED

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/lib/Plugins/include/

        DEPENDS
        TritonIR
        TritonTableGen
        TritonCanonicalizeIncGen
        MLIRLoweringDialectPluginPassesIncGen

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
        MLIRRegisterAllDialects
        MLIRRegisterAllPasses
        MLIRTransforms
        "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
        )

target_compile_options(MLIRLoweringDialectPlugin PRIVATE -fvisibility=default)
if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(MLIRLoweringDialectPlugin PROPERTIES
                          LIBRARY_OUTPUT_DIRECTORY
                          "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../plugins")
endif(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
