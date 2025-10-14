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

# get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
# get_property(f2reduce_libs GLOBAL PROPERTY F2REDUCE2_LIBS)

foreach( plugin ${TRITON_PLUGIN_PASSES} )
    add_mlir_library(${plugin}
        ${${plugin}_SOURCES}
        SHARED

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_BINARY_DIR}/lib

        DEPENDS
        TritonTableGen
        TritonCanonicalizeIncGen
        TritonPluginsIncGen

        LINK_LIBS PUBLIC
        # ${triton_libs}
        # ${f2reduce_libs}
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
        Python3::Module
        pybind11::headers
        "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
        )

    set_target_properties(${plugin} PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY
      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../plugins")


        target_compile_options(${plugin} PRIVATE -fvisibility=default)
endforeach()
