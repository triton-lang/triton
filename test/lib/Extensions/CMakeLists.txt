set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Extensions)
add_public_tablegen_target(TritonGPUExtensionIncGen)

set(GPU_EXTENSION_PASSES
    GPUExtensionTestLib
    )

set(GPUExtensionTestLib_SOURCES
    ExtensionHello.cpp
    )

# MODULE

foreach( plugin ${GPU_EXTENSION_PASSES} )
    add_mlir_library(${plugin}
        ${${plugin}_SOURCES}
        SHARED

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_BINARY_DIR}/test/lib/Extensions

        DEPENDS
        TritonGPUExtensionIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
    set_target_properties(${plugin} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                          ${PROJECT_BINARY_DIR}/test/lib/Extensions)
    # This is set to -fvisibility=hidden in the top level CMake file
    # which causes the llvmGetPassPluginInfo symbol to be hidden and
    # an "entry point not found" error. Reset it just for this target
    if(NOT MSVC)
      target_compile_options(${plugin} PRIVATE -fvisibility=default)
    endif()
endforeach()
