set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Extensions)
add_public_tablegen_target(TritonGPUExtensionIncGen)

set(GPU_EXTENSION_PASSES
	GPUExtensionTestLib
    )

set(GPUExtensionTestLib_SOURCES
    ExtensionHello.cpp
    )

foreach( plugin ${GPU_EXTENSION_PASSES} )
    add_mlir_library(${plugin}
        SHARED
        ${${plugin}_SOURCES}

        DEPENDS
        TritonGPUExtensionIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
    include_directories(${PROJECT_BINARY_DIR}/test/lib/Extensions) # Tablegen'd files
    set_target_properties(${plugin} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                          ${PROJECT_BINARY_DIR}/test/lib/Extensions)
endforeach()
