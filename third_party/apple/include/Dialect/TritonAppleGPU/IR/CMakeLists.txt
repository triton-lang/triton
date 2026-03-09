set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TritonAppleGPUAttrDefs.td)
mlir_tablegen(TritonAppleGPUAttrDefs.h.inc -gen-attrdef-decls --attrdefs-dialect=applegpu)
mlir_tablegen(TritonAppleGPUAttrDefs.cpp.inc -gen-attrdef-defs --attrdefs-dialect=applegpu)
add_public_tablegen_target(TritonAppleGPUAttrDefsIncGen)

set(LLVM_TARGET_DEFINITIONS TritonAppleGPUDialect.td)
mlir_tablegen(Dialect.h.inc   -gen-dialect-decls -dialect=applegpu)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs  -dialect=applegpu)
add_public_tablegen_target(TritonAppleGPUDialectIncGen)
