include_directories(${PROJECT_SOURCE_DIR}/Plugins/DialectPlugins/DialectPlugin/include)
include_directories(${PROJECT_BINARY_DIR}/Plugins/DialectPlugins/DialectPlugin/include)

# This is so add_mlir_doc doesn't hit "error: mkdir(/docs): Permission denied"
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

add_mlir_dialect(DialectPluginOps plugin)

set(LLVM_TARGET_DEFINITIONS DialectPluginOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
add_mlir_doc(DialectPluginOps DialectPluginOps DialectPlugin/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS DialectPluginDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=plugin)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=plugin)
add_mlir_doc(DialectPluginDialect DialectPluginDialect DialectPlugin/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS DialectPluginTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS DialectPluginPasses.td)
mlir_tablegen(DialectPluginPasses.h.inc --gen-pass-decls -name=plugin)
add_public_tablegen_target(MLIRDialectPluginPassesIncGen)
