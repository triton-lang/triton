set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/lib/Plugins/DialectPlugins/LoweringDialectPlugin/include)
include_directories(${PROJECT_BINARY_DIR}/lib/Plugins/DialectPlugins/LoweringDialectPlugin/include)

add_mlir_dialect(LoweringDialectPluginOps loweringdialectplugin)

set(LLVM_TARGET_DEFINITIONS LoweringDialectPluginOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
add_mlir_doc(LoweringDialectPluginOps LoweringDialectPluginOps LoweringDialectPlugin/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS LoweringDialectPluginDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(LoweringDialectPluginDialect LoweringDialectPluginDialect LoweringDialectPlugin/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS LoweringDialectPluginTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS LoweringDialectPluginPasses.td)
mlir_tablegen(LoweringDialectPluginPasses.h.inc --gen-pass-decls)

add_public_tablegen_target(MLIRLoweringDialectPluginPassesIncGen)
