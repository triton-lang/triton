set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS PluginLoweringOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
add_mlir_doc(PluginLoweringOps PluginLoweringOps PluginLowering/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS PluginLoweringDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(PluginLoweringDialect PluginLoweringDialect PluginLowering/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS PluginLoweringTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS PluginLoweringPasses.td)
mlir_tablegen(PluginLoweringPasses.h.inc --gen-pass-decls)

add_public_tablegen_target(PluginLoweringGen)
