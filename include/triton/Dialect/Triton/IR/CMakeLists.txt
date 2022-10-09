set(LLVM_TARGET_DEFINITIONS TritonOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS TritonDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS TritonOps.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS TritonInterfaces.td)
mlir_tablegen(AttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(AttrInterfaces.cpp.inc -gen-attr-interface-defs)

add_public_tablegen_target(TritonTableGen)
