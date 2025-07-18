set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS GluonOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(GluonOps GluonOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS GluonDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=gluon)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=gluon)
add_mlir_doc(GluonDialect GluonDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS GluonAttrDefs.td)
mlir_tablegen(GluonAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(GluonAttrDefs.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(GluonTableGen)
