set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TritonInstrumentDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=tti)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=tti)
add_mlir_doc(TritonInstrumentDialect TritonInstrumentDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TritonInstrumentOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
add_mlir_doc(TritonInstrumentOps TritonInstrumentOps dialects/ -gen-op-doc)

add_public_tablegen_target(TritonInstrumentTableGen)
