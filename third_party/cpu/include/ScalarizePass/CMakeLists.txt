set(LLVM_TARGET_DEFINITIONS ScalarizeInterface.td)
mlir_tablegen(ScalarizeInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ScalarizeInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ScalarizeInterfaceIncGen)
