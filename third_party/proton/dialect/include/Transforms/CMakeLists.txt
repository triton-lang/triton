set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Proton)
add_public_tablegen_target(ProtonTransformsIncGen)
