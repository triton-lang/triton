set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name NVWSTransforms)
add_public_tablegen_target(NVWSTransformsIncGen)
