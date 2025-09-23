#ifndef TRITON_TARGET_LLVMIR_PASSES
#define TRITON_TARGET_LLVMIR_PASSES

include "mlir/Pass/PassBase.td"

def LLVMDIScope: Pass<"enable-line-info", "mlir::ModuleOp"> {
  let summary = "Materialize LLVM line info";
  let description = [{
    This pass materializes line mapping information for LLVM IR dialect operations.
  }];
}

def LLVMDILocalVariable: Pass<"extract-variable-info", "mlir::ModuleOp"> {
  let summary = "Pull out source variable info from Location to DILocalVariable";
  let description = [{
    This pass pulled out source vararible's debuginfo from LLVM IR dialect's Location
      into LLVM's DILocalVariable and fused it into previous Location so it can be passed to LLVM IR later in debugging mode.
  }];
}

#endif
