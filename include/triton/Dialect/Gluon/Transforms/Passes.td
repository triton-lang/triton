#ifndef GLUON_PASSES
#define GLUON_PASSES

include "mlir/Pass/PassBase.td"

def GluonResolveAutoEncodingsPass : Pass<"gluon-resolve-auto-encodings", "mlir::ModuleOp"> {
  let summary = "Resolve automatic encodings";
  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
  ];

}

def GluonCanonicalize: Pass<"gluon-canonicalize"> {
  let summary = "reduced set of simplifications for TTGIR";

  let description = [{
    The `gluon-canonicalize` pass applies a reduced set of simplification
    and canonicalization patterns to the module.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::scf::SCFDialect",
  ];
}

def GluonInline: Pass<"gluon-inline"> {
  let summary = "reduced set of simplifications for TTGIR";

  let description = [{
    The `gluon-inline` pass applies a reduced set of simplification
    and canonicalization patterns to the module.
  }];
  let dependentDialects = [];
}

#endif
