#ifndef TRITONINSTRUMENT_PASSES
#define TRITONINSTRUMENT_PASSES

include "mlir/Pass/PassBase.td"

def TritonInstrumentFpSanitizer: Pass<"tritoninstrument-fp-sanitizer", "mlir::ModuleOp"> {
  let summary = "Replace floating-point ops with integer-payload equivalents";

  let description = [{
    Diagnose calls that must be inlined to expose tensor memory allocations or
    required kernel context. Then rewrite selected floating-point operations to
    use integer payload semantics. Run symbol DCE before this pass to remove
    unreachable private functions.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::math::MathDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::instrument::TritonInstrumentDialect"];

  let options = [
    Option<"homomorphicCasts", "homomorphic-casts", "bool", /*default*/"false",
           "Use truncating FPSan payload downcasts">
  ];
}

def TritonInstrumentGlobalSanitizer: Pass<"tritoninstrument-global-sanitizer", "mlir::ModuleOp"> {
  let summary = "Add gsan instrumentation to memory access operations.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::instrument::TritonInstrumentDialect"];
}

#endif // TRITON_INSTRUMENT_PASSES
