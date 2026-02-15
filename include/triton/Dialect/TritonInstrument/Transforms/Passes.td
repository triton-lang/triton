#ifndef TRITONINSTRUMENT_PASSES
#define TRITONINSTRUMENT_PASSES

include "mlir/Pass/PassBase.td"

def TritonInstrumentConcurrencySanitizer: Pass<"tritoninstrument-concurrency-sanitizer", "mlir::ModuleOp"> {
  let summary = "Add runtime verification of asynchronous operations";

  let description = "Instrument the program with runtime verification of asynchronous operations.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::instrument::TritonInstrumentDialect"];
}

def TritonInstrumentFpSanitizer: Pass<"tritoninstrument-fp-sanitizer", "mlir::ModuleOp"> {
  let summary = "Replace floating-point ops with integer-payload equivalents";

  let description = "Rewrite selected floating-point operations to use integer payload semantics for fpsan.";

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::math::MathDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::instrument::TritonInstrumentDialect"];
}

#endif // TRITON_INSTRUMENT_PASSES
