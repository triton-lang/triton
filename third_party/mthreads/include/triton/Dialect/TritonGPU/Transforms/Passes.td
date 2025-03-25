#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonGPUCoalesce: Pass<"tritongpu-coalesce", "mlir::ModuleOp"> {
  let summary = "coalesce";

  let description = [{
    TODO
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}


def TritonGPURemoveLayoutConversions : Pass<"tritongpu-remove-layout-conversions", "mlir::ModuleOp"> {
  let summary = "remove superfluous layout conversions";

  let description = [{
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];

}

def TritonGPUOptimizeThreadLocality : Pass<"tritongpu-optimize-thread-locality", "mlir::ModuleOp"> {
  let summary = "Reduce the cost of synchronization between threads in an SM";

  let description = [{
    Today, this optimizes reduction yielded by loop to be thread-local until after the loop completes.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUReorderInstructions: Pass<"tritongpu-reorder-instructions", "mlir::ModuleOp"> {
  let summary = "Reorder instructions";

  let description = "This pass reorder instructions so as to (1) decrease register pressure (e.g., by moving "
                    "conversions from shared memory before their first use) and (2) promote LLVM instruction "
                    "order more friendly to `ptxas`.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUReduceDataDuplication: Pass<"tritongpu-reduce-data-duplication", "mlir::ModuleOp"> {
  let summary = "Reduce data duplication in register by decomposing convert[distributed -> dotOperand] "
                "into convert[distributed -> shared -> dotOperand]";

  let description = "Decomposing conversions this way makes it possible to use CSE and reuse #shared tensors";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUCombineTensorSelectAndIf: Pass<"tritongpu-combine-tensor-select-and-if", "mlir::ModuleOp"> {
  let summary = "Combine tensor select and if";

  let description = "For select instruction that uses the same condidtion as the if instruction in the same block "
                    "this pass combines the select into the if instruction, making the select operands returned by the "
                    "then/else yields.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

#endif
