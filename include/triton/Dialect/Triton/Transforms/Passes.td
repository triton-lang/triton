#ifndef TRITON_PASSES
#define TRITON_PASSES

include "mlir/Pass/PassBase.td"

def TritonCombineOps : Pass</*cli-arg*/"triton-combine", /*Op*/"mlir::ModuleOp"> {
  let summary = "combine ops";
  let description = [{
    dot(a, b, 0) + c => dot(a, b, c)

    addptr(addptr(ptr, idx0), idx1) => addptr(ptr, AddI(idx0, idx1))

    select(cond, load(ptrs, broadcast(cond), ???), other) =>
        load(ptrs, broadcast(cond), other)
  }];

  let constructor = "mlir::triton::createCombineOpsPass()";

  let dependentDialects = ["mlir::arith::ArithmeticDialect"];
}

#endif
