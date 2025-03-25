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

  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def TritonReorderBroadcast : Pass</*cli-arg*/"triton-reorder-broadcast", /*Op*/"mlir::ModuleOp"> {
  let summary = "Moves broadcast and splat after elementwise operations";
  let description = [{
    elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))
  }];
  let constructor = "mlir::triton::createReorderBroadcastPass()";
  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonRewriteTensorPointer : Pass</*cli-arg*/"triton-rewrite-tensor-pointer", /*Op*/"mlir::ModuleOp"> {
  let summary = "Rewrite load/stores with tensor pointers into legacy load/stores";
  let description = [{
    This pass rewrites all load/store semantics initiated by a `tt.make_tensor_ptr` and `tt.advance` into legacy
    semantics. After this pass, `tt.make_tensor_ptr` and `tt.advance` will disappear, and it generates logics to compute
    the pointer/mask/other for each load/store.
  }];

  let constructor = "mlir::triton::createRewriteTensorPointerPass()";

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

#endif
