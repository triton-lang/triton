#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Gluon/Transforms/InferLayoutUtils.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/CoalesceUtils.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/xxhash.h"

#define DEBUG_TYPE "gluon-infer-coalesced-encodings"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gluon {

#define GEN_PASS_DEF_GLUONINFERCOALESCEDENCODINGSPASS
#include "triton/Dialect/Gluon/Transforms/Passes.h.inc"

namespace {

ttg::CGAEncodingAttr getDefaultCGALayout(RankedTensorType refTensorType,
                                         int numCTAs) {
  // TODO support numCTAs > 1
  assert(numCTAs == 1 && "only numCTAs == 1 is supported for now");
  return ttg::CGAEncodingAttr::getDefault(refTensorType.getContext(),
                                          refTensorType.getShape().size());
}

bool isCoalescedEncodingTensorType(Type ty) {
  auto tensorTy = dyn_cast<RankedTensorType>(ty);
  return tensorTy && isa<gluon::CoalescedEncodingAttr>(tensorTy.getEncoding());
}

LogicalResult inferCoalescedLayout(ModuleOp &mod) {
  ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);

  // infer function-level coalesced layout
  for (auto &op : *mod.getBody()) {
    auto func = dyn_cast<FuncOp>(&op);
    if (!func)
      continue;

    // 1. for every load/store with coalesced encoding,
    // infer coalesced encoding for ptrs
    //
    llvm::SmallVector<std::pair<Value, Attribute>> seedEncodings;
    func.walk([&](Operation *curr) {
      Value ptr = getMemAccessPtr(curr);
      if (!ptr)
        return;
      // We only convert `tensor<tt.ptr<>>` load/store
      bool isPtrTensor = false;
      if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
        isPtrTensor = isa<PointerType>(tensorType.getElementType());
      if (!isPtrTensor)
        return;
      // we only consider those with coalesced encoding
      if (!isCoalescedEncodingTensorType(ptr.getType()))
        return;

      // build a coalesced encoding
      int numWarps = ttg::lookupNumWarps(curr);
      int numCTAs = ttg::lookupNumCTAs(curr);
      auto tensorType = cast<RankedTensorType>(ptr.getType());
      auto cgaLayout = getDefaultCGALayout(tensorType, numCTAs);
      auto shapePerCTA = ttg::getShapePerCTA(cgaLayout.getCTASplitNum(),
                                             tensorType.getShape());
      auto layout = ttg::buildCoalescedEncoding(
          mod.getContext(), axisInfoAnalysis, curr, numWarps, threadsPerWarp,
          cgaLayout, shapePerCTA);
      // set seed value
      for (auto value : curr->getOperands())
        seedEncodings.push_back({value, layout});
    });

    // 2. propagate Coalesced Layout forward/backward
    //
    // for backward slice, it doesn't cross the set_auto_layout boundary
    // i.e. gl.set_auto_layout(val, gl.CoalescedLayout())
    // -> gl.set_auto_layout(val, a concrete coalesced layout)
    // then ResolveAutoLayoutPass will handle the rest
    //
    if (failed(inferLayout(func, isCoalescedEncodingTensorType, seedEncodings)))
      return failure();
  }
  return success();
}

} // anonymous namespace

class GluonInferCoalescedEncodingsPass
    : public impl::GluonInferCoalescedEncodingsPassBase<
          GluonInferCoalescedEncodingsPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    if (failed(inferCoalescedLayout(moduleOp)))
      return signalPassFailure();

    if (failed(doubleCheckEncodings(moduleOp, isCoalescedEncodingTensorType)))
      return signalPassFailure();
  }
};
} // namespace mlir::triton::gluon
