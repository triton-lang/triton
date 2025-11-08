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

ttg::CTALayoutAttr getDefaultCTALayout(RankedTensorType refTensorType,
                                       int numCTAs) {
  // TODO support numCTAs > 1
  assert(numCTAs == 1 && "only numCTAs == 1 is supported for now");
  return ttg::CTALayoutAttr::getDefault(refTensorType.getContext(),
                                        refTensorType.getShape().size());
}

bool isCoalescedEncodingTensorType(Type ty) {
  auto tensorTy = dyn_cast<RankedTensorType>(ty);
  return tensorTy && isa<gluon::CoalescedEncodingAttr>(tensorTy.getEncoding());
}

} // anonymous namespace

class GluonInferCoalescedEncodingsPass
    : public impl::GluonInferCoalescedEncodingsPassBase<
          GluonInferCoalescedEncodingsPass> {
  //
  // triton coalesce results for reference:
  // ./build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt --tritongpu-coalesce
  // custom_bench/tt_coalesc.mlir -debug-only tritongpu-coalesce > tmp.mlir
  //
  void runOnOperation() override {
    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    // 1. for every load/store with coalesced encoding,
    // infer coalesced encoding for ptrs
    //
    // similar to Coalesce.cpp
    //
    llvm::MapVector<Operation *, Attribute> layoutMap;
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(moduleOp);
    moduleOp.walk([&](Operation *curr) {
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
      if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType())) {
        auto encoding = tensorType.getEncoding();
        if (!encoding || !isa<gluon::CoalescedEncodingAttr>(encoding))
          return;
      }

      int numWarps = ttg::lookupNumWarps(curr);
      int numCTAs = ttg::lookupNumCTAs(curr);

      auto tensorType = cast<RankedTensorType>(ptr.getType());
      auto ctaLayout = getDefaultCTALayout(tensorType, numCTAs);
      auto shapePerCTA = ttg::getShapePerCTA(ctaLayout.getCTASplitNum(),
                                             tensorType.getShape());
      ttg::setCoalescedEncoding(&getContext(), axisInfoAnalysis, curr, numWarps,
                                threadsPerWarp, ctaLayout, shapePerCTA,
                                layoutMap);
    });

    // 2. propagate forward/backward
    // similar to ResolveAutoLayoutPass.cpp
    //
    // for backward slice, it doesn't cross the set_auto_layout boundary
    // i.e. gl.set_auto_layout(val, gl.CoalescedLayout())
    // -> gl.set_auto_layout(val, concrete coalesced layout)
    // then ResolveAutoLayoutPass will handle the rest
    //
    llvm::MapVector<FuncOp, llvm::MapVector<Value, LayoutInfo>> funcValueEnc;
    llvm::MapVector<FuncOp, llvm::PriorityWorklist<Value>> funcWorklist;
    llvm::MapVector<FuncOp, llvm::MapVector<Attribute, uint64_t>> funcHashMemo;
    auto seeded = moduleOp.walk([&](Operation *op) -> WalkResult {
      if (layoutMap.find(op) == layoutMap.end())
        return WalkResult::advance();
      Attribute layout = layoutMap[op];
      FuncOp func = op->getParentOfType<FuncOp>();
      return updateEncoding(llvm::to_vector_of<Value>(op->getOperands()),
                            LayoutInfo{layout, false}, &func,
                            funcValueEnc[func], funcWorklist[func],
                            funcHashMemo[func]);
    });

    if (seeded.wasInterrupted())
      return signalPassFailure();

    // Do layout inference
    if (failed(inferLayout(moduleOp, isCoalescedEncodingTensorType,
                           funcValueEnc, funcWorklist, funcHashMemo)))
      return signalPassFailure();

    if (failed(doubleCheckEncodings(moduleOp, isCoalescedEncodingTensorType)))
      return signalPassFailure();
  }
};
} // namespace mlir::triton::gluon
