/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/Support/ErrorHandling.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONGPUPLANCTAPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// Default CTA layouts are assigned during Triton-to-TritonGPU conversion. This
// pass only gives Dot/Reduce ops their preferred CTA layout and materializes
// the boundary with ttg.convert_layout; RemoveLayoutConversions cleans up
// later.
struct DotCTASplit {
  unsigned m;
  unsigned n;
};

ttg::DistributedEncodingTrait
cloneWithCGALayout(ttg::DistributedEncodingTrait layout,
                   llvm::ArrayRef<int64_t> shape, int numWarps,
                   int threadsPerWarp, ttg::CGAEncodingAttr newCGALayout) {
  if (auto blockedLayout = dyn_cast<ttg::BlockedEncodingAttr>(layout)) {
    return ttg::BlockedEncodingAttr::get(
        layout.getContext(), shape, blockedLayout.getSizePerThread(),
        blockedLayout.getOrder(), numWarps, threadsPerWarp, newCGALayout);
  }

  if (auto sliceLayout = dyn_cast<ttg::SliceEncodingAttr>(layout)) {
    return ttg::SliceEncodingAttr::get(
        layout.getContext(), sliceLayout.getDim(),
        cloneWithCGALayout(sliceLayout.getParent(), shape, numWarps,
                           threadsPerWarp, newCGALayout));
  }

  llvm::report_fatal_error("cloneWithCGALayout not implemented for layout");
}

Value convertValueToLayout(OpBuilder &builder, Location loc, Value value,
                           Attribute layout) {
  if (!layout)
    return value;

  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy || tensorTy.getEncoding() == layout)
    return value;

  auto newTy = tensorTy.cloneWithEncoding(layout);
  return ttg::ConvertLayoutOp::create(builder, loc, newTy, value);
}

Value cloneLoadWithLayout(OpBuilder &builder, Location loc, Value value,
                          Attribute layout, int numWarps, int threadsPerWarp) {
  Value loadValue = value;
  while (auto cvtOp = loadValue.getDefiningOp<ttg::ConvertLayoutOp>())
    loadValue = cvtOp.getSrc();

  Operation *loadOp = loadValue.getDefiningOp();
  if (!isa_and_nonnull<triton::LoadOp, triton::DescriptorLoadLikeOpInterface>(
          loadOp))
    return value;

  auto oldTy = cast<RankedTensorType>(loadValue.getType());
  auto oldLayout = cast<ttg::DistributedEncodingTrait>(oldTy.getEncoding());

  auto cgaLayout = ttg::getCGALayout(layout);
  auto newLoadLayout = cloneWithCGALayout(oldLayout, oldTy.getShape(), numWarps,
                                          threadsPerWarp, cgaLayout);
  if (oldTy.getEncoding() == newLoadLayout)
    return value;

  auto newTy = oldTy.cloneWithEncoding(newLoadLayout);

  OpBuilder loadBuilder(loadOp);
  loadBuilder.setInsertionPointAfter(loadOp);
  Operation *newLoad;
  if (auto scalarLoad = dyn_cast<triton::LoadOp>(loadOp)) {
    Value newPtr = convertValueToLayout(loadBuilder, scalarLoad.getLoc(),
                                        scalarLoad.getPtr(), newLoadLayout);
    Value newMask;
    if (Value mask = scalarLoad.getMask())
      newMask = convertValueToLayout(loadBuilder, scalarLoad.getLoc(), mask,
                                     newLoadLayout);
    Value newOther;
    if (Value other = scalarLoad.getOther())
      newOther = convertValueToLayout(loadBuilder, scalarLoad.getLoc(), other,
                                      newLoadLayout);
    newLoad = triton::LoadOp::create(
                  loadBuilder, scalarLoad.getLoc(), newTy, newPtr, newMask,
                  newOther, scalarLoad.getCache(), scalarLoad.getEvict(),
                  scalarLoad.getIsVolatile())
                  .getOperation();
    newLoad->setAttrs(loadOp->getAttrs());
  } else {
    newLoad = loadBuilder.clone(*loadOp);
    newLoad->getResult(0).setType(newTy);
  }

  return convertValueToLayout(builder, loc, newLoad->getResult(0), layout);
}

void convertOpOperandsToLayouts(Operation *op,
                                llvm::ArrayRef<Attribute> operandLayouts,
                                int numWarps, int threadsPerWarp) {
  assert(op->getNumOperands() == operandLayouts.size() &&
         "operand layout count mismatch");

  OpBuilder builder(op);
  Location loc = op->getLoc();
  for (auto [operand, layout] :
       llvm::zip(op->getOpOperands(), operandLayouts)) {
    Value value = cloneLoadWithLayout(builder, loc, operand.get(), layout,
                                      numWarps, threadsPerWarp);
    operand.set(convertValueToLayout(builder, loc, value, layout));
  }
}

void convertOpResultsFromLayouts(Operation *op,
                                 llvm::ArrayRef<Attribute> resultLayouts) {
  assert(op->getNumResults() == resultLayouts.size() &&
         "result layout count mismatch");

  OpBuilder builder(op->getContext());
  builder.setInsertionPointAfter(op);
  Location loc = op->getLoc();
  for (auto [result, resultLayout] :
       llvm::zip(op->getResults(), resultLayouts)) {
    if (!resultLayout)
      continue;

    auto originalTy = dyn_cast<RankedTensorType>(result.getType());
    if (!originalTy || originalTy.getEncoding() == resultLayout)
      continue;

    auto plannedTy = originalTy.cloneWithEncoding(resultLayout);
    result.setType(plannedTy);
    if (result.use_empty())
      continue;

    auto convert =
        ttg::ConvertLayoutOp::create(builder, loc, originalTy, result)
            .getResult();
    result.replaceAllUsesExcept(convert, convert.getDefiningOp());
  }
}

DotCTASplit getDotCTASplit(int64_t m, int64_t n, unsigned numCTAs) {
  constexpr unsigned kPreferredChunkSize = 128;
  constexpr unsigned kMinChunkSize = 64;
  auto isLegalChunkSize = [](unsigned chunk) { return chunk >= kMinChunkSize; };

  unsigned splitM = 1;
  unsigned splitN = numCTAs;
  for (unsigned chunkM = kPreferredChunkSize; isLegalChunkSize(chunkM);
       chunkM /= 2) {
    splitM = std::clamp<unsigned>(m / chunkM, 1, numCTAs);
    splitN = numCTAs / splitM;
    if (isLegalChunkSize(n / splitN))
      break;
  }

  return {splitM, splitN};
}

void planDot(triton::DotOp dot) {
  MLIRContext *ctx = dot.getContext();

  auto aTy = cast<RankedTensorType>(dot.getA().getType());
  auto bTy = cast<RankedTensorType>(dot.getB().getType());
  auto dTy = cast<RankedTensorType>(dot.getD().getType());

  auto aLayout = cast<ttg::DotOperandEncodingAttr>(aTy.getEncoding());
  auto bLayout = cast<ttg::DotOperandEncodingAttr>(bTy.getEncoding());
  auto dLayout = cast<ttg::BlockedEncodingAttr>(dTy.getEncoding());

  DotCTASplit split = getDotCTASplit(dTy.getShape()[0], dTy.getShape()[1],
                                     ttg::getNumCTAs(dLayout));

  OpBuilder builder(dot);
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  int numWarps = ttg::lookupNumWarps(dot);

  auto newCGALayout = ttg::CGAEncodingAttr::fromSplitParams(
      ctx, {split.m, split.n}, {split.m, split.n}, {1, 0});
  auto newDLayout = ttg::BlockedEncodingAttr::get(
      ctx, dTy.getShape(), dLayout.getSizePerThread(), dLayout.getOrder(),
      numWarps, threadsPerWarp, newCGALayout);
  auto newALayout = ttg::DotOperandEncodingAttr::get(
      ctx, aLayout.getOpIdx(), newDLayout, aLayout.getKWidth());
  auto newBLayout = ttg::DotOperandEncodingAttr::get(
      ctx, bLayout.getOpIdx(), newDLayout, bLayout.getKWidth());

  convertOpOperandsToLayouts(dot.getOperation(),
                             {newALayout, newBLayout, newDLayout}, numWarps,
                             threadsPerWarp);
  convertOpResultsFromLayouts(dot.getOperation(), {newDLayout});
}

ttg::CGAEncodingAttr getReduceCGALayout(triton::ReduceOp reduce,
                                        RankedTensorType srcTy) {
  unsigned rank = srcTy.getRank();
  auto order = ttg::getOrder(srcTy);
  auto sizePerThread = ttg::getContigPerThread(srcTy);
  auto srcLayout = cast<ttg::DistributedEncodingTrait>(srcTy.getEncoding());

  SmallVector<unsigned> ctasPerCGA(rank, 0);
  unsigned remainingCTAs = ttg::getNumCTAs(srcLayout);
  for (int i = rank - 1; i >= 0; --i) {
    unsigned dim = order[i];
    if (dim == reduce.getAxis()) {
      ctasPerCGA[dim] = 1;
      continue;
    }

    ctasPerCGA[dim] = std::min<unsigned>(
        srcTy.getShape()[dim] / sizePerThread[dim], remainingCTAs);
    ctasPerCGA[dim] = std::max(ctasPerCGA[dim], 1u);
    remainingCTAs /= ctasPerCGA[dim];
  }

  bool assignedRemainingCTAs = false;
  for (int i = rank - 1; i >= 0; --i) {
    unsigned dim = order[i];
    if (dim == reduce.getAxis())
      continue;
    ctasPerCGA[dim] *= remainingCTAs;
    assignedRemainingCTAs = true;
    break;
  }

  SmallVector<unsigned> ctaSplitNum = ctasPerCGA;
  if (!assignedRemainingCTAs && remainingCTAs > 0) {
    ctasPerCGA[order[rank - 1]] *= remainingCTAs;
    ctaSplitNum[order[rank - 1]] = ctasPerCGA[order[rank - 1]];
  }

  auto ctaOrder = ttg::getCTAOrder(srcLayout);
  return ttg::CGAEncodingAttr::fromSplitParams(reduce.getContext(), ctasPerCGA,
                                               ctaSplitNum, ctaOrder);
}

void planReduce(triton::ReduceOp reduce) {
  MLIRContext *ctx = reduce.getContext();
  Value src = reduce.getOperand(0);
  auto srcTy = cast<RankedTensorType>(src.getType());

  auto srcLayout = cast<ttg::DistributedEncodingTrait>(srcTy.getEncoding());

  OpBuilder builder(reduce);
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  int numWarps = ttg::lookupNumWarps(reduce);

  auto cgaLayout = getReduceCGALayout(reduce, srcTy);
  auto newSrcLayout = cloneWithCGALayout(srcLayout, srcTy.getShape(), numWarps,
                                         threadsPerWarp, cgaLayout);

  SmallVector<Attribute> operandLayouts(reduce.getNumOperands(), newSrcLayout);
  Attribute resultLayout;
  if (srcTy.getRank() > 1)
    resultLayout =
        ttg::SliceEncodingAttr::get(ctx, reduce.getAxis(), newSrcLayout);
  SmallVector<Attribute> resultLayouts(reduce.getNumResults(), resultLayout);

  convertOpOperandsToLayouts(reduce.getOperation(), operandLayouts, numWarps,
                             threadsPerWarp);
  convertOpResultsFromLayouts(reduce.getOperation(), resultLayouts);
}

struct PlanCTAPass : public impl::TritonGPUPlanCTAPassBase<PlanCTAPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    unsigned numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
    if (numCTAs == 1)
      return;

    mod.walk([&](Operation *op) {
      if (auto dot = dyn_cast<triton::DotOp>(op))
        planDot(dot);
      if (auto reduce = dyn_cast<triton::ReduceOp>(op))
        planReduce(reduce);
    });
  }
};

} // anonymous namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
