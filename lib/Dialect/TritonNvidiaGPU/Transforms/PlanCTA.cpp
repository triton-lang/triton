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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
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
// pass only gives Dot/Reduce ops their preferred CTA layout and materializes the
// boundary with ttg.convert_layout; RemoveLayoutConversions cleans up later.
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

void convertOpOperandsToLayouts(Operation *op,
                                llvm::ArrayRef<Attribute> operandLayouts) {
  assert(op->getNumOperands() == operandLayouts.size() &&
         "operand layout count mismatch");

  OpBuilder builder(op);
  Location loc = op->getLoc();
  for (auto [operand, layout] : llvm::zip(op->getOpOperands(), operandLayouts))
    operand.set(convertValueToLayout(builder, loc, operand.get(), layout));
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

    auto convert = ttg::ConvertLayoutOp::create(builder, loc, originalTy, result)
                       .getResult();
    result.replaceAllUsesExcept(convert, convert.getDefiningOp());
  }
}

DotCTASplit getDotCTASplit(int64_t m, int64_t n, unsigned numCTAs) {
  constexpr unsigned kPreferredChunkSize = 128;
  constexpr unsigned kMinChunkSize = 64;
  auto isLegalChunkSize = [](unsigned chunk) {
    return chunk >= kMinChunkSize;
  };

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

  auto aTy = dyn_cast<RankedTensorType>(dot.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(dot.getB().getType());
  auto dTy = dyn_cast<RankedTensorType>(dot.getD().getType());
  if (!aTy || !bTy || !dTy)
    return;

  auto aLayout = dyn_cast<ttg::DotOperandEncodingAttr>(aTy.getEncoding());
  auto bLayout = dyn_cast<ttg::DotOperandEncodingAttr>(bTy.getEncoding());
  auto dLayout = dyn_cast<ttg::BlockedEncodingAttr>(dTy.getEncoding());
  if (!aLayout || !bLayout || !dLayout)
    return;

  DotCTASplit split =
      getDotCTASplit(dTy.getShape()[0], dTy.getShape()[1],
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
                             {newALayout, newBLayout, newDLayout});
  convertOpResultsFromLayouts(dot.getOperation(), {newDLayout});
}

ttg::CGAEncodingAttr getReduceCGALayout(triton::ReduceOp reduce,
                                        RankedTensorType srcTy,
                                        unsigned numCTAs) {
  unsigned rank = srcTy.getRank();
  auto order = ttg::getOrder(srcTy);
  auto sizePerThread = ttg::getContigPerThread(srcTy);

  SmallVector<unsigned> ctasPerCGA(rank, 0);
  unsigned remainingCTAs = numCTAs;
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

  auto srcLayout = cast<ttg::DistributedEncodingTrait>(srcTy.getEncoding());
  auto ctaOrder = ttg::getCTAOrder(srcLayout);
  return ttg::CGAEncodingAttr::fromSplitParams(
      reduce.getContext(), ctasPerCGA, ctaSplitNum, ctaOrder);
}

void planReduce(triton::ReduceOp reduce, unsigned numCTAs) {
  if (reduce.getNumOperands() == 0)
    return;

  MLIRContext *ctx = reduce.getContext();
  Value src = reduce.getOperand(0);
  auto srcTy = dyn_cast<RankedTensorType>(src.getType());
  if (!srcTy || srcTy.getRank() == 0)
    return;

  auto srcLayout = dyn_cast<ttg::DistributedEncodingTrait>(srcTy.getEncoding());
  if (!srcLayout)
    return;

  OpBuilder builder(reduce);
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  int numWarps = ttg::lookupNumWarps(reduce);

  auto cgaLayout = getReduceCGALayout(reduce, srcTy, numCTAs);
  auto newSrcLayout = cloneWithCGALayout(srcLayout, srcTy.getShape(), numWarps,
                                         threadsPerWarp, cgaLayout);

  SmallVector<Attribute> operandLayouts(reduce.getNumOperands(), newSrcLayout);
  Attribute resultLayout;
  if (srcTy.getRank() > 1)
    resultLayout = ttg::SliceEncodingAttr::get(ctx, reduce.getAxis(),
                                               newSrcLayout);
  SmallVector<Attribute> resultLayouts(reduce.getNumResults(), resultLayout);

  convertOpOperandsToLayouts(reduce.getOperation(), operandLayouts);
  convertOpResultsFromLayouts(reduce.getOperation(), resultLayouts);
}

struct PlanCTAPass : public impl::TritonGPUPlanCTAPassBase<PlanCTAPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    unsigned numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
    if (numCTAs == 1)
      return;

    mod.walk([](triton::DotOp dot) { planDot(dot); });
    mod.walk([&](triton::ReduceOp reduce) { planReduce(reduce, numCTAs); });
  }
};

} // anonymous namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
