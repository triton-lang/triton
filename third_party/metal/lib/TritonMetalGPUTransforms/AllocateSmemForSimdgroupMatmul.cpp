#include "TritonMetalGPUTransforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonmetal-allocate-smem-for-simdgroup-matmul"

namespace {
using namespace mlir;
std::optional<std::pair<Value, unsigned>> findTTDotInputBasePtr(Value v) {
  // trace back through convert layout
  auto convertLayoutOp = v.getDefiningOp<ttg::ConvertLayoutOp>();
  if (!convertLayoutOp)
    return std::nullopt;
  auto convertLayoutInput = convertLayoutOp.getOperand();
  auto loadOp = convertLayoutInput.getDefiningOp<tt::LoadOp>();
  if (!loadOp)
    return std::nullopt;

  auto loadPtr = loadOp.getOperand(0);
  Value forOpInitArg;

  if (auto blockArg = dyn_cast<BlockArgument>(loadPtr)) {
    // assume for now that block argument is part of a for loop
    if (auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      forOpInitArg = forOp.getInitArgs()[blockArg.getArgNumber() -
                                         forOp.getNumInductionVars()];
    } else {
      return std::nullopt;
    }
  }

  if (!forOpInitArg)
    return std::nullopt;

  auto ptr = forOpInitArg;

  // backtrack along addptr chain
  while (ptr.getDefiningOp<tt::AddPtrOp>()) {
    auto addPtrOp = ptr.getDefiningOp<tt::AddPtrOp>();
    ptr = addPtrOp->getOperand(0);
  }

  auto splatOp = ptr.getDefiningOp<tt::SplatOp>();
  if (!splatOp)
    return std::nullopt;
  auto basePtr = splatOp.getOperand();

  if (auto kernelBlockArg = dyn_cast<BlockArgument>(basePtr)) {
    return std::make_pair(basePtr, kernelBlockArg.getArgNumber());
  }

  return std::nullopt;
}
} // namespace

namespace mlir {
#define GEN_PASS_DEF_TRITONMETALGPUALLOCATESMEMFORSIMDGROUPMATMUL
#include "TritonMetalGPUTransforms/Passes.h.inc"

struct TritonMetalGPUAllocateSmemForSimdgroupMatmulPass
    : impl::TritonMetalGPUAllocateSmemForSimdgroupMatmulBase<
          TritonMetalGPUAllocateSmemForSimdgroupMatmulPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod.getContext());

    int64_t dotIdx = 0; // support multiple dot ops per function

    mod.walk([&](tt::DotOp dotOp) {
      auto *ctx = dotOp.getContext();
      auto loc = dotOp.getLoc();

      auto aTensorTy = cast<RankedTensorType>(dotOp.getA().getType());
      auto bTensorTy = cast<RankedTensorType>(dotOp.getB().getType());
      auto retType = cast<RankedTensorType>(dotOp.getResult().getType());

      // result must have blocked encoding
      if (!isa<ttg::BlockedEncodingAttr>(retType.getEncoding()))
        return;

      auto sharedMemSpace = ttg::SharedMemorySpaceAttr::get(ctx);

      StringAttr kOffset = StringAttr::get(ctx, "offset");
      StringAttr kBlock = StringAttr::get(ctx, "block");
      StringAttr dim0 = StringAttr::get(ctx, "dim0");
      StringAttr dim1 = StringAttr::get(ctx, "dim1");

      auto makeSharedTy = [&](ArrayRef<int64_t> shape, Type elemTy) {
        int64_t M = shape[0], N = shape[1];
        unsigned alignment = elemTy.getIntOrFloatBitWidth() / 8;

        // row major: offset = row * N + col
        std::vector<std::vector<int32_t>> offsetBases;
        for (int64_t i = 1; i < N; i *= 2)
          offsetBases.push_back({0, (int32_t)i});
        for (int64_t i = 1; i < M; i *= 2)
          offsetBases.push_back({(int32_t)i, 0});

        // single CTA, 0 basis vectors
        std::vector<std::vector<int32_t>> blockBases = {};

        triton::LinearLayout ll({{kOffset, offsetBases}, {kBlock, blockBases}},
                                {{dim0, (int32_t)M}, {dim1, (int32_t)N}},
                                /*requireSurjective=*/true);
        auto enc = ttg::SharedLinearEncodingAttr::get(ctx, ll, alignment);
        return ttg::MemDescType::get(shape, elemTy, enc, sharedMemSpace,
                                     /*mutableMemory=*/true);
      };

      // allocate smem right before dot op
      // previously hoisted this outside of the loop, but that overlapped with
      // convert layout scratch space and may take up too much smem
      builder.setInsertionPoint(dotOp);

      auto aAlloc = ttg::LocalAllocOp::create(
          builder, loc,
          makeSharedTy(aTensorTy.getShape(), aTensorTy.getElementType()));
      aAlloc->setAttr("metal.dot_smem", StringAttr::get(ctx, "A"));
      aAlloc->setAttr(
          "metal.dot_idx",
          IntegerAttr::get(mlir::IntegerType::get(ctx, 32), dotIdx));
      auto bAlloc = ttg::LocalAllocOp::create(
          builder, loc,
          makeSharedTy(bTensorTy.getShape(), bTensorTy.getElementType()));
      bAlloc->setAttr("metal.dot_smem", StringAttr::get(ctx, "B"));
      bAlloc->setAttr(
          "metal.dot_idx",
          IntegerAttr::get(mlir::IntegerType::get(ctx, 32), dotIdx));

      dotOp->setAttr("metal.dot_idx",
                     IntegerAttr::get(mlir::IntegerType::get(ctx, 32), dotIdx));

      // try to find which arg idx the input tensors of tt.dot correspond to
      // TODO handle the case where tensor does not come from args
      auto aResult = findTTDotInputBasePtr(dotOp.getOperand(0));
      auto bResult = findTTDotInputBasePtr(dotOp.getOperand(1));
      if (!aResult || !bResult) {
        signalPassFailure();
        return;
      }
      auto aBasePair = *aResult;
      auto bBasePair = *bResult;
      auto aBasePtr = aBasePair.first;
      auto aPtrIdx = aBasePair.second;
      auto bBasePtr = bBasePair.first;
      auto bPtrIdx = bBasePair.second;

      // the tt.dot inputs come from tt.load, which loads tensor values into
      // thread registers based on tensor encoding instead of that, want to use
      // air.simdgroup_async_copy, which uses DMA to load into threadgroup
      // memory

      // dealloc right after the dot op to bound liveness tightly
      builder.setInsertionPointAfter(dotOp);
      ttg::LocalDeallocOp::create(builder, loc, aAlloc->getResult(0));
      ttg::LocalDeallocOp::create(builder, loc, bAlloc->getResult(0));

      dotIdx++;
    });
  }
};

} // namespace mlir
