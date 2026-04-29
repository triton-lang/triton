#include "Dialect/TritonMetalGPU/IR/Dialect.h"
#include "TritonMetalGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttmetalgpu = mlir::triton::metalgpu;

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonmetal-allocate-smem-for-simdgroup-matmul"

namespace mlir {
namespace {
std::pair<Value, tt::LoadOp> findTTDotTileBasePtr(Value v) {
  // trace back through convert layout
  auto convertLayoutOp = v.getDefiningOp<ttg::ConvertLayoutOp>();
  if (!convertLayoutOp)
    return {};
  auto convertLayoutInput = convertLayoutOp.getOperand();
  auto loadOp = convertLayoutInput.getDefiningOp<tt::LoadOp>();
  if (!loadOp)
    return {};

  auto loadPtr = loadOp.getOperand(0);
  if (!isa<RankedTensorType>(loadPtr.getType()))
    return {};
  return std::make_pair(loadPtr, loadOp);
}

std::optional<unsigned> findTTDotInputArg(tt::LoadOp loadOp) {
  auto blockArg = dyn_cast<BlockArgument>(loadOp.getPtr());
  if (!blockArg)
    return std::nullopt;

  auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!forOp)
    return std::nullopt;

  unsigned iterArgIdx = blockArg.getArgNumber() - forOp.getNumInductionVars();
  Value initPtr = forOp.getInitArgs()[iterArgIdx];

  Value cur = initPtr;
  while (auto addptr = cur.getDefiningOp<tt::AddPtrOp>())
    cur = addptr.getPtr();

  auto splatOp = cur.getDefiningOp<tt::SplatOp>();
  if (!splatOp)
    return std::nullopt;

  auto ptrArg = dyn_cast<BlockArgument>(splatOp.getSrc());
  if (!ptrArg)
    return std::nullopt;

  return ptrArg.getArgNumber();
}

Value findTTDotStoreBasePtr(tt::DotOp dotOp) {
  auto accumulator = dotOp.getResult();
  auto accUsers = accumulator.getUsers();
  scf::YieldOp yieldOp;
  for (auto user : accUsers) {
    yieldOp = dyn_cast<scf::YieldOp>(user);
    if (!yieldOp)
      return {};
  }

  auto forOp = dotOp->getParentOfType<scf::ForOp>();
  if (!forOp)
    return {};

  Value forOpResult;
  for (auto [yieldOpIdx, operand] : llvm::enumerate(yieldOp->getOperands())) {
    if (operand == accumulator) {
      forOpResult = forOp.getResult(yieldOpIdx);
    }
  }

  if (!forOpResult)
    return {};

  Operation *dotResultOp = forOp.getOperation();
  auto dotResult = forOpResult;
  while (!isa<tt::StoreOp>(*dotResultOp)) {
    auto dotResultUsers = dotResult.getUsers();
    Operation *nextOp = nullptr;
    for (auto dotResultUser : dotResultUsers) {
      nextOp = dotResultUser;
      break;
    }
    if (!nextOp)
      return {};
    dotResultOp = nextOp;
    if (!isa<tt::StoreOp>(*dotResultOp)) {
      if (dotResultOp->getNumResults() > 0)
        dotResult = dotResultOp->getResult(0);
      else
        return {};
    }
  }

  auto storeOp = dyn_cast<tt::StoreOp>(dotResultOp);
  if (!storeOp)
    return {};

  auto cBasePtr = storeOp.getOperand(0);
  if (!isa<RankedTensorType>(cBasePtr.getType()))
    return {};

  return cBasePtr;
}

struct DotOpToSimdgroupMMA : public OpRewritePattern<tt::DotOp> {
  using OpRewritePattern<tt::DotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    int64_t dotIdx = 0; // support multiple dot ops per function

    auto *ctx = dotOp.getContext();
    auto loc = dotOp.getLoc();

    auto aTensorTy = cast<RankedTensorType>(dotOp.getA().getType());
    auto bTensorTy = cast<RankedTensorType>(dotOp.getB().getType());
    auto retType = cast<RankedTensorType>(dotOp.getResult().getType());

    // result must have blocked encoding
    if (!isa<ttg::BlockedEncodingAttr>(retType.getEncoding()))
      return rewriter.notifyMatchFailure(dotOp,
                                         "Result must have blocked encoding");

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
    rewriter.setInsertionPoint(dotOp);

    auto aAlloc = ttg::LocalAllocOp::create(
        rewriter, loc,
        makeSharedTy(aTensorTy.getShape(), aTensorTy.getElementType()));
    aAlloc->setAttr("metal.dot_smem", StringAttr::get(ctx, "A"));
    aAlloc->setAttr("metal.dot_idx",
                    IntegerAttr::get(mlir::IntegerType::get(ctx, 32), dotIdx));
    auto bAlloc = ttg::LocalAllocOp::create(
        rewriter, loc,
        makeSharedTy(bTensorTy.getShape(), bTensorTy.getElementType()));
    bAlloc->setAttr("metal.dot_smem", StringAttr::get(ctx, "B"));
    bAlloc->setAttr("metal.dot_idx",
                    IntegerAttr::get(mlir::IntegerType::get(ctx, 32), dotIdx));

    dotOp->setAttr("metal.dot_idx",
                   IntegerAttr::get(mlir::IntegerType::get(ctx, 32), dotIdx));

    // dealloc right after the dot op to bound liveness tightly
    rewriter.setInsertionPointAfter(dotOp);
    ttg::LocalDeallocOp::create(rewriter, loc, aAlloc->getResult(0));
    ttg::LocalDeallocOp::create(rewriter, loc, bAlloc->getResult(0));

    dotIdx++;

    // try to find base ptr for the tile args of tt.dot
    // TODO handle the case where tensor does not come from tt.load
    auto aBasePtrResult = findTTDotTileBasePtr(dotOp.getOperand(0));
    auto aTilePtr = aBasePtrResult.first;
    auto aLoadOp = aBasePtrResult.second;
    auto aArgIdx = findTTDotInputArg(aLoadOp);
    auto bBasePtrResult = findTTDotTileBasePtr(dotOp.getOperand(1));
    auto bTilePtr = bBasePtrResult.first;
    auto bLoadOp = bBasePtrResult.second;
    auto bArgIdx = findTTDotInputArg(bLoadOp);

    if (!aTilePtr || !bTilePtr)
      return rewriter.notifyMatchFailure(dotOp,
                                         "Unable to find input tile ptr");
    if (!aArgIdx || !bArgIdx)
      return rewriter.notifyMatchFailure(dotOp,
                                         "Unable to find input tensor arg idx");

    // stride ptr is arg right after the data ptr
    // InjectTensorStrideArgs puts stride at dataArgIdx + 1
    auto funcOp = dotOp->getParentOfType<tt::FuncOp>();
    Value aStridePtr = funcOp.getArgument(*aArgIdx + 1);
    Value bStridePtr = funcOp.getArgument(*bArgIdx + 1);

    // load stride[0] then truncate to i32 for SimdgroupAsyncCopyOp
    auto i64Ty = rewriter.getI64Type();
    auto i32Ty = rewriter.getI32Type();

    // insert async copies after the allocs
    rewriter.setInsertionPointAfter(aAlloc);
    Value aStrideI64 =
        tt::LoadOp::create(rewriter, loc, i64Ty, aStridePtr, /*mask=*/Value{},
                           /*other=*/Value{}, /*cache=*/tt::CacheModifier::NONE,
                           /*evict=*/tt::EvictionPolicy::NORMAL,
                           /*isVolatile=*/false);
    Value aStrideI32 =
        arith::TruncIOp::create(rewriter, loc, i32Ty, aStrideI64);
    ttmetalgpu::SimdgroupAsyncCopyOp::create(rewriter, loc, aTilePtr,
                                             aStrideI32, aAlloc.getResult());

    rewriter.setInsertionPointAfter(bAlloc);
    Value bStrideI64 =
        tt::LoadOp::create(rewriter, loc, i64Ty, bStridePtr, /*mask=*/Value{},
                           /*other=*/Value{}, /*cache=*/tt::CacheModifier::NONE,
                           /*evict=*/tt::EvictionPolicy::NORMAL,
                           /*isVolatile=*/false);
    Value bStrideI32 =
        arith::TruncIOp::create(rewriter, loc, i32Ty, bStrideI64);
    ttmetalgpu::SimdgroupAsyncCopyOp::create(rewriter, loc, bTilePtr,
                                             bStrideI32, bAlloc.getResult());

    // replace dot op with custom mma op
    rewriter.setInsertionPoint(dotOp);
    Value mmaResult = ttmetalgpu::SimdgroupMMAOp::create(
        rewriter, loc, retType, aAlloc.getResult(), bAlloc.getResult(),
        dotOp.getC(), aStrideI32, bStrideI32);

    // try to find base ptr for where output tile is stored
    auto cTilePtr = findTTDotStoreBasePtr(dotOp);

    // tt.dot inputs come from tt.load, which loads tensor values into
    // thread registers based on tensor encoding.
    // Instead, use air.simdgroup_async_copy to take advantage
    // of DMA.
    // Need to trace back from tt.dot inputs to the load ops and replace them
    // with

    rewriter.replaceOp(dotOp, mmaResult);

    return llvm::success();
  }
};

} // namespace

#define GEN_PASS_DEF_TRITONMETALGPUALLOCATESMEMFORSIMDGROUPMATMUL
#include "TritonMetalGPUTransforms/Passes.h.inc"

class TritonMetalGPUAllocateSmemForSimdgroupMatmulPass
    : public impl::TritonMetalGPUAllocateSmemForSimdgroupMatmulBase<
          TritonMetalGPUAllocateSmemForSimdgroupMatmulPass> {
public:
  using impl::TritonMetalGPUAllocateSmemForSimdgroupMatmulBase<
      TritonMetalGPUAllocateSmemForSimdgroupMatmulPass>::
      TritonMetalGPUAllocateSmemForSimdgroupMatmulBase;

  void runOnOperation() override {
    llvm::errs() << "Running pass" << "\n";
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);

    patterns.add<DotOpToSimdgroupMMA>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      llvm::errs() << "Pass failed" << "\n";
      return signalPassFailure();
    }
  }
};

} // namespace mlir
