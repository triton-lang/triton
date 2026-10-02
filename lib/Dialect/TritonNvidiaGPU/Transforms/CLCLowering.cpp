#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPULOWERCLCPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

static ttg::MemDescType getResponseType(MLIRContext *ctx, unsigned numCTAs) {
  Attribute memorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
  auto cgaLayout = ttg::CGAEncodingAttr::fromSplitParams(
      ctx, {numCTAs}, /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto encoding =
      ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, cgaLayout);
  return ttg::MemDescType::get({2}, IntegerType::get(ctx, 64), encoding,
                               memorySpace, /*mutableMemory=*/true);
}

static ttg::MemDescType getBarrierType(MLIRContext *ctx, unsigned numCTAs) {
  Attribute memorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
  auto cgaLayout = ttg::CGAEncodingAttr::get1DLayout(ctx, numCTAs);
  auto encoding =
      ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, cgaLayout);
  return ttg::MemDescType::get({numCTAs}, IntegerType::get(ctx, 64), encoding,
                               memorySpace,
                               /*mutableMemory=*/true);
}

static LogicalResult lowerSite(ttng::CLCTryCancelSyncOp issue) {
  auto oldLoop = issue->getParentOfType<scf::WhileOp>();
  if (!oldLoop)
    return issue.emitOpError("expected to be nested in an scf.while");
  if (!issue.getResponse().hasOneUse())
    return issue.emitOpError(
        "expected response to have exactly one ttg.local_alloc user");
  auto markerAlloc =
      dyn_cast<ttg::LocalAllocOp>(*issue.getResponse().getUsers().begin());
  if (!markerAlloc)
    return issue.emitOpError("expected response user to be ttg.local_alloc");
  if (!markerAlloc.getResult().hasOneUse())
    return issue.emitOpError(
        "expected response marker to have exactly one ttng.clc_load_result "
        "user");
  auto markerLoad = dyn_cast<ttng::CLCLoadResultOp>(
      *markerAlloc.getResult().getUsers().begin());
  if (!markerLoad)
    return issue.emitOpError(
        "expected response marker user to be ttng.clc_load_result");
  MLIRContext *ctx = oldLoop.getContext();
  auto loc = issue.getLoc();
  auto numCTAs = ttg::lookupNumCTAs(oldLoop);
  OpBuilder builder(oldLoop);

  auto response = ttg::LocalAllocOp::create(
      builder, loc, getResponseType(ctx, numCTAs), Value(), 16);
  auto barrier =
      ttg::LocalAllocOp::create(builder, loc, getBarrierType(ctx, numCTAs));
  ttng::InitBarrierOp::create(builder, loc, barrier, 1);
  auto phaseZero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  auto one = arith::ConstantIntOp::create(builder, loc, 1, 32);
  auto pred = arith::ConstantIntOp::create(builder, loc, 1, 1);

  SmallVector<Value> newOperands{phaseZero};
  SmallVector<Type> newResultTypes{builder.getI32Type()};
  auto loop = mlir::replaceWhileOpWithNewSignature(builder, oldLoop,
                                                   newOperands, newResultTypes);
  oldLoop.erase();

  auto beforePhase = loop.getBeforeArguments().back();
  loop.getConditionOp().getArgsMutable().append(beforePhase);

  auto afterPhase = loop.getAfterArguments().back();
  auto yield = loop.getYieldOp();
  builder.setInsertionPoint(yield);
  Value nextPhase =
      arith::XOrIOp::create(builder, yield.getLoc(), afterPhase, one);
  yield.getResultsMutable().append(nextPhase);

  builder.setInsertionPoint(issue);
  auto expect = ttng::BarrierExpectOp::create(builder, loc, barrier, 16, pred);
  expect.setFromCTAAttr(builder.getI32IntegerAttr(0));
  ttng::CLCTryCancelOp::create(builder, loc, response, barrier);

  builder.setInsertionPoint(markerLoad);
  ttng::WaitBarrierOp::create(builder, markerLoad.getLoc(), barrier, afterPhase,
                              pred);
  markerLoad.getSrcMutable().set(response);

  markerAlloc.erase();
  issue.erase();

  builder.setInsertionPointAfter(loop);
  ttng::InvalBarrierOp::create(builder, loop.getLoc(), barrier);
  ttg::LocalDeallocOp::create(builder, loop.getLoc(), barrier);
  ttg::LocalDeallocOp::create(builder, loop.getLoc(), response);
  return success();
}

class TritonNvidiaGPULowerCLCPass
    : public impl::TritonNvidiaGPULowerCLCPassBase<
          TritonNvidiaGPULowerCLCPass> {
public:
  using Base =
      impl::TritonNvidiaGPULowerCLCPassBase<TritonNvidiaGPULowerCLCPass>;
  using Base::Base;

  void runOnOperation() override {
    SmallVector<ttng::CLCTryCancelSyncOp> issues;
    getOperation().walk(
        [&](ttng::CLCTryCancelSyncOp issue) { issues.push_back(issue); });
    for (ttng::CLCTryCancelSyncOp issue : issues) {
      if (ttg::lookupNumCTAs(issue) != 1) {
        issue.emitError("CLC lowering supports num_cta=1 at the moment");
        return signalPassFailure();
      }
      if (failed(lowerSite(issue)))
        return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::triton::nvidia_gpu
