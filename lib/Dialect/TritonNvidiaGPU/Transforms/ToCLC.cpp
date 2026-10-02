#include <array>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUTOCLCPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

static RankedTensorType getResponseTensorType(tt::FuncOp kernel) {
  MLIRContext *ctx = kernel.getContext();
  ModuleOp module = kernel->getParentOfType<ModuleOp>();
  SmallVector<int64_t> shape{2};
  auto encoding = ttg::BlockedEncodingAttr::get(
      ctx, shape, /*sizePerThread=*/{2}, /*order=*/{0},
      ttg::lookupNumWarps(kernel),
      ttg::TritonGPUDialect::getThreadsPerWarp(module),
      ttg::lookupNumCTAs(kernel));
  return RankedTensorType::get(shape, IntegerType::get(ctx, 64), encoding);
}

static ttg::MemDescType getResponseBufferType(MLIRContext *ctx,
                                              unsigned numCTAs) {
  Attribute memorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
  auto cgaLayout = ttg::CGAEncodingAttr::fromSplitParams(
      ctx, {numCTAs}, /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto encoding =
      ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, cgaLayout);
  return ttg::MemDescType::get({2}, IntegerType::get(ctx, 64), encoding,
                               memorySpace, /*mutableMemory=*/true);
}

static LogicalResult validateKernel(tt::FuncOp kernel) {
  if (!kernel.getBody().hasOneBlock())
    return kernel.emitError(
        "CLC conversion requires one top-level kernel block");
  return success();
}

static void convertToCLC(tt::FuncOp kernel) {
  Block &body = kernel.getBody().front();
  auto returnOp = cast<tt::ReturnOp>(body.getTerminator());
  Location loc = kernel.getLoc();
  MLIRContext *ctx = kernel.getContext();

  SmallVector<Operation *> kernelOps;
  for (Operation &op : body.without_terminator())
    kernelOps.push_back(&op);

  std::array<SmallVector<tt::GetProgramIdOp>, 3> pidsByAxis;
  kernel.walk([&](tt::GetProgramIdOp pidOp) {
    pidsByAxis[pidOp.getAxisAsInt()].push_back(pidOp);
  });

  SmallVector<unsigned> usedAxes;
  for (unsigned axis = 0; axis < pidsByAxis.size(); ++axis) {
    if (!pidsByAxis[axis].empty())
      usedAxes.push_back(axis);
  }

  OpBuilder builder(returnOp);
  SmallVector<Value> initialPIDs;
  SmallVector<Type> pidTypes;
  for (unsigned axis : usedAxes) {
    Value pid = tt::GetProgramIdOp::create(builder, loc, axis);
    initialPIDs.push_back(pid);
    pidTypes.push_back(builder.getI32Type());
  }
  Value active = arith::ConstantIntOp::create(builder, loc, 1, 1);

  SmallVector<Value> inits(initialPIDs);
  inits.push_back(active);
  SmallVector<Type> stateTypes(pidTypes);
  stateTypes.push_back(builder.getI1Type());
  auto loop = scf::WhileOp::create(builder, loc, stateTypes, inits);

  Block *before = builder.createBlock(&loop.getBefore());
  for (Value init : inits)
    before->addArgument(init.getType(), loc);
  builder.setInsertionPointToEnd(before);
  scf::ConditionOp::create(builder, loc, before->getArguments().back(),
                           before->getArguments());

  Block *after = builder.createBlock(&loop.getAfter());
  for (Type type : stateTypes)
    after->addArgument(type, loc);
  builder.setInsertionPointToStart(after);
  auto responseType = getResponseTensorType(kernel);
  Value response =
      CLCTryCancelSyncOp::create(builder, loc, responseType).getResponse();
  Value responseBuffer = ttg::LocalAllocOp::create(
      builder, loc, getResponseBufferType(ctx, ttg::lookupNumCTAs(kernel)),
      response, 16);

  for (Operation *op : kernelOps)
    op->moveBefore(after, after->end());

  for (auto [axisIndex, axis] : llvm::enumerate(usedAxes)) {
    Value logicalPID = after->getArgument(axisIndex);
    for (tt::GetProgramIdOp pid : pidsByAxis[axis]) {
      pid.getResult().replaceAllUsesWith(logicalPID);
      pid.erase();
    }
  }

  builder.setInsertionPointToEnd(after);
  Value result = CLCLoadResultOp::create(builder, loc, responseBuffer);
  Value nextActive = CLCIsCanceledOp::create(builder, loc, result);

  SmallVector<Value> nextState;
  if (!usedAxes.empty()) {
    auto nextPID = scf::IfOp::create(builder, loc, pidTypes, nextActive,
                                     /*withElseRegion=*/true);
    builder.setInsertionPointToStart(nextPID.thenBlock());
    SmallVector<Value> canceledPIDs;
    for (unsigned axis : usedAxes) {
      canceledPIDs.push_back(
          CLCGetProgramIdOp::create(builder, loc, result, axis));
    }
    scf::YieldOp::create(builder, loc, canceledPIDs);

    builder.setInsertionPointToStart(nextPID.elseBlock());
    scf::YieldOp::create(builder, loc, after->getArguments().drop_back());

    builder.setInsertionPointAfter(nextPID);
    nextState.append(nextPID.getResults().begin(), nextPID.getResults().end());
  }
  nextState.push_back(nextActive);
  scf::YieldOp::create(builder, loc, nextState);
}

class TritonNvidiaGPUToCLCPass
    : public impl::TritonNvidiaGPUToCLCPassBase<TritonNvidiaGPUToCLCPass> {
public:
  using Base = impl::TritonNvidiaGPUToCLCPassBase<TritonNvidiaGPUToCLCPass>;
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (tt::FuncOp func : module.getOps<tt::FuncOp>()) {
      if (!tt::isKernel(func)) {
        auto walk = func.walk([&](tt::GetProgramIdOp pid) {
          pid.emitError(
              "CLC conversion requires all tt.get_program_id operations to be "
              "in the root function of this kernel after inlining");
          return WalkResult::interrupt();
        });
        if (walk.wasInterrupted())
          return signalPassFailure();
      } else {
        if (failed(validateKernel(func)))
          return signalPassFailure();
        convertToCLC(func);
      }
    }
  }
};

} // namespace
} // namespace mlir::triton::nvidia_gpu
