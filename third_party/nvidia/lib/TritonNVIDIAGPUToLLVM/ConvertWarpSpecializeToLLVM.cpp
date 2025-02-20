#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTWARPSPECIALIZETOLLVM
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

static void createWarpGroupBarrier(ImplicitLocOpBuilder &b, unsigned barIdx,
                                   unsigned numThreads) {
  assert(barIdx < 16 && "not enough barriers");

  PTXBuilder ptxBuilder;
  std::string ptxString =
      ("bar.sync " + Twine(barIdx) + ", " + Twine(numThreads)).str();
  (*ptxBuilder.create<>(ptxString))();
  ptxBuilder.launch(b, b.getLoc(), void_ty(b.getContext()));
}

static LogicalResult lowerWarpSpecialize(LLVM::LLVMFuncOp func) {
  SmallVector<WarpSpecializeOp> wsOps;
  func.walk([&](WarpSpecializeOp op) { wsOps.push_back(op); });

  // Nothing to do. This kernel is not warp specialized.
  if (wsOps.empty())
    return success();

  // Function calls inside `ttg.warp_specialize` ops are not supported.
  for (WarpSpecializeOp op : wsOps) {
    auto check = [&](LLVM::CallOp op) -> WalkResult {
      return mlir::emitError(op.getLoc(),
                             "TODO: function calls inside warp specialize "
                             "partitions are not supported");
    };
    if (op.getPartitionOpHolder().walk(check).wasInterrupted())
      return failure();
  }

  unsigned threadsPerWarp =
      TritonGPUDialect::getThreadsPerWarp(cast<ModuleOp>(func->getParentOp()));
  unsigned defaultWarpGroupSize = threadsPerWarp * lookupNumWarps(func);

  // HACK: Turn all `nvvm.barrier0` ops into warp group barriers.
  func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Walk into default regions but not partition regions.
    if (isa<WarpSpecializePartitionsOp>(op))
      return WalkResult::skip();

    if (auto bar = dyn_cast<NVVM::Barrier0Op>(op)) {
      ImplicitLocOpBuilder b(bar.getLoc(), bar);
      createWarpGroupBarrier(b, 0, defaultWarpGroupSize);
      bar.erase();
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });

  // Each partition executes simultaneously, so each will get a different
  // barrier ID, but note this means there is a maximum of 16 barriers.
  for (WarpSpecializeOp op : wsOps) {
    for (auto [idx, partition] : llvm::enumerate(op.getPartitionRegions())) {
      unsigned barIdx = idx + 1;
      if (barIdx >= 16) {
        return func.emitError(
            "cannot support more than 15 warp group partitions");
      }
      unsigned warpGroupSize = threadsPerWarp * op.getPartitionNumWarps()[idx];
      partition->walk([&](NVVM::Barrier0Op bar) {
        ImplicitLocOpBuilder b(bar.getLoc(), bar);
        createWarpGroupBarrier(b, barIdx, warpGroupSize);
        bar.erase();
      });
    }
  }

  // Generate the function header.
  MLIRContext *ctx = func.getContext();
  TritonLLVMOpBuilder b(func.getLoc(), ctx);
  Block *entry = &func.getBody().front();
  Block *header = b.createBlock(entry, func.getArgumentTypes());
  Block *switchLoop = b.createBlock(entry);
  b.setInsertionPointToStart(header);

  // This is the absolute thread ID.
  Value tid = b.create<NVVM::ThreadIdXOp>(b.getIntegerType(32));
  Value isDefault = b.create<LLVM::ICmpOp>(LLVM::ICmpPredicate::ult, tid,
                                           b.i32_val(defaultWarpGroupSize));
  b.create<LLVM::CondBrOp>(isDefault, entry, switchLoop);

  // Forward arguments from the header into the old entry block.
  for (auto [arg, oldArg] :
       llvm::zip(header->getArguments(), entry->getArguments()))
    oldArg.replaceAllUsesWith(arg);
  entry->eraseArguments([](auto) { return true; });

  return success();
}

namespace {
struct ConvertWarpSpecializeToLLVM
    : public mlir::triton::impl::ConvertWarpSpecializeToLLVMBase<
          ConvertWarpSpecializeToLLVM> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<LLVM::LLVMFuncOp> kernels;
    for (auto func : mod.getOps<LLVM::LLVMFuncOp>()) {
      if (func.isPublic())
        kernels.push_back(func);
    }
    for (LLVM::LLVMFuncOp kernel : kernels)
      if (failed(lowerWarpSpecialize(kernel)))
        return signalPassFailure();
  }
};
} // namespace
