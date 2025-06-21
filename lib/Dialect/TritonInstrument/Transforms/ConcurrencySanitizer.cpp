#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace instrument {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tti = mlir::triton::instrument;

#define GEN_PASS_DEF_TRITONINSTRUMENTCONCURRENCYSANITIZER
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h.inc"

namespace {

bool canAllocBeInstrumented(triton::gpu::LocalAllocOp op) {
  if (llvm::any_of(op->getUsers(),
                   [](Operation *user) { return isa<tt::CallOp>(user); })) {
    op->emitWarning("Allocation is used in a function call, cannot instrument");
    return false;
  }
  if (llvm::all_of(op->getUsers(), [](Operation *user) {
        return !isa<ttg::MemDescSubviewOp>(user);
      })) {
    return true;
  }
  if (llvm::all_of(op->getUsers(), [](Operation *user) {
        auto subview = dyn_cast<ttg::MemDescSubviewOp>(user);
        return subview && llvm::all_of(subview.getOffsets().drop_front(),
                                       [](Value offset) {
                                         return isConstantIntValue(offset, 0);
                                       });
      })) {
    return true;
  }
  op->emitWarning(
      "Allocation is used in an inconsistent way, cannot instrument");
  return false;
}

// Interpret local_allocs that are used in ttg.memdesc_subview as multibuffered
bool isMultiBuffered(triton::gpu::LocalAllocOp op) {
  return llvm::any_of(op->getUsers(), [](Operation *user) {
    return isa<ttg::MemDescSubviewOp>(user);
  });
}

uint64_t getAllocationOffset(triton::gpu::LocalAllocOp op) {
  auto offsetAttr = op->getAttr("allocation.offset");
  if (!offsetAttr) {
    llvm::report_fatal_error(
        "ConcurrencySanitizer should run after AllocateSharedMemory pass.");
  }
  return cast<IntegerAttr>(offsetAttr).getInt();
}

unsigned getNumBuffers(triton::gpu::LocalAllocOp op) {
  ttg::MemDescType ty = op.getType();
  return ty.getShape().size();
}

unsigned getSubBufferSize(triton::gpu::LocalAllocOp op) {
  ttg::MemDescType ty = op.getType();
  unsigned elSize = ty.getElementType().getIntOrFloatBitWidth();
  return product(ty.getShape().drop_front()) * elSize;
}

tt::FuncOp getEntryPoint(ModuleOp module) {
  SmallVector<tt::FuncOp> publicFuncs = llvm::to_vector(
      llvm::make_filter_range(module.getOps<tt::FuncOp>(),
                              [](tt::FuncOp func) { return func.isPublic(); }));
  assert(publicFuncs.size() == 1 && "Expected exactly one public function");
  return publicFuncs.front();
}

} // namespace

class ConcurrencySanitizerPass
    : public impl::TritonInstrumentConcurrencySanitizerBase<
          ConcurrencySanitizerPass> {
public:
  void runOnOperation() override {
    module = getOperation();
    // Collect shared memory buffers allocated in the module
    // TODO: We should actually map the region in IR + the offset in the buffer
    // to the local_alloc to give user a better error message
    llvm::SetVector<int32_t> shMemBufsSet;
    module.walk([&](triton::gpu::LocalAllocOp op) {
      if (!canAllocBeInstrumented(op)) {
        return WalkResult::advance();
      }
      int32_t baseOffset = getAllocationOffset(op);
      shMemBufsSet.insert(baseOffset);
      if (isMultiBuffered(op)) {
        unsigned numBuffers = getNumBuffers(op);
        assert(numBuffers > 0 && "Expected at least one buffer");
        unsigned subBufferSize = getSubBufferSize(op);
        for (unsigned i = 1; i < numBuffers; ++i) {
          shMemBufsSet.insert(baseOffset + i * subBufferSize);
        }
      }
      return WalkResult::advance();
    });

    tt::FuncOp entryPoint = getEntryPoint(module);
    assert(entryPoint);

    if (shMemBufsSet.empty()) {
      return;
    }

    SmallVector<int32_t> shMemBufsValues = llvm::to_vector(shMemBufsSet);
    // Pad to the next power of 2 with zeros
    uint64_t nextPowerOf2 = llvm::NextPowerOf2(shMemBufsValues.size() - 1);
    shMemBufsValues.resize(nextPowerOf2, 0);

    ImplicitLocOpBuilder b(entryPoint.getLoc(), entryPoint);
    b.setInsertionPointToStart(&entryPoint.getBody().front());
    Value shMemBufs = createSharedBufferPointers(b, shMemBufsValues);

    // Create state tensors:
    // 1. Barrier, tracking which barriers are tracking the buffer
    // 2. State, a bitfield tracking if the buffer is written (0x1) or read
    // (0x2)
    Value barriers = createConstIntTensor(b, 0, b.getIntegerType(64),
                                          shMemBufsValues.size());
    Value state =
        createConstIntTensor(b, 0, b.getIntegerType(8), shMemBufsValues.size());

    instrumentMemoryOperations(b, shMemBufs, barriers, state);
  }

private:
  void instrumentMemoryOperations(ImplicitLocOpBuilder &b, Value buffers,
                                  Value barriers, Value state) {
    module.walk([&](Operation *op) {
      if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
        b.setLoc(copyOp.getLoc());
        b.setInsertionPoint(copyOp);
        auto checkOp =
            b.create<tti::ExperimentalCheckAsyncWriteWithMbarSharedOp>(
                copyOp.getResult(), copyOp.getBarrier(), buffers, state,
                barriers);
        state = checkOp.getOutStates();
        barriers = checkOp.getOutBarriers();
      }
      if (auto waitOp = dyn_cast<ttng::WaitBarrierOp>(op)) {
        b.setLoc(waitOp.getLoc());
        b.setInsertionPoint(waitOp);
        auto checkOp = b.create<tti::ExperimentalCheckWaitMbarOp>(
            waitOp.getAlloc(), barriers, state);
        state = checkOp.getOutStates();
        barriers = checkOp.getOutBarriers();
      }
    });
  }

  ttg::BlockedEncodingAttr getBlockedEncoding(unsigned int size) {
    MLIRContext *ctx = module.getContext();
    unsigned int warps =
        mlir::cast<mlir::IntegerAttr>(module->getAttr("ttg.num-warps"))
            .getInt();
    auto ctaLayout = ttg::CTALayoutAttr::getDefault(ctx, /*rank=*/1);
    return ttg::BlockedEncodingAttr::get(ctx,
                                         /*sizePerThread=*/{size},
                                         /*threadsPerWarp=*/{32},
                                         /*warpsPerCTA=*/{warps},
                                         /*order=*/{0}, ctaLayout);
  }

  Value createSharedBufferPointers(ImplicitLocOpBuilder &builder,
                                   SmallVector<int32_t> values) {
    int64_t size = values.size();
    assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
    auto tensorType = RankedTensorType::get({size}, builder.getIntegerType(64),
                                            getBlockedEncoding(size));
    SmallVector<APInt> apInts = llvm::to_vector(
        llvm::map_range(values, [](int64_t v) { return APInt(64, v); }));
    auto denseAttr = DenseElementsAttr::get(tensorType, apInts);
    return builder.create<tti::ExperimentalSharedBufferPointersOp>(tensorType,
                                                                   values);
  }

  Value createConstIntTensor(ImplicitLocOpBuilder &builder, int val,
                             Type elType, int64_t size) {
    assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
    auto tensorType =
        RankedTensorType::get({size}, elType, getBlockedEncoding(size));
    auto denseAttr = DenseElementsAttr::get(
        tensorType, APInt(elType.getIntOrFloatBitWidth(), val));
    return builder.create<arith::ConstantOp>(tensorType, denseAttr);
  }

  ModuleOp module;
};

} // namespace instrument
} // namespace triton
} // namespace mlir
