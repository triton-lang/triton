#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
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

bool isBarrier(triton::gpu::LocalAllocOp op) {
  // Is there InitBarrierOp in the forward slice of the op?
  bool foundInitBarrier = false;
  SetVector<Operation *> forwardSlice;
  ForwardSliceOptions options;
  options.filter = [&](Operation *op) {
    if (isa<ttng::InitBarrierOp>(op)) {
      foundInitBarrier = true;
      return false;
    }
    return true;
  };
  getForwardSlice(op.getOperation(), &forwardSlice, options);
  return foundInitBarrier;
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
  return ty.getShape()[0];
}

unsigned getSubBufferSize(triton::gpu::LocalAllocOp op) {
  ttg::MemDescType ty = op.getType();
  unsigned elSize = ty.getElementType().getIntOrFloatBitWidth() / 8;
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
    llvm::SetVector<int32_t> barrierSet;
    module.walk([&](triton::gpu::LocalAllocOp op) {
      if (!canAllocBeInstrumented(op)) {
        return WalkResult::advance();
      }
      int32_t baseOffset = getAllocationOffset(op);
      auto &setToAdd = isBarrier(op) ? barrierSet : shMemBufsSet;
      setToAdd.insert(baseOffset);
      if (isMultiBuffered(op)) {
        unsigned numBuffers = getNumBuffers(op);
        assert(numBuffers > 0 && "Expected at least one buffer");
        unsigned subBufferSize = getSubBufferSize(op);
        for (unsigned i = 1; i < numBuffers; ++i) {
          setToAdd.insert(baseOffset + i * subBufferSize);
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
    SmallVector<int32_t> barrierValues = llvm::to_vector(barrierSet);
    // Pad to the next power of 2 with zeros
    shMemBufsValues.resize(llvm::NextPowerOf2(shMemBufsValues.size() - 1), 0);
    if (!barrierValues.empty()) {
      barrierValues.resize(llvm::NextPowerOf2(barrierValues.size() - 1), 0);
    }

    ImplicitLocOpBuilder b(entryPoint.getLoc(), entryPoint);
    b.setInsertionPointToStart(&entryPoint.getBody().front());
    shBuffers = createSharedBufferPointers(b, shMemBufsValues);

    if (!barrierValues.empty()) {
      barriers = createSharedBufferPointers(b, barrierValues);

      // Create state tensors:
      writeBarriersType = RankedTensorType::get(
          {(long)shMemBufsValues.size()}, b.getIntegerType(64),
          getThreadLocalBlockedEncoding(shMemBufsValues.size()));
      TypedValue<RankedTensorType> writeBarriers =
          tti::createConstIntTensor(b, b.getLoc(), 0, writeBarriersType);
      writeBarriersAlloc = createInitializedScratchMemory(b, writeBarriers);

      readBarriersType = RankedTensorType::get(
          {(long)shMemBufsValues.size(), (long)barrierValues.size()},
          b.getIntegerType(8),
          getReadBarriersEncoding(shMemBufsValues.size(),
                                  barrierValues.size()));
      TypedValue<RankedTensorType> readBarriers =
          tti::createConstIntTensor(b, b.getLoc(), 0, readBarriersType);
      readBarriersAlloc = createInitializedScratchMemory(b, readBarriers);
    }

    // Create write commits tensor
    writeCommitsType = RankedTensorType::get(
        {(long)shMemBufsValues.size()}, b.getIntegerType(8),
        getThreadLocalBlockedEncoding(shMemBufsValues.size()));
    TypedValue<RankedTensorType> writeCommits =
        tti::createConstIntTensor(b, b.getLoc(), 0, writeCommitsType);
    writeCommitsAlloc = createInitializedScratchMemory(b, writeCommits);

    instrumentMemoryOperations(b);
  }

private:
  void addWriteChecks(ImplicitLocOpBuilder &b, Value buf, Value pred) {
    if (barriers) {
      b.create<tti::ExperimentalCheckOutstandingWritesOp>(
          buf, shBuffers, writeBarriersAlloc, writeBarriersType, pred);
    }
    b.create<tti::ExperimentalCheckWriteCommitOp>(
        buf, shBuffers, writeCommitsAlloc, writeCommitsType, pred);
  }

  void addReadChecks(ImplicitLocOpBuilder &b, Value buf, Value pred) {
    if (barriers) {
      b.create<tti::ExperimentalCheckOutstandingReadsOp>(
          buf, shBuffers, readBarriersAlloc, readBarriersType, pred);
    }
  }

  void instrumentMemoryOperations(ImplicitLocOpBuilder &b) {
    module.walk([&](Operation *op) {
      b.setLoc(op->getLoc());
      b.setInsertionPoint(op);
      if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
        auto buf = copyOp.getResult();
        auto pred = copyOp.getPred();
        auto barrier = copyOp.getBarrier();
        assert(barriers);
        addWriteChecks(b, buf, pred);
        addReadChecks(b, buf, pred);
        b.create<tti::ExperimentalMarkAsWriteOp>(buf, barrier, shBuffers,
                                                 writeBarriersAlloc,
                                                 writeBarriersType, pred);
      }
      if (auto mmav5Op = dyn_cast<ttng::TCGen5MMAOp>(op)) {
        auto pred = mmav5Op.getPred();
        b.setInsertionPoint(mmav5Op);
        if (isa<ttg::NVMMASharedEncodingAttr>(
                mmav5Op.getA().getType().getEncoding())) {
          addWriteChecks(b, mmav5Op.getA(), pred);
          for (auto barrier : mmav5Op.getBarriers()) {
            assert(barriers);
            b.create<tti::ExperimentalMarkAsReadOp>(
                mmav5Op.getA(), barrier, shBuffers, barriers, readBarriersAlloc,
                readBarriersType, pred);
          }
        }
        if (isa<ttg::NVMMASharedEncodingAttr>(
                mmav5Op.getB().getType().getEncoding())) {
          addWriteChecks(b, mmav5Op.getB(), pred);
          for (auto barrier : mmav5Op.getBarriers()) {
            assert(barriers);
            b.create<tti::ExperimentalMarkAsReadOp>(
                mmav5Op.getB(), barrier, shBuffers, barriers, readBarriersAlloc,
                readBarriersType, pred);
          }
        }
      }
      if (auto waitOp = dyn_cast<ttng::WaitBarrierOp>(op)) {
        assert(barriers);
        auto pred = waitOp.getPred();
        auto barrier = waitOp.getAlloc();
        b.create<tti::ExperimentalClearWriteBarrierOp>(
            barrier, writeBarriersAlloc, writeBarriersType, pred);
        b.create<tti::ExperimentalClearReadBarrierOp>(
            barrier, barriers, readBarriersAlloc, readBarriersType, pred);
      }
      if (auto asyncCopyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
        addWriteChecks(b, asyncCopyOp.getResult(), nullptr);
        addReadChecks(b, asyncCopyOp.getResult(), nullptr);
        b.create<tti::ExperimentalStageWriteForCommitOp>(
            asyncCopyOp.getResult(), shBuffers, writeCommitsAlloc,
            writeCommitsType, nullptr);
      }
      if (auto asyncCommitGroupOp = dyn_cast<ttg::AsyncCommitGroupOp>(op)) {
        b.create<tti::ExperimentalCommitWritesOp>(writeCommitsAlloc,
                                                  writeCommitsType, nullptr);
      }
      if (auto asyncWaitOp = dyn_cast<ttg::AsyncWaitOp>(op)) {
        b.create<tti::ExperimentalClearWriteCommitsOp>(
            writeCommitsAlloc, writeCommitsType, asyncWaitOp.getNum(), nullptr);
      }
    });
  }

  ttg::BlockedEncodingAttr getThreadLocalBlockedEncoding(unsigned int size) {
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

  ttg::BlockedEncodingAttr getReadBarriersEncoding(unsigned int buffers,
                                                   unsigned int barriers) {
    MLIRContext *ctx = module.getContext();
    unsigned int warps =
        mlir::cast<mlir::IntegerAttr>(module->getAttr("ttg.num-warps"))
            .getInt();
    auto ctaLayout = ttg::CTALayoutAttr::getDefault(ctx, /*rank=*/2);
    return ttg::BlockedEncodingAttr::get(ctx,
                                         /*sizePerThread=*/{buffers, barriers},
                                         /*threadsPerWarp=*/{1, 32},
                                         /*warpsPerCTA=*/{1, warps},
                                         /*order=*/{0, 1}, ctaLayout);
  }

  Value createSharedBufferPointers(ImplicitLocOpBuilder &builder,
                                   SmallVector<int32_t> values) {
    int64_t size = values.size();
    assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
    Type elType = builder.getI64Type();
    auto tensorType = RankedTensorType::get(
        {size}, elType, getThreadLocalBlockedEncoding(size));
    SmallVector<APInt> apInts = llvm::to_vector(
        llvm::map_range(values, [](int64_t v) { return APInt(64, v); }));
    auto denseAttr = DenseElementsAttr::get(tensorType, apInts);
    auto op = builder.create<tti::ExperimentalSharedBufferPointersOp>(
        tensorType, values);
    return op;
  }

  ttg::DistributedEncodingTrait
  getSingleDimSliceEncoding(ttg::BlockedEncodingAttr encoding, int dim) {
    int rank = encoding.getOrder().size();
    MLIRContext *ctx = encoding.getContext();
    assert(dim < rank && "Expected dim to be less than rank");
    ttg::DistributedEncodingTrait sliceEncoding = encoding;
    for (int i = 0; i < rank; ++i) {
      if (i != dim) {
        sliceEncoding = ttg::SliceEncodingAttr::get(ctx, i, encoding);
      }
    }
    return sliceEncoding;
  }

  Value expandAllSlicedDims(ImplicitLocOpBuilder &b, Value tensor) {
    auto type = cast<RankedTensorType>(tensor.getType());
    auto sliceEncoding = dyn_cast<ttg::SliceEncodingAttr>(type.getEncoding());
    while (sliceEncoding) {
      int dim = sliceEncoding.getDim();
      auto shape = type.getShape();
      auto newShape = SmallVector<int64_t>(shape);
      newShape.insert(newShape.begin() + dim, 1);
      auto newType = RankedTensorType::get(newShape, type.getElementType(),
                                           sliceEncoding.getParent());
      tensor = b.create<tt::ExpandDimsOp>(newType, tensor, dim);
      type = newType;
      sliceEncoding = dyn_cast<ttg::SliceEncodingAttr>(type.getEncoding());
    }
    return tensor;
  }

  Value createInitializedScratchMemory(ImplicitLocOpBuilder &b,
                                       TypedValue<RankedTensorType> tensor) {
    auto encoding = tensor.getType().getEncoding();
    Type elType = tensor.getType().getElementType();
    int elSize = elType.getIntOrFloatBitWidth() / 8;
    int numEls = product(tensor.getType().getShape());
    int64_t sizeInBytes = numEls * elSize;
    Type ptrType = triton::getPointerType(elType);
    auto alloc =
        b.create<tt::gpu::GlobalScratchAllocOp>(ptrType, sizeInBytes, elSize);
    createStoreScratchMemory(b, b.getLoc(), alloc, tensor, tensor.getType());
    return alloc;
  }

  ModuleOp module;

  Value shBuffers;
  Value barriers;
  RankedTensorType writeBarriersType;
  Value writeBarriersAlloc;
  RankedTensorType readBarriersType;
  Value readBarriersAlloc;
  RankedTensorType writeCommitsType;
  Value writeCommitsAlloc;
};

} // namespace instrument
} // namespace triton
} // namespace mlir
