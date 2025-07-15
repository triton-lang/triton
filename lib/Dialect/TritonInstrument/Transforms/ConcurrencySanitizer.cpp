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

bool canAllocBeInstrumented(Operation *op) {
  if (llvm::any_of(op->getUsers(),
                   [](Operation *user) { return isa<tt::CallOp>(user); })) {
    op->emitWarning("Allocation is used in a function call, cannot instrument");
    return false;
  }
  if (llvm::all_of(op->getUsers(), [](Operation *user) {
        return !isa<ttg::MemDescIndexOp>(user);
      })) {
    return true;
  }
  if (llvm::all_of(op->getUsers(), [](Operation *user) {
        return isa<ttg::MemDescIndexOp>(user);
      })) {
    return true;
  }
  op->emitWarning(
      "Allocation is used in an inconsistent way, cannot instrument");
  return false;
}

// Interpret local_allocs that are used in ttg.memdesc_index as multibuffered
bool isMultiBuffered(Operation *op) {
  return llvm::any_of(op->getUsers(), [](Operation *user) {
    return isa<ttg::MemDescIndexOp>(user);
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

uint64_t getAllocationOffset(ttng::TMEMAllocOp op) {
  auto colOffsetAttr = op->getAttr("tensor_memory_col_offset");
  auto rowOffsetAttr = op->getAttr("tensor_memory_row_offset");
  if (!colOffsetAttr || !rowOffsetAttr) {
    llvm::report_fatal_error(
        "ConcurrencySanitizer should run after AllocateSharedMemory and "
        "TensorMemoryAllocation pass.");
  }
  int colOffset = cast<IntegerAttr>(colOffsetAttr).getInt();
  int rowOffset = cast<IntegerAttr>(rowOffsetAttr).getInt();
  return colOffset | (rowOffset << 16);
}

unsigned getNumBuffers(Operation *op) {
  ttg::MemDescType ty = cast<ttg::MemDescType>(op->getResultTypes().front());
  return ty.getShape()[0];
}

unsigned getSubBufferSize(triton::gpu::LocalAllocOp op) {
  ttg::MemDescType ty = op.getType();
  unsigned elSize = ty.getElementType().getIntOrFloatBitWidth() / 8;
  return product(ty.getShape().drop_front()) * elSize;
}

unsigned getSubBufferSize(ttng::TMEMAllocOp op) {
  int numCols = ttng::getTmemAllocSizes(op.getType()).numCols;
  int numSubBuffers = getNumBuffers(op);
  return numCols / numSubBuffers;
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
    llvm::SmallVector<llvm::SetVector<int32_t>> bufSets(numMemTypes);
    llvm::SetVector<int32_t> barrierSet;
    module.walk([&](triton::gpu::LocalAllocOp op) {
      if (!canAllocBeInstrumented(op)) {
        return WalkResult::advance();
      }
      int32_t baseOffset = getAllocationOffset(op);
      auto &setToAdd =
          isBarrier(op) ? barrierSet : bufSets[(int)MemType::SHARED];
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

    module.walk([&](ttng::TMEMAllocOp op) {
      if (!canAllocBeInstrumented(op)) {
        return WalkResult::advance();
      }
      int32_t baseOffset = getAllocationOffset(op);
      bufSets[(int)MemType::TMEM].insert(baseOffset);
      if (isMultiBuffered(op)) {
        unsigned numBuffers = getNumBuffers(op);
        assert(numBuffers > 0 && "Expected at least one buffer");
        unsigned subBufferSize = getSubBufferSize(op);
        for (unsigned i = 1; i < numBuffers; ++i) {
          bufSets[(int)MemType::TMEM].insert(baseOffset + i * subBufferSize);
        }
      }
      return WalkResult::advance();
    });

    tt::FuncOp entryPoint = getEntryPoint(module);
    assert(entryPoint);

    if (bufSets[(int)MemType::SHARED].empty() &&
        bufSets[(int)MemType::TMEM].empty()) {
      return;
    }

    SmallVector<int32_t> barrierValues = llvm::to_vector(barrierSet);
    if (!barrierValues.empty()) {
      barrierValues.resize(llvm::NextPowerOf2(barrierValues.size() - 1), 0);
    }

    ImplicitLocOpBuilder b(entryPoint.getLoc(), entryPoint);
    b.setInsertionPointToStart(&entryPoint.getBody().front());

    SmallVector<SmallVector<int32_t>> bufValues(numMemTypes);
    for (MemType memType : {MemType::SHARED, MemType::TMEM}) {
      bufValues[(int)memType] = llvm::to_vector(bufSets[(int)memType]);
      if (bufValues[(int)memType].empty()) {
        continue;
      }
      bufValues[(int)memType].resize(
          llvm::NextPowerOf2(bufValues[(int)memType].size() - 1), 0);
      buffersTensor[(int)memType] =
          createBufferPointersTensor(b, memType, bufValues[(int)memType]);
    }

    if (!barrierValues.empty()) {
      // Barriers allocations are in shared memory
      barriers = createBufferPointersTensor(b, MemType::SHARED, barrierValues);

      for (MemType memType : {MemType::SHARED, MemType::TMEM}) {
        int iMemType = (int)memType;
        // Create state tensors:
        int numBufs = bufValues[iMemType].size();
        if (numBufs > 0) {
          writeBarriersType[iMemType] =
              RankedTensorType::get({numBufs}, b.getIntegerType(64),
                                    getThreadLocalBlockedEncoding(numBufs));
          TypedValue<RankedTensorType> writeBarriers =
              tti::createConstIntTensor(b, b.getLoc(), 0,
                                        writeBarriersType[iMemType]);
          writeBarriersAlloc[iMemType] =
              createInitializedScratchMemory(b, writeBarriers);

          readBarriersType[iMemType] = RankedTensorType::get(
              {numBufs, (long)barrierValues.size()}, b.getIntegerType(8),
              getReadBarriersEncoding(numBufs, barrierValues.size()));
          TypedValue<RankedTensorType> readBarriers = tti::createConstIntTensor(
              b, b.getLoc(), 0, readBarriersType[iMemType]);
          readBarriersAlloc[iMemType] =
              createInitializedScratchMemory(b, readBarriers);
        }
      }
    }

    // Create write commits tensor
    if (!bufValues[(int)MemType::SHARED].empty()) {
      writeCommitsType = RankedTensorType::get(
          {(long)bufValues[(int)MemType::SHARED].size()}, b.getIntegerType(8),
          getThreadLocalBlockedEncoding(
              bufValues[(int)MemType::SHARED].size()));
      TypedValue<RankedTensorType> writeCommits =
          tti::createConstIntTensor(b, b.getLoc(), 0, writeCommitsType);
      writeCommitsAlloc = createInitializedScratchMemory(b, writeCommits);
    }

    instrumentMemoryOperations(b);
  }

private:
  void addWriteChecks(ImplicitLocOpBuilder &b, Value buf, Value pred,
                      MemType memType) {
    if (barriers) {
      b.create<tti::ExperimentalCheckOutstandingWritesOp>(
          buf, buffersTensor[(int)memType], writeBarriersAlloc[(int)memType],
          writeBarriersType[(int)memType], pred);
    }
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED) {
      b.create<tti::ExperimentalCheckWriteCommitOp>(
          buf, buffersTensor[(int)memType], writeCommitsAlloc, writeCommitsType,
          pred);
    }
  }

  void addReadChecks(ImplicitLocOpBuilder &b, Value buf, Value pred,
                     MemType memType) {
    if (barriers) {
      b.create<tti::ExperimentalCheckOutstandingReadsOp>(
          buf, buffersTensor[(int)memType], readBarriersAlloc[(int)memType],
          readBarriersType[(int)memType], pred);
    }
  }

  struct MemEffects {
    enum class RW { Read, Write };
    Value buf;
    RW rw;
    SmallVector<std::tuple<Value, Value>> barriersAndPreds;
    bool commitTracking = false;
    Value pred;
  };

  SmallVector<MemEffects> getMemEffects(Operation *op) {
    SmallVector<MemEffects> effects;
    if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
      effects.emplace_back(
          MemEffects{.buf = copyOp.getResult(),
                     .rw = MemEffects::RW::Write,
                     .barriersAndPreds = {{copyOp.getBarrier(), nullptr}},
                     .pred = copyOp.getPred()});
    }
    if (auto copyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      effects.emplace_back(MemEffects{.buf = copyOp.getResult(),
                                      .rw = MemEffects::RW::Write,
                                      .commitTracking = true});
    }
    if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(op)) {
      effects.emplace_back(
          MemEffects{.buf = loadOp.getSrc(), .rw = MemEffects::RW::Read});
    }
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
      effects.emplace_back(
          MemEffects{.buf = storeOp.getDst(), .rw = MemEffects::RW::Write});
    }
    if (auto mmav5Op = dyn_cast<ttng::TCGen5MMAOp>(op)) {
      SmallVector<std::tuple<Value, Value>> barriersAndPreds = llvm::to_vector(
          llvm::zip(mmav5Op.getBarriers(), mmav5Op.getBarrierPreds()));

      effects.emplace_back(MemEffects{.buf = mmav5Op.getA(),
                                      .rw = MemEffects::RW::Read,
                                      .barriersAndPreds = barriersAndPreds,
                                      .pred = mmav5Op.getPred()});

      effects.emplace_back(MemEffects{.buf = mmav5Op.getB(),
                                      .rw = MemEffects::RW::Read,
                                      .barriersAndPreds = barriersAndPreds,
                                      .pred = mmav5Op.getPred()});

      effects.emplace_back(MemEffects{.buf = mmav5Op.getAccumulator(),
                                      .rw = MemEffects::RW::Write,
                                      .barriersAndPreds = barriersAndPreds,
                                      .pred = mmav5Op.getPred()});
    }
    return effects;
  }

  void instrumentMemoryOperations(ImplicitLocOpBuilder &b) {
    module.walk([&](Operation *op) {
      b.setLoc(op->getLoc());
      b.setInsertionPoint(op);
      SmallVector<MemEffects> effects = getMemEffects(op);
      if (!effects.empty()) {
        for (MemEffects effect : effects) {
          Value buf = effect.buf;
          auto bufType = cast<ttg::MemDescType>(buf.getType());
          MemType memType = MemType::TMEM;
          if (isa<ttg::NVMMASharedEncodingAttr>(bufType.getEncoding())) {
            memType = MemType::SHARED;
          }
          if (effect.rw == MemEffects::RW::Read) {
            // For op that is reading, we only need to check if anything else
            // is writing to the same buffer.
            addWriteChecks(b, buf, effect.pred, memType);
            if (!effect.barriersAndPreds.empty()) {
              for (auto [barrier, pred] : effect.barriersAndPreds) {
                if (pred && effect.pred) {
                  pred = b.create<arith::AndIOp>(effect.pred, pred);
                }
                b.create<tti::ExperimentalMarkAsReadOp>(
                    buf, barrier, buffersTensor[(int)memType], barriers,
                    readBarriersAlloc[(int)memType],
                    readBarriersType[(int)memType], pred);
              }
            }
            // TODO: commit tracking for reads
          }
          if (effect.rw == MemEffects::RW::Write) {
            // Op is writing to the buffer, we need to check if anything else
            // is reading or writing to the same buffer.
            addWriteChecks(b, buf, effect.pred, memType);
            addReadChecks(b, buf, effect.pred, memType);
            if (!effect.barriersAndPreds.empty()) {
              for (auto [barrier, pred] : effect.barriersAndPreds) {
                if (pred && effect.pred) {
                  pred = b.create<arith::AndIOp>(effect.pred, pred);
                }
                b.create<tti::ExperimentalMarkAsWriteOp>(
                    buf, barrier, buffersTensor[(int)memType],
                    writeBarriersAlloc[(int)memType],
                    writeBarriersType[(int)memType], pred);
              }
            }
            if (effect.commitTracking) {
              b.create<tti::ExperimentalStageWriteForCommitOp>(
                  buf, buffersTensor[(int)memType], writeCommitsAlloc,
                  writeCommitsType, effect.pred);
            }
          }
        }
      }

      // if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
      //   MemType memType = MemType::SHARED;
      //   auto buf = copyOp.getResult();
      //   auto pred = copyOp.getPred();
      //   auto barrier = copyOp.getBarrier();
      //   assert(barriers);
      //   addWriteChecks(b, buf, pred, memType);
      //   addReadChecks(b, buf, pred, memType);
      //   b.create<tti::ExperimentalMarkAsWriteOp>(
      //       buf, barrier, buffersTensor[(int)memType],
      //       writeBarriersAlloc[(int)memType],
      //       writeBarriersType[(int)memType], pred);
      // }
      // if (auto mmav5Op = dyn_cast<ttng::TCGen5MMAOp>(op)) {
      //   auto pred = mmav5Op.getPred();
      //   b.setInsertionPoint(mmav5Op);

      //   auto instrumentOperand = [&](TypedValue<ttg::MemDescType> operand) {
      //     MemType memType = MemType::TMEM;
      //     if (isa<ttg::NVMMASharedEncodingAttr>(
      //             operand.getType().getEncoding())) {
      //       memType = MemType::SHARED;
      //     }
      //     addWriteChecks(b, operand, pred, memType);
      //     for (auto barrier : mmav5Op.getBarriers()) {
      //       assert(barriers);
      //       b.create<tti::ExperimentalMarkAsReadOp>(
      //           operand, barrier, buffersTensor[(int)memType], barriers,
      //           readBarriersAlloc[(int)memType],
      //           readBarriersType[(int)memType], pred);
      //     }
      //   };

      //   instrumentOperand(mmav5Op.getA());
      //   instrumentOperand(mmav5Op.getB());

      //   addWriteChecks(b, mmav5Op.getAccumulator(), pred, MemType::TMEM);
      //   addReadChecks(b, mmav5Op.getAccumulator(), pred, MemType::TMEM);
      //   for (auto barrier : mmav5Op.getBarriers()) {
      //     assert(barriers);
      //     b.create<tti::ExperimentalMarkAsWriteOp>(
      //         mmav5Op.getAccumulator(), barrier,
      //         buffersTensor[(int)MemType::TMEM],
      //         writeBarriersAlloc[(int)MemType::TMEM],
      //         writeBarriersType[(int)MemType::TMEM], pred);
      //   }
      // }
      if (auto waitOp = dyn_cast<ttng::WaitBarrierOp>(op)) {
        assert(barriers);
        auto pred = waitOp.getPred();
        auto barrier = waitOp.getAlloc();
        for (MemType memType : {MemType::SHARED, MemType::TMEM}) {
          if (writeBarriersAlloc[(int)memType]) {
            b.create<tti::ExperimentalClearWriteBarrierOp>(
                barrier, writeBarriersAlloc[(int)memType],
                writeBarriersType[(int)memType], pred);
            b.create<tti::ExperimentalClearReadBarrierOp>(
                barrier, barriers, readBarriersAlloc[(int)memType],
                readBarriersType[(int)memType], pred);
          }
        }
      }
      // if (auto asyncCopyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      //   addWriteChecks(b, asyncCopyOp.getResult(), nullptr, MemType::SHARED);
      //   addReadChecks(b, asyncCopyOp.getResult(), nullptr, MemType::SHARED);
      //   b.create<tti::ExperimentalStageWriteForCommitOp>(
      //       asyncCopyOp.getResult(), buffersTensor[(int)MemType::SHARED],
      //       writeCommitsAlloc, writeCommitsType, nullptr);
      // }
      if (auto asyncCommitGroupOp = dyn_cast<ttg::AsyncCommitGroupOp>(op)) {
        b.create<tti::ExperimentalCommitWritesOp>(writeCommitsAlloc,
                                                  writeCommitsType, nullptr);
      }
      if (auto asyncWaitOp = dyn_cast<ttg::AsyncWaitOp>(op)) {
        b.create<tti::ExperimentalClearWriteCommitsOp>(
            writeCommitsAlloc, writeCommitsType, asyncWaitOp.getNum(), nullptr);
      }
      // if (auto tmemLoadOp = dyn_cast<ttng::TMEMLoadOp>(op)) {
      //   addWriteChecks(b, tmemLoadOp.getSrc(), nullptr, MemType::TMEM);
      // }
      // if (auto tmemStoreOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
      //   addWriteChecks(b, tmemStoreOp.getDst(), nullptr, MemType::TMEM);
      //   addReadChecks(b, tmemStoreOp.getDst(), nullptr, MemType::TMEM);
      // }
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

  Value createBufferPointersTensor(ImplicitLocOpBuilder &builder,
                                   MemType memType,
                                   SmallVector<int32_t> values) {
    int64_t size = values.size();
    assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
    Type elType = builder.getI64Type();
    auto tensorType = RankedTensorType::get(
        {size}, elType, getThreadLocalBlockedEncoding(size));
    SmallVector<APInt> apInts = llvm::to_vector(
        llvm::map_range(values, [](int64_t v) { return APInt(64, v); }));
    auto denseAttr = DenseElementsAttr::get(tensorType, apInts);
    auto op = builder.create<tti::ExperimentalBufferPointersOp>(
        tensorType, values, memType);
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

  static constexpr int numMemTypes = getMaxEnumValForMemType() + 1;
  Value buffersTensor[numMemTypes];
  Value barriers;
  RankedTensorType writeBarriersType[numMemTypes];
  Value writeBarriersAlloc[numMemTypes];
  RankedTensorType readBarriersType[numMemTypes];
  Value readBarriersAlloc[numMemTypes];
  RankedTensorType writeCommitsType;
  Value writeCommitsAlloc;
};

} // namespace instrument
} // namespace triton
} // namespace mlir
