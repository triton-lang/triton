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

constexpr int NUM_THREADS = 16;
constexpr int TMA_THREAD_OFFSET = NUM_THREADS;
constexpr int TC_THREAD_OFFSET = TMA_THREAD_OFFSET + NUM_THREADS;
constexpr int TOTAL_NUM_THREADS = TC_THREAD_OFFSET + NUM_THREADS;
constexpr int THREADS_BITMASK_SIZE = llvm::NextPowerOf2(TOTAL_NUM_THREADS);

bool isTMAOp(Operation *op) {
  return isa<ttng::AsyncTMACopyGlobalToLocalOp,
             ttng::AsyncTMACopyLocalToGlobalOp, ttng::AsyncTMAGatherOp,
             ttng::AsyncTMAScatterOp>(op);
}

bool isTensorCoreOp(Operation *op) {
  return isa<ttng::TCGen5MMAOp, ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp,
             ttng::TCGen5CommitOp>(op);
}

int getCurrentThread(Operation *op) {
  int thread = 0;
  if (isTMAOp(op)) {
    thread += TMA_THREAD_OFFSET;
    return thread;
  }
  if (isTensorCoreOp(op)) {
    thread += TC_THREAD_OFFSET;
    return thread;
  }
  return thread;
}

// Peer threads are the equivalent threads in the TMA, TC and normal
// thread classes.
SmallVector<int, 3> getThreadPeers(int thread) {
  int baseThread = thread % NUM_THREADS;
  return {baseThread, baseThread + TMA_THREAD_OFFSET,
          baseThread + TC_THREAD_OFFSET};
}

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
        return isa<ttg::MemDescIndexOp>(user) || isa<ttg::LocalDeallocOp>(user);
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

bool hasCpAsync(ModuleOp module) {
  bool hasCpAsync = false;
  module.walk([&](Operation *op) {
    if (isa<ttg::AsyncCopyGlobalToLocalOp, ttg::AsyncCommitGroupOp,
            ttg::AsyncWaitOp>(op)) {
      hasCpAsync = true;
    }
  });
  return hasCpAsync;
}

bool hasWGMMA(ModuleOp module) {
  bool hasWGMMA = false;
  module.walk([&](Operation *op) {
    if (isa<ttng::WarpGroupDotOp, ttng::WarpGroupDotWaitOp>(op)) {
      hasWGMMA = true;
    }
  });
  return hasWGMMA;
}

} // namespace

class ConcurrencySanitizerPass
    : public impl::TritonInstrumentConcurrencySanitizerBase<
          ConcurrencySanitizerPass> {
public:
  void runOnOperation() override {
    module = getOperation();
    // Collect shared memory buffers allocated in the module
    llvm::SmallVector<llvm::SetVector<int32_t>> bufSets(numMemTypes);
    llvm::SetVector<int32_t> barrierSet;
    module.walk([&](triton::gpu::LocalAllocOp op) {
      if (!canAllocBeInstrumented(op)) {
        return WalkResult::advance();
      }
      int32_t baseOffset = getAllocationOffset(op);
      auto &setToAdd =
          isBarrier(op) ? barrierSet : bufSets[(int)MemType::SHARED_MEM];
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
      bufSets[(int)MemType::TENSOR_MEM].insert(baseOffset);
      if (isMultiBuffered(op)) {
        unsigned numBuffers = getNumBuffers(op);
        assert(numBuffers > 0 && "Expected at least one buffer");
        unsigned subBufferSize = getSubBufferSize(op);
        for (unsigned i = 1; i < numBuffers; ++i) {
          bufSets[(int)MemType::TENSOR_MEM].insert(baseOffset +
                                                   i * subBufferSize);
        }
      }
      return WalkResult::advance();
    });

    tt::FuncOp entryPoint = getEntryPoint(module);
    assert(entryPoint);

    if (bufSets[(int)MemType::SHARED_MEM].empty() &&
        bufSets[(int)MemType::TENSOR_MEM].empty()) {
      return;
    }

    SmallVector<int32_t> barrierValues = llvm::to_vector(barrierSet);
    if (!barrierValues.empty()) {
      barrierValues.resize(llvm::NextPowerOf2(barrierValues.size() - 1), 0);
    }

    ImplicitLocOpBuilder b(entryPoint.getLoc(), entryPoint);
    b.setInsertionPointToStart(&entryPoint.getBody().front());

    SmallVector<SmallVector<int32_t>> bufValues(numMemTypes);
    for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
      int iMemType = (int)memType;
      bufValues[iMemType] = llvm::to_vector(bufSets[iMemType]);
      if (bufValues[iMemType].empty()) {
        continue;
      }
      bufValues[iMemType].resize(
          llvm::NextPowerOf2(bufValues[iMemType].size() - 1), 0);
      buffersTensor[iMemType] =
          createBufferPointersTensor(b, memType, bufValues[iMemType]);
      int numBufs = bufValues[iMemType].size();

      writeStateType[iMemType] =
          RankedTensorType::get({numBufs}, b.getIntegerType(8),
                                getThreadLocalBlockedEncoding(numBufs));
      TypedValue<RankedTensorType> writeState =
          tti::createConstIntTensor(b, b.getLoc(), 0, writeStateType[iMemType]);
      writeStateAlloc[iMemType] = createInitializedScratchMemory(b, writeState);

      createZeroInitStateTensor(b, numBufs, 0, 64,
                                writeVisibilityType[iMemType],
                                writeVisibilityAlloc[iMemType]);
      createZeroInitStateTensor(b, numBufs, THREADS_BITMASK_SIZE, 64,
                                readVisibilityType[iMemType],
                                readVisibilityAlloc[iMemType]);
    }

    if (!barrierValues.empty()) {
      // Barriers allocations are in shared memory
      barriers =
          createBufferPointersTensor(b, MemType::SHARED_MEM, barrierValues);

      for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
        int iMemType = (int)memType;
        // Create state tensors:
        int numBufs = bufValues[iMemType].size();
        int numBarriers = barrierValues.size();
        if (numBufs > 0) {
          createZeroInitStateTensor(b, numBufs, numBarriers, 8,
                                    writeTrackingType[iMemType],
                                    writeTrackingAlloc[iMemType]);
          createZeroInitStateTensor(b, numBufs, numBarriers, 64,
                                    readTrackingType[iMemType],
                                    readTrackingAlloc[iMemType]);

          writeBarriersType[iMemType] = RankedTensorType::get(
              {numBufs, numBarriers}, b.getIntegerType(8),
              getThreadLocalBlockedEncoding(numBufs, numBarriers));
          TypedValue<RankedTensorType> writeBarriers =
              tti::createConstIntTensor(b, b.getLoc(), 0,
                                        writeBarriersType[iMemType]);
          writeBarriersAlloc[iMemType] =
              createInitializedScratchMemory(b, writeBarriers);

          readBarriersType[iMemType] = RankedTensorType::get(
              {numBufs, numBarriers}, b.getIntegerType(8),
              getThreadLocalBlockedEncoding(numBufs, numBarriers));
          TypedValue<RankedTensorType> readBarriers = tti::createConstIntTensor(
              b, b.getLoc(), 0, readBarriersType[iMemType]);
          readBarriersAlloc[iMemType] =
              createInitializedScratchMemory(b, readBarriers);
        }
      }
    }

    // Create write commits tensor for cp-async
    if (hasCpAsync(module)) {
      assert(!bufValues[(int)MemType::SHARED_MEM].empty());
      asyncCpCommitsType = RankedTensorType::get(
          {(long)bufValues[(int)MemType::SHARED_MEM].size()},
          b.getIntegerType(8),
          getThreadLocalBlockedEncoding(
              bufValues[(int)MemType::SHARED_MEM].size()));
      TypedValue<RankedTensorType> writeCommits =
          tti::createConstIntTensor(b, b.getLoc(), 0, asyncCpCommitsType);
      asyncCpCommitsAlloc = createInitializedScratchMemory(b, writeCommits);
    }

    // Create reads commits tensor for wgmma
    if (hasWGMMA(module)) {
      assert(!bufValues[(int)MemType::SHARED_MEM].empty());
      wgmmaCommitsType = RankedTensorType::get(
          {(long)bufValues[(int)MemType::SHARED_MEM].size()},
          b.getIntegerType(8),
          getThreadLocalBlockedEncoding(
              bufValues[(int)MemType::SHARED_MEM].size()));
      TypedValue<RankedTensorType> readCommits =
          tti::createConstIntTensor(b, b.getLoc(), 0, wgmmaCommitsType);
      wgmmaCommitsAlloc = createInitializedScratchMemory(b, readCommits);
    }

    instrumentMemoryOperations(b);
  }

private:
  void addWriteChecks(ImplicitLocOpBuilder &b, Value buf, Value pred,
                      MemType memType, bool hwPipelined, int thread) {
    if (barriers) {
      b.create<tti::ExperimentalVerifyWriteVisibilityOp>(
          buf, thread, buffersTensor[(int)memType],
          writeVisibilityAlloc[(int)memType], writeVisibilityType[(int)memType],
          pred);
    }
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM && asyncCpCommitsAlloc) {
      b.create<tti::ExperimentalCheckOutstandingCommitsOp>(
          buf, buffersTensor[(int)memType], asyncCpCommitsAlloc,
          asyncCpCommitsType, "async_copy_global_to_shared", pred);
    }
  }

  void addReadChecks(ImplicitLocOpBuilder &b, Value buf, Value pred,
                     MemType memType, int thread) {
    if (barriers) {
      b.create<tti::ExperimentalVerifyReadVisibilityOp>(
          buf, thread, buffersTensor[(int)memType],
          readVisibilityAlloc[(int)memType], readVisibilityType[(int)memType],
          pred);
    }
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM && wgmmaCommitsAlloc) {
      b.create<tti::ExperimentalCheckOutstandingCommitsOp>(
          buf, buffersTensor[(int)memType], wgmmaCommitsAlloc, wgmmaCommitsType,
          "warpgroup_mma operand read", pred);
    }
  }

  struct MemEffects {
    enum class RW { Read, Write } rw;
    enum class TrackingKind {
      None,
      Barrier,
      wgmmaCommit,
      asyncCpCommit
    } trackingKind = TrackingKind::None;
    Value buf;
    SmallVector<std::tuple<Value, Value>> barriersAndPreds;
    bool hwPipelined = false;
    Value pred;
  };

  SmallVector<MemEffects> getMemEffects(Operation *op) {
    SmallVector<MemEffects> effects;
    if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Write,
                     .trackingKind = MemEffects::TrackingKind::Barrier,
                     .buf = copyOp.getResult(),
                     .barriersAndPreds = {{copyOp.getBarrier(), nullptr}},
                     .pred = copyOp.getPred()});
    }
    if (auto storeOp = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op)) {
      effects.emplace_back(MemEffects{
          .rw = MemEffects::RW::Read,
          .trackingKind = MemEffects::TrackingKind::None, // async tma writes
                                                          // not modelled yet
          .buf = storeOp.getSrc()});
    }
    if (auto gatherOp = dyn_cast<ttng::AsyncTMAGatherOp>(op)) {
      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Write,
                     .trackingKind = MemEffects::TrackingKind::Barrier,
                     .buf = gatherOp.getResult(),
                     .barriersAndPreds = {{gatherOp.getBarrier(), nullptr}},
                     .pred = gatherOp.getPred()});
    }
    if (auto scatterOp = dyn_cast<ttng::AsyncTMAScatterOp>(op)) {
      effects.emplace_back(MemEffects{
          .rw = MemEffects::RW::Read,
          .trackingKind = MemEffects::TrackingKind::None, // async tma writes
                                                          // not modelled yet
          .buf = scatterOp.getSrc(),
      });
    }
    if (auto copyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Write,
                     .trackingKind = MemEffects::TrackingKind::asyncCpCommit,
                     .buf = copyOp.getResult()});
    }
    if (auto loadOp = dyn_cast<ttg::LocalLoadOp>(op)) {
      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Read, .buf = loadOp.getSrc()});
    }
    if (auto storeOp = dyn_cast<ttg::LocalStoreOp>(op)) {
      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Write, .buf = storeOp.getDst()});
    }
    if (auto allocOp = dyn_cast<ttg::LocalAllocOp>(op)) {
      if (allocOp.getSrc()) {
        effects.emplace_back(MemEffects{.rw = MemEffects::RW::Write,
                                        .buf = allocOp.getResult()});
      }
    }
    if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(op)) {
      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Read, .buf = loadOp.getSrc()});
    }
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Write, .buf = storeOp.getDst()});
    }
    if (auto allocOp = dyn_cast<ttng::TMEMAllocOp>(op)) {
      if (allocOp.getSrc()) {
        effects.emplace_back(MemEffects{.rw = MemEffects::RW::Write,
                                        .buf = allocOp.getResult()});
      }
    }
    if (auto mmav5Op = dyn_cast<ttng::TCGen5MMAOp>(op)) {
      SmallVector<std::tuple<Value, Value>> barriersAndPreds = llvm::to_vector(
          llvm::zip(mmav5Op.getBarriers(), mmav5Op.getBarrierPreds()));

      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Read,
                     .trackingKind = MemEffects::TrackingKind::Barrier,
                     .buf = mmav5Op.getA(),
                     .barriersAndPreds = barriersAndPreds,
                     .pred = mmav5Op.getPred()});

      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Read,
                     .trackingKind = MemEffects::TrackingKind::Barrier,
                     .buf = mmav5Op.getB(),
                     .barriersAndPreds = barriersAndPreds,
                     .pred = mmav5Op.getPred()});

      effects.emplace_back(
          MemEffects{.rw = MemEffects::RW::Write,
                     .trackingKind = MemEffects::TrackingKind::Barrier,
                     .buf = mmav5Op.getAccumulator(),
                     .barriersAndPreds = barriersAndPreds,
                     .hwPipelined = true,
                     .pred = mmav5Op.getPred()});
    }
    if (auto wgmmaOp = dyn_cast<ttng::WarpGroupDotOp>(op)) {
      if (wgmmaOp.getIsAsync() == true) {
        if (isa<ttg::SharedEncodingTrait>(
                wgmmaOp.getA().getType().getEncoding())) {
          effects.emplace_back(
              MemEffects{.rw = MemEffects::RW::Read,
                         .trackingKind = MemEffects::TrackingKind::wgmmaCommit,
                         .buf = wgmmaOp.getA()});
        }
        if (isa<ttg::SharedEncodingTrait>(
                wgmmaOp.getB().getType().getEncoding())) {
          effects.emplace_back(
              MemEffects{.rw = MemEffects::RW::Read,
                         .trackingKind = MemEffects::TrackingKind::wgmmaCommit,
                         .buf = wgmmaOp.getB()});
        }
      }
    }
    return effects;
  }

  void instrumentMemoryOperations(ImplicitLocOpBuilder &b) {
    module.walk([&](Operation *op) {
      // TODO: get the correct thread
      int thread = getCurrentThread(op);
      b.setLoc(op->getLoc());
      b.setInsertionPoint(op);
      SmallVector<MemEffects> effects = getMemEffects(op);
      if (!effects.empty()) {
        for (MemEffects effect : effects) {
          Value buf = effect.buf;
          auto bufType = cast<ttg::MemDescType>(buf.getType());
          MemType memType = MemType::TENSOR_MEM;
          if (isa<ttg::SharedEncodingTrait>(bufType.getEncoding())) {
            memType = MemType::SHARED_MEM;
          }
          if (effect.rw == MemEffects::RW::Read) {
            // For op that is reading, we only need to check if anything else
            // is writing to the same buffer.
            addWriteChecks(b, buf, effect.pred, memType, effect.hwPipelined,
                           thread);
            if (effect.trackingKind == MemEffects::TrackingKind::Barrier) {
              b.create<tti::ExperimentalSetReadVisibilityOp>(
                  buf, thread, buffersTensor[(int)memType],
                  readVisibilityAlloc[(int)memType],
                  readVisibilityType[(int)memType], effect.pred);
              for (auto [barrier, pred] : effect.barriersAndPreds) {
                if (pred && effect.pred) {
                  pred = b.create<arith::AndIOp>(effect.pred, pred);
                }
                b.create<tti::ExperimentalTrackVisibleReadsOp>(
                    barrier, thread, barriers,
                    readVisibilityAlloc[(int)memType],
                    readVisibilityType[(int)memType],
                    readTrackingAlloc[(int)memType],
                    readTrackingType[(int)memType], pred);
              }
            }
            if (effect.trackingKind == MemEffects::TrackingKind::wgmmaCommit) {
              assert(isa<ttng::WarpGroupDotOp>(op));
              assert(memType == MemType::SHARED_MEM);
              b.create<tti::ExperimentalStageAccessForCommitOp>(
                  buf, buffersTensor[(int)memType], wgmmaCommitsAlloc,
                  wgmmaCommitsType, effect.pred);
            }
            assert(effect.trackingKind !=
                   MemEffects::TrackingKind::asyncCpCommit);
          }
          if (effect.rw == MemEffects::RW::Write) {
            if (isa<ttg::LocalAllocOp, ttng::TMEMAllocOp>(op)) {
              // For allocs place insert point after the alloc - we want to
              // check if it is not overwriting any earlier allocation, but the
              // memref value can be referenced only after it is created.
              b.setInsertionPointAfter(op);
            }
            // Op is writing to the buffer, we need to check if anything else
            // is reading or writing to the same buffer.
            addWriteChecks(b, buf, effect.pred, memType, effect.hwPipelined,
                           thread);
            addReadChecks(b, buf, effect.pred, memType, thread);
            if (effect.trackingKind == MemEffects::TrackingKind::Barrier) {
              b.create<tti::ExperimentalSetWriteVisibilityOp>(
                  buf, thread, buffersTensor[(int)memType],
                  writeVisibilityAlloc[(int)memType],
                  writeVisibilityType[(int)memType], effect.pred);
              b.create<tti::ExperimentalClearWriteTrackingOp>(
                  buf, buffersTensor[(int)memType],
                  writeTrackingAlloc[(int)memType],
                  writeTrackingType[(int)memType], effect.pred);
              b.create<tti::ExperimentalClearReadVisibilityOp>(
                  buf, buffersTensor[(int)memType],
                  readVisibilityAlloc[(int)memType],
                  readVisibilityType[(int)memType], effect.pred);
              b.create<tti::ExperimentalClearReadTrackingOp>(
                  buf, buffersTensor[(int)memType],
                  readTrackingAlloc[(int)memType],
                  readTrackingType[(int)memType], effect.pred);
              for (auto [barrier, pred] : effect.barriersAndPreds) {
                if (pred && effect.pred) {
                  pred = b.create<arith::AndIOp>(effect.pred, pred);
                }
                b.create<tti::ExperimentalTrackVisibleWritesOp>(
                    barrier, thread, barriers,
                    writeVisibilityAlloc[(int)memType],
                    writeVisibilityType[(int)memType],
                    writeTrackingAlloc[(int)memType],
                    writeTrackingType[(int)memType], pred);
                b.create<tti::ExperimentalTrackVisibleReadsOp>(
                    barrier, thread, barriers,
                    readVisibilityAlloc[(int)memType],
                    readVisibilityType[(int)memType],
                    readTrackingAlloc[(int)memType],
                    readTrackingType[(int)memType], pred);
              }
            }
            if (effect.trackingKind ==
                MemEffects::TrackingKind::asyncCpCommit) {
              assert(memType == MemType::SHARED_MEM);
              b.create<tti::ExperimentalStageAccessForCommitOp>(
                  buf, buffersTensor[(int)memType], asyncCpCommitsAlloc,
                  asyncCpCommitsType, effect.pred);
            }
            assert(effect.trackingKind !=
                   MemEffects::TrackingKind::wgmmaCommit);
          }
        }
      }

      if (auto waitOp = dyn_cast<ttng::WaitBarrierOp>(op)) {
        assert(barriers);
        auto pred = waitOp.getPred();
        auto barrier = waitOp.getAlloc();
        for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
          if (writeBarriersAlloc[(int)memType]) {
            // Transfer visible writes and reads to all peer threads
            for (int peer : getThreadPeers(thread)) {
              b.create<tti::ExperimentalTransferVisibleWritesOp>(
                  barrier, peer, barriers, writeVisibilityAlloc[(int)memType],
                  writeVisibilityType[(int)memType],
                  writeTrackingAlloc[(int)memType],
                  writeTrackingType[(int)memType], pred);
              b.create<tti::ExperimentalTransferVisibleReadsOp>(
                  barrier, peer, barriers, readVisibilityAlloc[(int)memType],
                  readVisibilityType[(int)memType],
                  readTrackingAlloc[(int)memType],
                  readTrackingType[(int)memType], pred);
            }
            b.create<tti::ExperimentalClearWriteBarrierOp>(
                barrier, barriers, writeBarriersAlloc[(int)memType],
                writeBarriersType[(int)memType], writeStateAlloc[(int)memType],
                writeStateType[(int)memType], pred);
            b.create<tti::ExperimentalClearReadBarrierOp>(
                barrier, barriers, readBarriersAlloc[(int)memType],
                readBarriersType[(int)memType], pred);
          }
        }
      }
      if (auto commitOp = dyn_cast<ttng::TCGen5CommitOp>(op)) {
        for (MemType memType : {MemType::TENSOR_MEM, MemType::SHARED_MEM}) {
          b.create<tti::ExperimentalTrackVisibleWritesOp>(
              commitOp.getBarrier(), thread, barriers,
              writeVisibilityAlloc[(int)memType],
              writeVisibilityType[(int)memType],
              writeTrackingAlloc[(int)memType], writeTrackingType[(int)memType],
              commitOp.getPred());
          b.create<tti::ExperimentalTrackVisibleReadsOp>(
              commitOp.getBarrier(), thread, barriers,
              readVisibilityAlloc[(int)memType],
              readVisibilityType[(int)memType], readTrackingAlloc[(int)memType],
              readTrackingType[(int)memType], commitOp.getPred());
        }
      }
      if (auto asyncCommitGroupOp = dyn_cast<ttg::AsyncCommitGroupOp>(op)) {
        b.create<tti::ExperimentalCommitAccessesOp>(
            asyncCpCommitsAlloc, asyncCpCommitsType, nullptr);
      }
      if (auto asyncWaitOp = dyn_cast<ttg::AsyncWaitOp>(op)) {
        b.create<tti::ExperimentalClearOutstandingCommitsOp>(
            asyncCpCommitsAlloc, asyncCpCommitsType, asyncWaitOp.getNum(),
            nullptr);
      }
      if (auto expectOp = dyn_cast<ttng::BarrierExpectOp>(op)) {
        if (writeBarriersAlloc[(int)MemType::SHARED_MEM]) {
          b.create<tti::ExperimentalCheckBarrierWritesClearedOp>(
              expectOp.getAlloc(), barriers,
              writeBarriersAlloc[(int)MemType::SHARED_MEM],
              writeBarriersType[(int)MemType::SHARED_MEM], expectOp.getPred());
        }
        if (writeBarriersAlloc[(int)MemType::TENSOR_MEM]) {
          b.create<tti::ExperimentalCheckBarrierWritesClearedOp>(
              expectOp.getAlloc(), barriers,
              writeBarriersAlloc[(int)MemType::TENSOR_MEM],
              writeBarriersType[(int)MemType::TENSOR_MEM], expectOp.getPred());
        }
      }
      if (auto wgmmaOp = dyn_cast<ttng::WarpGroupDotOp>(op)) {
        if (wgmmaOp.getIsAsync() == true) {
          // Add commit (implicit in ttgir) after staging wgmma's operand for
          // read
          b.create<tti::ExperimentalCommitAccessesOp>(
              wgmmaCommitsAlloc, wgmmaCommitsType, nullptr);
        }
      }
      if (auto wgmmaWaitOp = dyn_cast<ttng::WarpGroupDotWaitOp>(op)) {
        b.create<tti::ExperimentalClearOutstandingCommitsOp>(
            wgmmaCommitsAlloc, wgmmaCommitsType, wgmmaWaitOp.getPendings(),
            nullptr);
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

  ttg::BlockedEncodingAttr
  getThreadLocalBlockedEncoding(unsigned int buffers, unsigned int barriers) {
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

  void createZeroInitStateTensor(ImplicitLocOpBuilder &b, int m, int n,
                                 int bitWidth, RankedTensorType &type,
                                 Value &alloc) {
    SmallVector<int64_t> shape = {m};
    if (n > 0) {
      shape.push_back(n);
    }
    ttg::BlockedEncodingAttr encoding = getThreadLocalBlockedEncoding(m);
    if (n > 0) {
      encoding = getThreadLocalBlockedEncoding(m, n);
    }
    type = RankedTensorType::get(shape, b.getIntegerType(bitWidth), encoding);
    TypedValue<RankedTensorType> tensor =
        tti::createConstIntTensor(b, b.getLoc(), 0, type);
    alloc = createInitializedScratchMemory(b, tensor);
  }

  ModuleOp module;

  static constexpr int numMemTypes = getMaxEnumValForMemType() + 1;
  Value buffersTensor[numMemTypes];
  Value barriers;

  /////////////////////////////
  // mbarrier synchronization
  /////////////////////////////

  // WriteVisibility tracks which writes to buffers are visible to which
  // threads. 1D tensor, [num_buffers x i64], elements are bitmasks representing
  // threads
  RankedTensorType writeVisibilityType[numMemTypes];
  Value writeVisibilityAlloc[numMemTypes];

  // WriteTracking describes which writes to buffers are being tracked by which
  // barriers. 2D tensor, [num_buffers x num_barriers x i8], elements are bool
  // flags
  RankedTensorType writeTrackingType[numMemTypes];
  Value writeTrackingAlloc[numMemTypes];

  // ReadVisibility tracks which reads from buffers are visible to which
  // threads. 2D tensor, [num_buffers x num_threads i64], elements are bitmasks
  // representing threads.
  RankedTensorType readVisibilityType[numMemTypes];
  Value readVisibilityAlloc[numMemTypes];

  // ReadTracking describes which reads from buffers are being tracked by which
  // barriers. 2D tensor, [num_buffers x num_barriers x i64], elements are
  // bitmasks representing threads.
  RankedTensorType readTrackingType[numMemTypes];
  Value readTrackingAlloc[numMemTypes];

  /////////////////////////////
  // TODO: legacy, remove
  /////////////////////////////

  // Tensor tracking which barriers are tracking writes to which buffers
  RankedTensorType writeBarriersType[numMemTypes];
  Value writeBarriersAlloc[numMemTypes];

  // Tensor tracking which buffers are being written to at the moment
  RankedTensorType writeStateType[numMemTypes];
  Value writeStateAlloc[numMemTypes];

  // Tensor tracking which barriers are tracking reads from which buffers
  RankedTensorType readBarriersType[numMemTypes];
  Value readBarriersAlloc[numMemTypes];

  /////////////////////////////
  // cp_async synchronization
  // TODO: make it work with WS by supporting threads
  /////////////////////////////

  // Tensor tracking number of outstanding write commits for given buffer
  RankedTensorType asyncCpCommitsType;
  Value asyncCpCommitsAlloc;

  /////////////////////////////
  // wgmma synchronization
  // TODO: make it work with WS by supporting threads
  /////////////////////////////

  // Tensor tracking number of outstanding read commits for given buffer
  RankedTensorType wgmmaCommitsType;
  Value wgmmaCommitsAlloc;
};

} // namespace instrument
} // namespace triton
} // namespace mlir
