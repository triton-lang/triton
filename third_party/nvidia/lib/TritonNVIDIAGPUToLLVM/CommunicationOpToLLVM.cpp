#include "Allocation.h"
#include "Dialect/NVGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

using namespace mlir;
using namespace mlir::triton;
namespace ttng = mlir::triton::nvidia_gpu;
namespace ttg = mlir::triton::gpu;
using Ordering = LLVM::AtomicOrdering;

namespace {
// Only the elected thread runs the protocol; all participating threads join
// before reading its result or accessing the transferred payload.
struct Protocol {
  Protocol(Operation *op, ConversionPatternRewriter &rewriter,
           const NVIDIA::TargetInfo &targetInfo)
      : isWait(isa<ttng::CommunicationWaitOp>(op)), rewriter(rewriter),
        targetInfo(targetInfo), loc(op->getLoc()), b(loc, rewriter) {
    // [Perf] One-warp polls use a vote; blocking waits are faster with scratch.
    warpResult = ttng::isSingleWarpPoll(op) && !op->getResult(0).use_empty();
    Value pred;
    if (isWait)
      skip = b.icmp_ne(getThreadId(rewriter, loc), b.i32_val(0));
    else if (isa<ttng::CommunicationIsAbortedOp>(op)) {
      // [Perf] A fixed query lane lets LLVM move election out of polling loops.
      pred = b.icmp_eq(getThreadId(rewriter, loc), b.i32_val(0));
    } else
      pred = targetInfo.getComputeCapability() >= 90
                 ? LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter)
                 : b.icmp_eq(getThreadId(rewriter, loc), b.i32_val(0));
    reduceResult = !isWait && !warpResult && ttng::isWholeCTA(op) &&
                   !op->getResult(0).use_empty();
    if (!reduceResult && !warpResult && !op->getResult(0).use_empty()) {
      scratch = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
      // Keep scratch addressing outside the elected control flow.
      scratch =
          LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.nvvm.move.ptr",
                                          scratch.getType(), {scratch})
              .getResult(0);
    }
    Block *before = rewriter.getInsertionBlock();
    join = before->splitBlock(rewriter.getInsertionPoint());
    Type resultType = i1_ty;
    if (isWait && !warpResult)
      resultType = VectorType::get({1}, i8_ty);
    if (reduceResult || warpResult || (isWait && scratch))
      joinedResult = join->addArgument(resultType, loc);
    done = rewriter.createBlock(join);
    result = done->addArgument(resultType, loc);
    Block *entry = block();
    rewriter.setInsertionPointToEnd(before);
    if (isWait) {
      LLVM::CondBrOp::create(rewriter, loc, skip, join,
                             joinedResult ? ValueRange{resultValue(false)}
                                          : ValueRange{},
                             entry, ValueRange{});
    } else {
      LLVM::CondBrOp::create(rewriter, loc, pred, entry, ValueRange{}, join,
                             joinedResult ? ValueRange{b.false_val()}
                                          : ValueRange{});
    }
    auto wait = dyn_cast<ttng::CommunicationWaitOp>(op);
    if (wait && !wait.getBlocking()) {
      // [Correctness] Nonblocking waits may run in an outer polling loop.
      // Rotation and peeling can separate election from result broadcast and
      // hang. Put the convergent, noduplicate marker before election so it
      // covers every participant.
      rewriter.setInsertionPoint(before->getTerminator());
      auto markerType =
          LLVM::LLVMFunctionType::get(void_ty(op->getContext()), {});
      auto marker = ttg::appendOrGetExternFuncOp(rewriter, op,
                                                 "llvm.sideeffect", markerType);
      auto markerCall = LLVM::createLLVMCallOp(rewriter, loc, marker, {});
      markerCall.setConvergent(true);
      markerCall.setNoduplicate(true);
    }
    rewriter.setInsertionPointToEnd(entry);
  }

  Block *block() { return rewriter.createBlock(done); }

  Value resultValue(bool value) {
    if (isWait && !warpResult)
      return b.bitcast(b.i8_val(value), VectorType::get({1}, i8_ty));
    return value ? b.true_val() : b.false_val();
  }

  Value load(Value ptr, Ordering order, StringRef scope = {}) {
    return b.load(i64_ty, ptr, /*alignment=*/8, /*isVolatile=*/false,
                  /*isNonTemporal=*/false, /*isInvariant=*/false,
                  /*isInvariantGroup=*/false, order, scope);
  }

  Value isAborted(Value ptr, int64_t abortedValue) {
    return b.icmp_eq(load(ptr, Ordering::acquire), b.i64_val(abortedValue));
  }

  Value finish() {
    rewriter.setInsertionPointToStart(done);
    if (scratch && !isWait)
      b.store(result, scratch, /*alignment=*/1);
    LLVM::BrOp::create(rewriter, loc,
                       joinedResult ? ValueRange{result} : ValueRange{}, join);
    rewriter.setInsertionPointToStart(join);
    if (isWait && scratch) {
      Value mask =
          b.bitcast(b.xor_(skip, b.true_val()), VectorType::get({1}, i1_ty));
      LLVM::MaskedStoreOp::create(rewriter, loc, joinedResult, scratch, mask,
                                  rewriter.getI32IntegerAttr(1), UnitAttr());
    }
    if (reduceResult) {
      Value id = nvgpu::WarpGroupBarrierIdOp::create(rewriter, loc);
      Value reduced = NVVM::BarrierReductionOp::create(
          rewriter, loc, i32_ty, id, NVVM::BarrierReduction::OR,
          b.zext(i32_ty, joinedResult), /*aligned=*/true);
      return b.trunc(i1_ty, reduced);
    }
    targetInfo.barrier(loc, rewriter,
                       ttg::AddrSpace::Local | ttg::AddrSpace::GlobalRead |
                           ttg::AddrSpace::GlobalWrite);
    if (warpResult) {
      return NVVM::VoteSyncOp::create(rewriter, loc, i1_ty, b.i32_val(-1),
                                      joinedResult, NVVM::VoteSyncKind::any);
    }
    return scratch ? b.load(i1_ty, scratch) : b.false_val();
  }

  bool isWait, reduceResult, warpResult;
  ConversionPatternRewriter &rewriter;
  const NVIDIA::TargetInfo &targetInfo;
  Location loc;
  TritonLLVMOpBuilder b;
  Block *done, *join;
  Value skip, result, scratch, joinedResult;
};

template <typename Op>
struct CommunicationConversion : ConvertOpToLLVMPattern<Op> {
  CommunicationConversion(LLVMTypeConverter &converter,
                          const NVIDIA::TargetInfo &targetInfo,
                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Op>(converter, benefit), targetInfo(targetInfo) {
  }
  const NVIDIA::TargetInfo &targetInfo;
};

struct WaitConversion : CommunicationConversion<ttng::CommunicationWaitOp> {
  using CommunicationConversion::CommunicationConversion;

  LogicalResult
  matchAndRewrite(ttng::CommunicationWaitOp op, OpAdaptor a,
                  ConversionPatternRewriter &rewriter) const override {
    // [Perf] Expose uniform counts in multiwarp regions and blocking one-warp
    // CTAs; omit the redundant shuffle in one-warp polls.
    Value count = a.getCount();
    if (op.getKind() != ttng::CommunicationWaitKind::Send &&
        (ttg::lookupNumWarps(op) != 1 ||
         (op.getBlocking() && ttng::isWholeCTA(op)))) {
      auto before = TritonLLVMOpBuilder(op.getLoc(), rewriter);
      Value constant =
          LLVM::IsConstantOp::create(rewriter, op.getLoc(), i1_ty, count);
      count = emitPredicated(
          rewriter, op.getLoc(), before.xor_(constant, before.true_val()),
          ValueRange{count}, [&] {
            return SmallVector<Value>{
                LLVM::NVIDIA::shuffleIdx(op.getLoc(), rewriter, count, 0)};
          })[0];
    }
    Protocol p(op, rewriter, targetInfo);
    auto &b = p.b;
    auto loc = op.getLoc();
    Block *entry = rewriter.getInsertionBlock();
    Block *setup = p.block();
    Block *checkAbort = p.block();
    Block *successBlock = p.block();
    Block *afterAbort = successBlock;
    bool completion = op.getKind() == ttng::CommunicationWaitKind::Send;

    rewriter.setInsertionPointToEnd(entry);
    auto loadTarget = [&]() -> Value {
      // [Perf] A monotonic send-count load avoids an unnecessary acquire and
      // cache invalidation. Callers must finish all submissions and synchronize
      // them with the waiting region, then exclude new submissions during the
      // wait. Atomic coherence suffices to read this stable count. Receive/ACK
      // cursors retain acquire ordering.
      Value cursor = p.load(
          a.getCursor(), completion ? Ordering::monotonic : Ordering::acquire,
          "device");
      if (completion)
        return b.sub(cursor, b.i64_val(1));
      // [Perf] Comparison-only targets let LLVM fold count-one readiness tests.
      auto flags = !op.getBlocking() && !op.getMonitor() && !op.getConsume()
                       ? LLVM::IntegerOverflowFlags::nuw |
                             LLVM::IntegerOverflowFlags::nsw
                       : LLVM::IntegerOverflowFlags::none;
      return b.add(cursor, count, flags);
    };
    auto publishTarget = [&](Value target) {
      b.store(target, a.getMonitorPtr(), /*alignment=*/8,
              /*isVolatile=*/false, /*isNonTemporal=*/false,
              /*isInvariantGroup=*/false, Ordering::release);
    };
    Value target;
    if (op.getMonitor()) {
      target = loadTarget();
      publishTarget(target);
    }
    Value aborted = p.isAborted(a.getAborted(), op.getAbortedValue());
    LLVM::CondBrOp::create(rewriter, loc, aborted, p.done,
                           ValueRange{p.resultValue(false)}, setup,
                           ValueRange{});
    rewriter.setInsertionPointToStart(setup);
    if (!completion) {
      Block *nonzero = p.block();
      rewriter.setInsertionPointToStart(setup);
      LLVM::CondBrOp::create(rewriter, loc, b.icmp_eq(count, b.i64_val(0)),
                             p.done, ValueRange{p.resultValue(true)}, nonzero,
                             ValueRange{});
      rewriter.setInsertionPointToStart(nonzero);
    }
    if (!target)
      target = loadTarget();
    Value first = p.load(a.getCounter(), Ordering::monotonic);
    Value firstReady = b.icmp_sge(first, target);
    if (op.getBlocking()) {
      auto ip = rewriter.saveInsertionPoint();
      Block *poll = rewriter.createBlock(checkAbort);
      Block *checkReady = rewriter.createBlock(successBlock);
      rewriter.restoreInsertionPoint(ip);
      afterAbort = checkReady;
      Value spins = poll->addArgument(i32_ty, loc);
      Value checkedSpins = checkAbort->addArgument(i32_ty, loc);
      Value checkedReady = checkAbort->addArgument(i1_ty, loc);

      // The first miss cannot reach delayed publication or the abort period.
      LLVM::CondBrOp::create(rewriter, loc, firstReady, checkAbort,
                             ValueRange{b.i32_val(0), b.true_val()}, poll,
                             ValueRange{b.i32_val(1)});
      rewriter.setInsertionPointToStart(poll);
      Value current = p.load(a.getCounter(), Ordering::monotonic);
      Value matched = b.icmp_sge(current, target);
      Value next = b.add(spins, b.i32_val(1));
      Value mask = b.i32_val(op.getMonitor() ? 2047 : (2047 ^ 32));
      Value slow = b.icmp_eq(b.and_(next, mask), b.i32_val(0));
      // One cold edge handles either readiness or a periodic event.
      if (op.getMonitor()) {
        LLVM::CondBrOp::create(rewriter, loc, b.or_(matched, slow), checkAbort,
                               ValueRange{next, matched}, poll,
                               ValueRange{next});
      } else {
        Block *event = rewriter.createBlock(checkAbort);
        Block *miss = rewriter.createBlock(checkAbort);
        Block *publish = rewriter.createBlock(checkAbort);
        Block *period = rewriter.createBlock(checkAbort);
        rewriter.setInsertionPointToEnd(poll);
        LLVM::CondBrOp::create(rewriter, loc, b.or_(matched, slow), event,
                               ValueRange{}, poll, ValueRange{next});
        rewriter.setInsertionPointToStart(event);
        LLVM::CondBrOp::create(rewriter, loc, matched, checkAbort,
                               ValueRange{next, b.true_val()}, miss,
                               ValueRange{});
        rewriter.setInsertionPointToStart(miss);
        LLVM::CondBrOp::create(rewriter, loc, b.icmp_eq(next, b.i32_val(32)),
                               publish, period);
        rewriter.setInsertionPointToStart(publish);
        publishTarget(target);
        LLVM::BrOp::create(rewriter, loc, ValueRange{next}, poll);
        rewriter.setInsertionPointToStart(period);
        Value check = b.icmp_eq(b.and_(next, b.i32_val(2047)), b.i32_val(0));
        LLVM::CondBrOp::create(rewriter, loc, check, checkAbort,
                               ValueRange{next, b.false_val()}, poll,
                               ValueRange{next});
      }
      rewriter.setInsertionPointToStart(checkReady);
      LLVM::CondBrOp::create(rewriter, loc, checkedReady, successBlock,
                             ValueRange{}, poll, ValueRange{checkedSpins});
    } else {
      LLVM::CondBrOp::create(rewriter, loc, firstReady, checkAbort,
                             ValueRange{}, p.done,
                             ValueRange{p.resultValue(false)});
    }

    rewriter.setInsertionPointToStart(checkAbort);
    aborted = p.isAborted(a.getAborted(), op.getAbortedValue());
    LLVM::CondBrOp::create(rewriter, loc, aborted, p.done,
                           ValueRange{p.resultValue(false)}, afterAbort,
                           ValueRange{});
    rewriter.setInsertionPointToStart(successBlock);
    if (op.getKind() == ttng::CommunicationWaitKind::Recv)
      p.load(a.getCounter(), Ordering::acquire);
    if (op.getConsume())
      b.store(target, a.getCursor(), /*alignment=*/8,
              /*isVolatile=*/false, /*isNonTemporal=*/false,
              /*isInvariantGroup=*/false, Ordering::release, "device");
    LLVM::BrOp::create(rewriter, loc, ValueRange{p.resultValue(true)}, p.done);
    rewriter.replaceOp(op, p.finish());
    return success();
  }
};

struct SubmitConversion : CommunicationConversion<ttng::CommunicationSubmitOp> {
  using CommunicationConversion::CommunicationConversion;

  LogicalResult
  matchAndRewrite(ttng::CommunicationSubmitOp op, OpAdaptor a,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Order every participating thread's payload stores and acknowledgement
    // reads.
    targetInfo.barrier(loc, rewriter,
                       ttg::AddrSpace::Local | ttg::AddrSpace::GlobalRead |
                           ttg::AddrSpace::GlobalWrite);
    auto layout = op.getRequestLayout();
    Protocol p(op, rewriter, targetInfo);
    auto &b = p.b;
    Block *entry = rewriter.getInsertionBlock();
    Block *reserve = p.block();
    Block *refresh = p.block();
    Block *updateHead = p.block();
    Block *claim = p.block();
    Block *publish = p.block();
    rewriter.setInsertionPointToEnd(entry);
    Value aborted = p.isAborted(a.getAborted(), op.getAbortedValue());
    LLVM::CondBrOp::create(rewriter, loc, aborted, p.done,
                           ValueRange{p.resultValue(false)}, reserve,
                           ValueRange{});
    rewriter.setInsertionPointToStart(reserve);
    Value cachedHead = p.load(a.getCachedHead(), Ordering::acquire, "device");
    Value tail = p.load(a.getTail(), Ordering::monotonic, "device");
    // A stale head can lag by more than a ring; avoid free-space underflow.
    Value occupancy = b.sub(tail, cachedHead);
    Value limit = b.i64_val(op.getCapacity() - 1);
    // [Perf] Keep successful reservations contiguous, ahead of retry blocks.
    auto available = LLVM::CondBrOp::create(
        rewriter, loc, b.icmp_ult(occupancy, limit), claim, refresh);
    available.setBranchWeightsAttr(rewriter.getDenseI32ArrayAttr({2000, 1}));
    rewriter.setInsertionPointToStart(refresh);
    Value head = p.load(a.getHead(), Ordering::acquire);
    // [Perf] Skip publication when the acquired cache already covers this head.
    LLVM::CondBrOp::create(rewriter, loc, b.icmp_ugt(head, cachedHead),
                           updateHead, entry);
    rewriter.setInsertionPointToStart(updateHead);
    // [Correctness] Relay the CPU's completed reads and ready-word clears to
    // other producers before they reuse slots through cached_head.
    LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::umax,
                              a.getCachedHead(), head, Ordering::release,
                              "device");
    LLVM::BrOp::create(rewriter, loc, entry);
    rewriter.setInsertionPointToStart(claim);
    Value cas = LLVM::AtomicCmpXchgOp::create(
        rewriter, loc, a.getTail(), tail, b.add(tail, b.i64_val(1)),
        Ordering::monotonic, Ordering::monotonic, "device");
    auto claimed = LLVM::CondBrOp::create(
        rewriter, loc, b.extract_val(i1_ty, cas, 1), publish, entry);
    claimed.setBranchWeightsAttr(rewriter.getDenseI32ArrayAttr({2000, 1}));

    rewriter.setInsertionPointToStart(publish);
    // A reserved slot must be published even if abort races with the CAS.
    Value index = b.urem(tail, b.i64_val(op.getCapacity()));
    Value slot = b.mul(index, b.i64_val(layout.getStride()));
    Value ptr = b.gep(a.getBuffer().getType(), i8_ty, a.getBuffer(), slot);
    auto store = [&](int offset, Value value,
                     Ordering order = Ordering::not_atomic) {
      Value field = b.gep(ptr.getType(), i8_ty, ptr, b.i64_val(offset));
      b.store(value, field, value.getType().getIntOrFloatBitWidth() / 8,
              /*isVolatile=*/false, /*isNonTemporal=*/false,
              /*isInvariantGroup=*/false, order);
    };
    store(layout.getTypeOffset(), b.i32_val(op.getRequestType()));
    store(layout.getHandleOffset(), a.getHandle());
    if (op.getIsSend()) {
      store(layout.getSrcOffset(), a.getSrcOffset());
      store(layout.getDstOffset(), a.getDstOffset());
      store(layout.getLengthOffset(), a.getNbytes());
      store(layout.getBypassArOffset(), b.i32_val(op.getBypassValue()));
    }
    store(layout.getReadyOffset(), b.i64_val(op.getReadyValue()),
          Ordering::release);
    if (op.getIsSend())
      LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::add,
                                a.getSendCount(), b.i64_val(1),
                                Ordering::monotonic, "device");
    LLVM::BrOp::create(rewriter, loc, ValueRange{p.resultValue(true)}, p.done);
    rewriter.replaceOp(op, p.finish());
    return success();
  }
};

struct IsAbortedConversion
    : CommunicationConversion<ttng::CommunicationIsAbortedOp> {
  using CommunicationConversion::CommunicationConversion;

  LogicalResult
  matchAndRewrite(ttng::CommunicationIsAbortedOp op, OpAdaptor a,
                  ConversionPatternRewriter &rewriter) const override {
    Protocol p(op, rewriter, targetInfo);
    Value aborted = p.isAborted(a.getAborted(), op.getAbortedValue());
    LLVM::BrOp::create(rewriter, op.getLoc(), ValueRange{aborted}, p.done);
    rewriter.replaceOp(op, p.finish());
    return success();
  }
};
} // namespace

void mlir::triton::NVIDIA::populateCommunicationOpToLLVMPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    PatternBenefit benefit, const NVIDIA::TargetInfo &targetInfo) {
  patterns.add<WaitConversion, SubmitConversion, IsAbortedConversion>(
      converter, targetInfo, benefit);
}
