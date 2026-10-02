#include "Dialect/NVGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;
namespace ttng = mlir::triton::nvidia_gpu;
namespace ttg = mlir::triton::gpu;
using Ordering = LLVM::AtomicOrdering;

namespace {
// Aligned reductions require every thread in the CTA.
bool isWholeCTA(Operation *op) {
  auto totalWarps = op->getParentOfType<ModuleOp>()->getAttrOfType<IntegerAttr>(
      "ttg.total-num-warps");
  return totalWarps && totalWarps.getInt() == ttg::lookupNumWarps(op);
}

// The elected thread owns the protocol. Share its result only after it leaves
// the polling loop, so other warps never participate in that loop.
Value runProtocol(Operation *op, ConversionPatternRewriter &rewriter,
                  const NVIDIA::TargetInfo &targetInfo, StringRef code,
                  ValueRange inputs, ArrayRef<StringRef> constraints) {
  auto loc = op->getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value pred = b.icmp_eq(getThreadId(rewriter, loc), b.i32_val(0));
  PTXBuilder ptx;
  SmallVector<PTXBuilder::Operand *> args{ptx.newOperand("=r"),
                                          ptx.newOperand(pred, "b")};
  for (auto [input, constraint] : llvm::zip_equal(inputs, constraints))
    args.push_back(ptx.newOperand(input, constraint));
  (*ptx.create(code.str()))(args, /*onlyAttachMLIRArgs=*/true);
  Value result = b.trunc(i1_ty, ptx.launch(rewriter, loc, i32_ty));
  Value scratch;
  if (!op->getResult(0).use_empty()) {
    scratch = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
    targetInfo.storeShared(rewriter, loc, scratch, result, pred);
  }
  targetInfo.barrier(loc, rewriter,
                     ttg::AddrSpace::Local | ttg::AddrSpace::GlobalRead |
                         ttg::AddrSpace::GlobalWrite);
  return scratch ? b.load(i1_ty, scratch) : b.false_val();
}

// Only the elected thread runs the protocol; all participating threads join
// before reading its result or accessing the transferred payload.
struct Protocol {
  Protocol(Operation *op, ConversionPatternRewriter &rewriter,
           const NVIDIA::TargetInfo &targetInfo)
      : rewriter(rewriter), targetInfo(targetInfo), loc(op->getLoc()),
        b(loc, rewriter) {
    pred = targetInfo.getComputeCapability() >= 90
               ? LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter)
               : b.icmp_eq(getThreadId(rewriter, loc), b.i32_val(0));
    reduceResult = isWholeCTA(op) && !op->getResult(0).use_empty();
    if (!reduceResult && !op->getResult(0).use_empty())
      scratch = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
    Block *before = rewriter.getInsertionBlock();
    join = before->splitBlock(rewriter.getInsertionPoint());
    if (reduceResult)
      joinedResult = join->addArgument(i1_ty, loc);
    done = rewriter.createBlock(join);
    result = done->addArgument(i1_ty, loc);
    Block *entry = block();
    rewriter.setInsertionPointToEnd(before);
    if (reduceResult)
      LLVM::CondBrOp::create(rewriter, loc, pred, entry, ValueRange{}, join,
                             ValueRange{b.false_val()});
    else
      LLVM::CondBrOp::create(rewriter, loc, pred, entry, join);
    rewriter.setInsertionPointToStart(entry);
  }

  Block *block() { return rewriter.createBlock(done); }

  Value load(Value ptr, Ordering order, StringRef scope = {}) {
    return b.load(i64_ty, ptr, /*alignment=*/8, /*isVolatile=*/false,
                  /*isNonTemporal=*/false, /*isInvariant=*/false,
                  /*isInvariantGroup=*/false, order, scope);
  }

  Value finish() {
    rewriter.setInsertionPointToStart(done);
    if (scratch)
      targetInfo.storeShared(rewriter, loc, scratch, result, pred);
    LLVM::BrOp::create(rewriter, loc,
                       reduceResult ? ValueRange{result} : ValueRange{}, join);
    rewriter.setInsertionPointToStart(join);
    if (reduceResult) {
      Value id = nvgpu::WarpGroupBarrierIdOp::create(rewriter, loc);
      return LLVM::createLLVMIntrinsicCallOp(
                 rewriter, loc, "llvm.nvvm.barrier.cta.red.or.aligned.all",
                 i1_ty, {id, joinedResult})
          .getResult(0);
    }
    targetInfo.barrier(loc, rewriter,
                       ttg::AddrSpace::Local | ttg::AddrSpace::GlobalRead |
                           ttg::AddrSpace::GlobalWrite);
    return scratch ? b.load(i1_ty, scratch) : b.false_val();
  }

  bool reduceResult;
  ConversionPatternRewriter &rewriter;
  const NVIDIA::TargetInfo &targetInfo;
  Location loc;
  TritonLLVMOpBuilder b;
  Block *done, *join;
  Value pred, result, scratch, joinedResult;
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
    // LLVM loop rewrites increased polling cost and ready-wait latency.
    bool completion = op.getKind() == "send";
    std::string abortValue = std::to_string(op.getAbortedValue());
    std::string code = R"({
      .reg .pred p, ready, aborted;
      .reg .b64 target, current, state;
      .reg .u32 spins, remainder;
      .reg .u32 result;
      mov.u32 result, 0;
      @!$1 bra done;
      ld.acquire.sys.global.b64 state, [$4];
      setp.eq.u64 aborted, state, )" +
                       abortValue + R"(;
      @aborted bra done;
    )";
    if (!completion)
      code += R"(
        setp.eq.u64 p, $6, 0;
        @p bra success;
      )";
    code += "ld.acquire.gpu.global.b64 target, [$3];\n";
    code += completion ? "sub.u64 target, target, 1;\n"
                       : "add.u64 target, target, $6;\n";
    code += R"(
      mov.u32 spins, 0;
    poll:
      ld.relaxed.sys.global.b64 current, [$2];
    )";
    code += completion ? "setp.ge.s64 ready, current, target;\n"
                       : "setp.ge.u64 ready, current, target;\n";
    code += R"(
      @ready bra check_abort;
      add.u32 spins, spins, 1;
      setp.eq.u32 p, spins, 32;
      @p st.release.sys.global.b64 [$5], target;
      and.b32 remainder, spins, 2047;
      setp.ne.u32 p, remainder, 0;
      @p bra poll;
    check_abort:
      ld.acquire.sys.global.b64 state, [$4];
      setp.eq.u64 aborted, state, )" +
            abortValue + R"(;
      @aborted bra done;
      @!ready bra poll;
    )";
    if (op.getAcquire())
      code += "ld.acquire.sys.global.b64 current, [$2];\n";
    if (op.getConsume())
      code += "red.release.gpu.global.max.u64 [$3], target;\n";
    code += R"(
    success:
      mov.u32 result, 1;
    done:
      mov.u32 $0, result;
    })";
    rewriter.replaceOp(
        op, runProtocol(op, rewriter, targetInfo, code,
                        {a.getCounter(), a.getCursor(), a.getAborted(),
                         a.getMonitor(), a.getCount()},
                        {"l", "l", "l", "l", "l"}));
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
    // LLVM folds constant-capacity remainders; the original PTX is faster for
    // runtime capacities. This intrinsic removes the unused path before
    // codegen.
    Value constant =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.is.constant.i64",
                                        i1_ty, {a.getCapacity()})
            .getResult(0);
    Block *before = rewriter.getInsertionBlock();
    Block *done = before->splitBlock(rewriter.getInsertionPoint());
    Value result = done->addArgument(i1_ty, loc);
    Block *known = rewriter.createBlock(done);
    Block *dynamic = rewriter.createBlock(done);
    rewriter.setInsertionPointToEnd(before);
    LLVM::CondBrOp::create(rewriter, loc, constant, known, dynamic);
    rewriter.setInsertionPointToStart(known);
    Value value = emitLLVM(op, a, rewriter);
    LLVM::BrOp::create(rewriter, loc, ValueRange{value}, done);
    rewriter.setInsertionPointToStart(dynamic);
    value = emitPTX(op, a, rewriter);
    LLVM::BrOp::create(rewriter, loc, ValueRange{value}, done);
    rewriter.setInsertionPointToStart(done);
    rewriter.replaceOp(op, result);
    return success();
  }

  Value emitLLVM(ttng::CommunicationSubmitOp op, OpAdaptor a,
                 ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto layout = op.getRequestLayout();
    Protocol p(op, rewriter, targetInfo);
    auto &b = p.b;
    Block *entry = rewriter.getInsertionBlock();
    Block *reserve = p.block();
    Block *refresh = p.block();
    Block *claim = p.block();
    Block *publish = p.block();
    rewriter.setInsertionPointToStart(entry);
    Value aborted = b.icmp_eq(p.load(a.getAborted(), Ordering::acquire),
                              b.i64_val(op.getAbortedValue()));
    LLVM::CondBrOp::create(rewriter, loc, aborted, p.done,
                           ValueRange{b.false_val()}, reserve, ValueRange{});
    rewriter.setInsertionPointToStart(reserve);
    Value head = p.load(a.getCachedHead(), Ordering::acquire, "device");
    Value tail = p.load(a.getTail(), Ordering::monotonic, "device");
    Value free = b.sub(a.getCapacity(), b.sub(tail, head));
    LLVM::CondBrOp::create(rewriter, loc, b.icmp_ugt(free, b.i64_val(1)), claim,
                           refresh);
    rewriter.setInsertionPointToStart(refresh);
    head = p.load(a.getHead(), Ordering::acquire);
    LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::umax,
                              a.getCachedHead(), head, Ordering::monotonic,
                              "device");
    LLVM::BrOp::create(rewriter, loc, entry);
    rewriter.setInsertionPointToStart(claim);
    Value cas = LLVM::AtomicCmpXchgOp::create(
        rewriter, loc, a.getTail(), tail, b.add(tail, b.i64_val(1)),
        Ordering::monotonic, Ordering::monotonic, "device");
    LLVM::CondBrOp::create(rewriter, loc, b.extract_val(i1_ty, cas, 1), publish,
                           entry);

    rewriter.setInsertionPointToStart(publish);
    // A reserved slot must be published even if abort races with the CAS.
    Value slot = b.mul(b.urem(tail, a.getCapacity()), b.i64_val(layout[0]));
    Value ptr = b.gep(a.getBuffer().getType(), i8_ty, a.getBuffer(), slot);
    auto store = [&](int offset, Value value,
                     Ordering order = Ordering::not_atomic) {
      Value field = b.gep(ptr.getType(), i8_ty, ptr, b.i64_val(offset));
      b.store(value, field, value.getType().getIntOrFloatBitWidth() / 8,
              /*isVolatile=*/false, /*isNonTemporal=*/false,
              /*isInvariantGroup=*/false, order);
    };
    store(layout[2], b.i32_val(op.getRequestType()));
    store(layout[3], a.getHandle());
    if (op.getIsSend()) {
      store(layout[4], a.getSrcOffset());
      store(layout[5], a.getDstOffset());
      store(layout[6], a.getNbytes());
      store(layout[7], b.i32_val(op.getBypassValue()));
    }
    store(layout[1], b.i64_val(op.getReadyValue()), Ordering::release);
    if (op.getIsSend())
      LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::add,
                                a.getSendCount(), b.i64_val(1),
                                Ordering::monotonic, "device");
    LLVM::BrOp::create(rewriter, loc, ValueRange{b.true_val()}, p.done);
    return p.finish();
  }

  Value emitPTX(ttng::CommunicationSubmitOp op, OpAdaptor a,
                ConversionPatternRewriter &rewriter) const {
    auto layout = op.getRequestLayout();
    std::string code = R"({
      .reg .pred p;
      .reg .b64 head, tail, free, next, old, slot, state, ready;
      .reg .u32 request_type, bypass;
      .reg .u32 result;
      mov.u32 result, 0;
      @!$1 bra done;
    reserve:
      ld.acquire.sys.global.b64 state, [$12];
      setp.eq.u64 p, state, )" +
                       std::to_string(op.getAbortedValue()) + R"(;
      @p bra done;
      ld.acquire.gpu.global.b64 head, [$4];
      ld.relaxed.gpu.global.b64 tail, [$3];
      sub.u64 free, tail, head;
      sub.u64 free, $6, free;
      setp.gt.u64 p, free, 1;
      @p bra claim;
      ld.acquire.sys.global.b64 head, [$2];
      red.relaxed.gpu.global.max.u64 [$4], head;
      bra reserve;
    claim:
      add.u64 next, tail, 1;
      atom.relaxed.gpu.global.cas.b64 old, [$3], tail, next;
      setp.ne.u64 p, old, tail;
      @p bra reserve;
      // A reserved slot must be published even if abort races with the CAS.
      rem.u64 slot, tail, $6;
      mad.lo.u64 slot, slot, )" +
                       std::to_string(layout[0]) + R"(, $5;
    )";
    auto store = [&](StringRef width, int offset, StringRef value,
                     StringRef order = "") {
      code += "st." + order.str() + "global." + width.str() + " [slot+" +
              std::to_string(offset) + "], " + value.str() + ";\n";
    };
    code +=
        "mov.u32 request_type, " + std::to_string(op.getRequestType()) + ";\n";
    store("u32", layout[2], "request_type");
    store("u32", layout[3], "$7");
    if (op.getIsSend()) {
      store("u64", layout[4], "$8");
      store("u64", layout[5], "$9");
      store("u64", layout[6], "$10");
      code += "mov.u32 bypass, " + std::to_string(op.getBypassValue()) + ";\n";
      store("u32", layout[7], "bypass");
    }
    code += "mov.u64 ready, " +
            std::to_string(static_cast<uint64_t>(op.getReadyValue())) + ";\n";
    store("u64", layout[1], "ready", "release.sys.");
    if (op.getIsSend())
      code += "red.relaxed.gpu.global.add.u64 [$11], 1;\n";
    code += R"(
      mov.u32 result, 1;
    done:
      mov.u32 $0, result;
    })";
    return runProtocol(op, rewriter, targetInfo, code,
                       {a.getHead(), a.getTail(), a.getCachedHead(),
                        a.getBuffer(), a.getCapacity(), a.getHandle(),
                        a.getSrcOffset(), a.getDstOffset(), a.getNbytes(),
                        a.getSendCount(), a.getAborted()},
                       {"l", "l", "l", "l", "l", "r", "l", "l", "l", "l", "l"});
  }
};

struct IsAbortedConversion
    : CommunicationConversion<ttng::CommunicationIsAbortedOp> {
  using CommunicationConversion::CommunicationConversion;

  LogicalResult
  matchAndRewrite(ttng::CommunicationIsAbortedOp op, OpAdaptor a,
                  ConversionPatternRewriter &rewriter) const override {
    Protocol p(op, rewriter, targetInfo);
    Value aborted = p.b.icmp_eq(p.load(a.getAborted(), Ordering::acquire),
                                p.b.i64_val(op.getAbortedValue()));
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
