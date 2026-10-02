#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;
namespace ttng = mlir::triton::nvidia_gpu;
namespace ttg = mlir::triton::gpu;

namespace {
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
    auto layout = op.getRequestLayout();
    // The release publication also orders payload stores and acknowledgement
    // readers from every participating thread, via this rendezvous.
    targetInfo.barrier(op.getLoc(), rewriter,
                       ttg::AddrSpace::Local | ttg::AddrSpace::GlobalRead |
                           ttg::AddrSpace::GlobalWrite);
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
    rewriter.replaceOp(
        op,
        runProtocol(op, rewriter, targetInfo, code,
                    {a.getHead(), a.getTail(), a.getCachedHead(), a.getBuffer(),
                     a.getCapacity(), a.getHandle(), a.getSrcOffset(),
                     a.getDstOffset(), a.getNbytes(), a.getSendCount(),
                     a.getAborted()},
                    {"l", "l", "l", "l", "l", "r", "l", "l", "l", "l", "l"}));
    return success();
  }
};

struct IsAbortedConversion
    : CommunicationConversion<ttng::CommunicationIsAbortedOp> {
  using CommunicationConversion::CommunicationConversion;

  LogicalResult
  matchAndRewrite(ttng::CommunicationIsAbortedOp op, OpAdaptor a,
                  ConversionPatternRewriter &rewriter) const override {
    std::string code = R"({
      .reg .pred p;
      .reg .b64 state;
      .reg .u32 result;
      mov.u32 result, 0;
      @!$1 bra done;
      ld.acquire.sys.global.b64 state, [$2];
      setp.eq.u64 p, state, )" +
                       std::to_string(op.getAbortedValue()) + R"(;
      selp.u32 result, 1, 0, p;
    done:
      mov.u32 $0, result;
    })";
    rewriter.replaceOp(op, runProtocol(op, rewriter, targetInfo, code,
                                       {a.getAborted()}, {"l"}));
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
