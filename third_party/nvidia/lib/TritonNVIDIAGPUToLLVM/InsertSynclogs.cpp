#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>

namespace mlir::triton {
#define GEN_PASS_DEF_INSERTSYNCLOGS
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// Array of NVVM/PTX op prefixes.
const llvm::SmallVector<const char *, 25> opNames{
    "barrier.sync",
    "nvvm.barrier0",
    "cp.async.bulk.commit.group",
    "cp.async.commit.group",
    "cp.async.bulk.wait.group",
    "nvvm.cp.async.bulk.wait_group",
    "cp.async.wait.group",
    "cp.async.bulk.tensor",
    "cp.reduce.async.bulk.tensor",
    "fence.proxy",
    "tensormap.cp_fenceproxy",
    "fence.mbarrier_init",
    "wgmma.fence",
    "wgmma.commit_group",
    "wgmma.wait_group",
    "mbarrier.init",
    "mbarrier.wait",
    "mbarrier.arrive",
    "mbarrier.inval",
    "mbarrier.expect_tx",
    "mbarrier.test_wait",
    "mbarrier.try_wait",
    "tcgen05.wait",
    "tcgen05.commit",
    "cp.async.mbarrier.arrive"};

void writeLogToBuffer(IRRewriter &rewriter, ModuleOp mod,
                      LLVM::LLVMFuncOp kernelFunc, Operation *op,
                      Value synclogBuffer) {
  auto ctx = rewriter.getContext();
  auto loc = kernelFunc->getLoc();
  TritonLLVMOpBuilder llvmBuilder(loc, rewriter);
  auto i32Type = type::i32Ty(ctx);
  auto i64Type = type::i64Ty(ctx);
  auto globalPtrTy = LLVM::LLVMPointerType::get(ctx, 1);

  StringRef writeSynclogFnName = "write_synclog";
  LLVM::LLVMFuncOp writeSynclogFn =
      mod.lookupSymbol<LLVM::LLVMFuncOp>(writeSynclogFnName);
  if (!writeSynclogFn) {
    rewriter.setInsertionPoint(kernelFunc);
    writeSynclogFn = rewriter.create<LLVM::LLVMFuncOp>(
        mod.getLoc(), writeSynclogFnName,
        LLVM::LLVMFunctionType::get(void_ty(ctx), {globalPtrTy}));
    writeSynclogFn.setPrivate();

    auto *funcBlock = writeSynclogFn.addEntryBlock(rewriter);
    rewriter.setInsertionPointToEnd(funcBlock);
    auto returnOp = rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});
    rewriter.setInsertionPointToStart(funcBlock);

    auto synclogBufferPtr = funcBlock->getArgument(0);

    // Split global timer into 32-bit low and high parts.
    auto time64 = rewriter
                      .create<LLVM::InlineAsmOp>(
                          loc, i64Type, ValueRange{},       // operands
                          "mov.u64 $0, %globaltimer;",      // asm_string
                          llvm::formatv("=l").str().data(), // constraints
                          true,                             // has_side_effects
                          false,                            // is_align_stack
                          LLVM::TailCallKind::None,         // tail_call_kind
                          LLVM::AsmDialectAttr::get(
                              ctx,
                              LLVM::AsmDialect::AD_ATT), // asm_dialect
                          ArrayAttr()                    // operand_attrs
                          )
                      .getRes();
    auto time_lo = llvmBuilder.trunc(i32Type, time64);
    auto const32 = llvmBuilder.i64_val(32);
    auto time_hi_untrunc = llvmBuilder.lshr(time64, const32);
    auto time_hi = llvmBuilder.trunc(i32Type, time_hi_untrunc);

    // Get block/thread indices and cta rank.
    auto blockIdx_x = rewriter.create<NVVM::BlockIdXOp>(loc, i32Type);
    auto threadIdx_x = rewriter.create<NVVM::ThreadIdXOp>(loc, i32Type);
    auto blockIdx_y = rewriter.create<NVVM::BlockIdYOp>(loc, i32Type);
    auto threadIdx_y = rewriter.create<NVVM::ThreadIdYOp>(loc, i32Type);
    auto blockIdx_z = rewriter.create<NVVM::BlockIdZOp>(loc, i32Type);
    auto threadIdx_z = rewriter.create<NVVM::ThreadIdZOp>(loc, i32Type);
    auto ctaRank = rewriter.create<NVVM::ClusterId>(loc, i32Type);

    // Only emit synclog if threadIdx_x % 32 == 0, and threadIdx_y, threadIdx_z,
    // blockIdx_x, blockIdx_y, blockIdx_z are all 0. This is to keep output
    // smaller for parsing.
    auto const32_i32 = llvmBuilder.i32_val(32);
    auto tidXMod32 = llvmBuilder.urem(threadIdx_x, const32_i32);
    auto const0 = llvmBuilder.i32_val(0);
    auto cond1 = llvmBuilder.icmp_eq(tidXMod32, const0);
    auto cond2 = llvmBuilder.icmp_eq(threadIdx_y, const0);
    auto cond3 = llvmBuilder.icmp_eq(threadIdx_z, const0);
    auto cond4 = llvmBuilder.icmp_eq(blockIdx_x, const0);
    auto cond5 = llvmBuilder.icmp_eq(blockIdx_y, const0);
    auto cond6 = llvmBuilder.icmp_eq(blockIdx_z, const0);
    auto and1 = llvmBuilder.and_(cond1, cond2);
    auto and2 = llvmBuilder.and_(cond3, cond4);
    auto and3 = llvmBuilder.and_(cond5, cond6);
    auto and4 = llvmBuilder.and_(and1, and2);
    auto and5 = llvmBuilder.and_(and3, and4);

    auto *prevBlock = rewriter.getInsertionBlock();
    auto *ifBlock =
        rewriter.splitBlock(prevBlock, rewriter.getInsertionPoint());
    auto *thenBlock = rewriter.splitBlock(ifBlock, returnOp->getIterator());
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<LLVM::CondBrOp>(loc, and5, ifBlock, thenBlock);
    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<LLVM::BrOp>(loc, thenBlock);
    rewriter.setInsertionPointToStart(ifBlock);

    // Atomically allocate space in synclog buffer.
    auto const9 = llvmBuilder.i32_val(9);
    auto last = rewriter.create<LLVM::AtomicRMWOp>(
        loc, LLVM::AtomicBinOp::add, synclogBufferPtr, const9,
        LLVM::AtomicOrdering::monotonic);
    auto const1 = llvmBuilder.i32_val(1);
    auto offset = llvmBuilder.add(const1, last);
    auto synclogPtr = llvmBuilder.gep(globalPtrTy, i32Type, synclogBufferPtr,
                                      offset.getResult());

    // Write to buffer.
    auto storeInBuffer = [&](Value valToStore, int index) {
      auto ptr = llvmBuilder.gep(globalPtrTy, i32Type, synclogPtr,
                                 llvmBuilder.i32_val(index));
      llvmBuilder.store(valToStore, ptr);
    };
    storeInBuffer(time_lo, 0);
    storeInBuffer(time_hi, 1);
    storeInBuffer(threadIdx_x, 2);
    storeInBuffer(threadIdx_y, 3);
    storeInBuffer(threadIdx_z, 4);
    storeInBuffer(blockIdx_x, 5);
    storeInBuffer(blockIdx_y, 6);
    storeInBuffer(blockIdx_z, 7);
    storeInBuffer(ctaRank, 8);
  }
  rewriter.setInsertionPoint(op);
  rewriter.create<LLVM::CallOp>(loc, writeSynclogFn, ValueRange{synclogBuffer});
}

void printSynclog(IRRewriter &rewriter, NVIDIA::TargetInfo &targetInfo,
                  LLVM::LLVMFuncOp kernelFunc, Value synclogBuffer) {
  auto returnOp = *kernelFunc.getOps<LLVM::ReturnOp>().begin();
  rewriter.setInsertionPoint(returnOp);
  auto ctx = rewriter.getContext();
  auto loc = returnOp->getLoc();
  TritonLLVMOpBuilder llvmBuilder(loc, rewriter);
  auto i32Type = type::i32Ty(ctx);
  auto i64Type = type::i64Ty(ctx);
  auto globalPtrTy = LLVM::LLVMPointerType::get(ctx, 1);

  // Get block/thread indices and cta rank.
  auto currentBlockIdx_x = rewriter.create<NVVM::BlockIdXOp>(loc, i32Type);
  auto currentThreadIdx_x = rewriter.create<NVVM::ThreadIdXOp>(loc, i32Type);
  auto currentBlockIdx_y = rewriter.create<NVVM::BlockIdYOp>(loc, i32Type);
  auto currentThreadIdx_y = rewriter.create<NVVM::ThreadIdYOp>(loc, i32Type);
  auto currentBlockIdx_z = rewriter.create<NVVM::BlockIdZOp>(loc, i32Type);
  auto currentThreadIdx_z = rewriter.create<NVVM::ThreadIdZOp>(loc, i32Type);

  // Only one thread will dump the synclog buffer (threadIdx_x, threadIdx_y,
  // threadIdx_z, blockIdx_x, blockIdx_y, blockIdx_z = 0).
  auto const0 = llvmBuilder.i32_val(0);
  auto cond1 = llvmBuilder.icmp_eq(currentThreadIdx_x, const0);
  auto cond2 = llvmBuilder.icmp_eq(currentThreadIdx_y, const0);
  auto cond3 = llvmBuilder.icmp_eq(currentThreadIdx_z, const0);
  auto cond4 = llvmBuilder.icmp_eq(currentBlockIdx_x, const0);
  auto cond5 = llvmBuilder.icmp_eq(currentBlockIdx_y, const0);
  auto cond6 = llvmBuilder.icmp_eq(currentBlockIdx_z, const0);
  auto and1 = llvmBuilder.and_(cond1, cond2);
  auto and2 = llvmBuilder.and_(cond3, cond4);
  auto and3 = llvmBuilder.and_(cond5, cond6);
  auto and4 = llvmBuilder.and_(and1, and2);
  auto and5 = llvmBuilder.and_(and3, and4);

  auto *prevBlock = rewriter.getInsertionBlock();
  auto *ifBlock = rewriter.splitBlock(prevBlock, rewriter.getInsertionPoint());
  auto *thenBlock = rewriter.splitBlock(ifBlock, returnOp->getIterator());
  rewriter.setInsertionPointToEnd(prevBlock);
  rewriter.create<LLVM::CondBrOp>(loc, and5, ifBlock, thenBlock);
  rewriter.setInsertionPointToStart(ifBlock);

  // Load from buffer.
  auto loadFromBuffer = [&](Value baseIndex, int offset) {
    auto index = llvmBuilder.add(baseIndex, llvmBuilder.i32_val(offset));
    auto ptr =
        llvmBuilder.gep(globalPtrTy, i32Type, synclogBuffer, index.getResult());
    return llvmBuilder.load(i32Type, ptr);
  };

  // Initialize loop variables. synclog_buffer[0] is the length of the buffer.
  auto synclogLength = llvmBuilder.load(i32Type, synclogBuffer);
  auto const1 = llvmBuilder.i32_val(1);
  auto indexPtr =
      rewriter.create<LLVM::AllocaOp>(loc, globalPtrTy, i32Type, const1);
  llvmBuilder.store(const1, indexPtr);
  auto *loopConditionBlock =
      rewriter.splitBlock(ifBlock, rewriter.getInsertionPoint());
  rewriter.setInsertionPointToEnd(ifBlock);
  rewriter.create<LLVM::BrOp>(loc, loopConditionBlock);

  // Loop condition. Loop until index >= synclogLength.
  rewriter.setInsertionPointToStart(loopConditionBlock);
  auto index = llvmBuilder.load(i32Type, indexPtr);
  auto cond = llvmBuilder.icmp_slt(index, synclogLength);
  auto *loopBodyBlock =
      rewriter.splitBlock(loopConditionBlock, rewriter.getInsertionPoint());
  rewriter.setInsertionPointToEnd(loopConditionBlock);
  rewriter.create<LLVM::CondBrOp>(loc, cond, loopBodyBlock, thenBlock);

  // Loop body. Iterate through buffer 9 elements at a time.
  rewriter.setInsertionPointToStart(loopBodyBlock);
  auto time_lo = loadFromBuffer(index, 0);
  auto time_hi = loadFromBuffer(index, 1);
  auto const32 = llvmBuilder.i64_val(32);
  auto time_lo_ext = llvmBuilder.zext(i64Type, time_lo);
  auto time_hi_ext = llvmBuilder.zext(i64Type, time_hi);
  auto time_hi_shl = llvmBuilder.shl(time_hi_ext, const32);
  auto time = llvmBuilder.or_(time_lo_ext, time_hi_shl);
  auto threadIdx_x = loadFromBuffer(index, 2);
  auto threadIdx_y = loadFromBuffer(index, 3);
  auto threadIdx_z = loadFromBuffer(index, 4);
  auto blockIdx_x = loadFromBuffer(index, 5);
  auto blockIdx_y = loadFromBuffer(index, 6);
  auto blockIdx_z = loadFromBuffer(index, 7);
  auto ctaRank = loadFromBuffer(index, 8);
  auto const9 = llvmBuilder.i32_val(9);
  auto nextIndex = llvmBuilder.add(index, const9);
  llvmBuilder.store(nextIndex, indexPtr);

  std::string formatStr;
  llvm::raw_string_ostream os(formatStr);
  os << "time=%lu thread=%u,%u,%u block=%u,%u,%u cta_rank=%u ";
  llvm::SmallVector<Value> args{time,        threadIdx_x, threadIdx_y,
                                threadIdx_z, blockIdx_x,  blockIdx_y,
                                blockIdx_z,  ctaRank};
  targetInfo.printf(rewriter, formatStr, args);
  rewriter.create<LLVM::BrOp>(loc, loopConditionBlock);
}

namespace {
struct InsertSynclogs
    : public mlir::triton::impl::InsertSynclogsBase<InsertSynclogs> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto ctx = &getContext();
    IRRewriter rewriter(ctx);

    auto it = llvm::find_if(mod.getOps<LLVM::LLVMFuncOp>(), [](auto func) {
      return func->hasAttr("nvvm.kernel");
    });
    if (it == mod.getOps<LLVM::LLVMFuncOp>().end()) {
      assert(false && "kernel function not found");
      signalPassFailure();
    }
    LLVM::LLVMFuncOp kernelFunc = *it;

    // Get pointer to synclog buffer. This is assumed to be the second last
    // argument of the kernel.
    std::optional<Value> prevPtr;
    std::optional<Value> currPtr;
    for (auto operand : kernelFunc.getArguments()) {
      if (isa<LLVM::LLVMPointerType>(operand.getType())) {
        prevPtr = currPtr;
        currPtr = operand;
      }
    }
    if (!prevPtr.has_value()) {
      if (!currPtr.has_value()) {
        assert(false && "synclog buffer not found");
        signalPassFailure();
      }
      prevPtr = currPtr;
    }
    auto synclogBuffer = *prevPtr;

    auto shouldEmitSynclog = [&](std::string &opName) -> int {
      return llvm::any_of(opNames, [&](const auto &elem) {
        return opName.find(elem) != std::string::npos;
      });
    };

    llvm::SmallVector<Operation *> asyncOps;
    mod.walk([&](Operation *op) {
      std::string instrString("");
      llvm::TypeSwitch<Operation *>(op)
          .Case<LLVM::InlineAsmOp>([&](auto inlineAsmOp) {
            instrString = inlineAsmOp.getAsmString().str();
          })
          .Case<LLVM::CallIntrinsicOp>([&](auto callIntrinsicOp) {
            instrString = callIntrinsicOp.getIntrin().str();
          })
          .Default(
              [&](auto) { instrString = op->getName().getStringRef().str(); });
      if (shouldEmitSynclog(instrString)) {
        asyncOps.push_back(op);
      }
    });

    for (auto op : asyncOps) {
      writeLogToBuffer(rewriter, mod, kernelFunc, op, synclogBuffer);
    }

    // Using hardcoded compute capability and ptx version since the printing is
    // independent of them.
    NVIDIA::TargetInfo targetInfo(/*computeCapability=*/80, /*ptxVersion=*/80);
    // Print synclog buffer at the end of the kernel.
    printSynclog(rewriter, targetInfo, kernelFunc, synclogBuffer);
  }
};
} // namespace
