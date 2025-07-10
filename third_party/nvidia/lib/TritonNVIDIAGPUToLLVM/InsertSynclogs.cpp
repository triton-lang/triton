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

// Array of NVVM/PTX op prefixes and their corresponding header numbers.
// NVVM prefixes are only included if they are different from the PTX prefixes.
const llvm::SmallVector<std::pair<const char *, int>, 25> opNames{
    std::make_pair("barrier.sync", 0),
    std::make_pair("nvvm.barrier0", 0),
    std::make_pair("cp.async.bulk.commit_group", 1),
    std::make_pair("nvvm.cp.async.bulk.commit.group", 1),
    std::make_pair("cp.async.commit_group", 2),
    std::make_pair("nvvm.cp.async.commit.group", 2),
    std::make_pair("cp.async.bulk.wait_group", 3),
    std::make_pair("cp.async.wait_group", 4),
    std::make_pair("nvvm.cp.async.wait.group", 4),
    std::make_pair("cp.async.bulk.tensor", 5),
    std::make_pair("cp.reduce.async.bulk.tensor", 6),
    std::make_pair("fence.proxy", 7),
    std::make_pair("tensormap.cp_fenceproxy", 8),
    std::make_pair("fence.mbarrier_init", 9),
    std::make_pair("wgmma.fence", 10),
    std::make_pair("wgmma.commit_group", 11),
    std::make_pair("wgmma.wait_group", 12),
    std::make_pair("mbarrier.init", 13),
    std::make_pair("mbarrier.wait", 14),
    std::make_pair("mbarrier.arrive", 15),
    std::make_pair("mbarrier.inval", 16),
    std::make_pair("mbarrier.expect_tx", 17),
    std::make_pair("mbarrier.test_wait", 18),
    std::make_pair("mbarrier.try_wait", 19),
    std::make_pair("tcgen05.wait", 20),
    std::make_pair("tcgen05.commit", 21),
    std::make_pair("cp.async.mbarrier.arrive", 22)};

constexpr int synclogCap = 1 << 26;

// Add synclog ptr argument to kernel function and return new, updated kernel.
LLVM::LLVMFuncOp addSynclogPtr(IRRewriter &rewriter, ModuleOp mod,
                               LLVM::LLVMFuncOp kernelFunc) {
  auto loc = kernelFunc.getLoc();
  auto ctx = rewriter.getContext();
  auto synclogPtrTy = LLVM::LLVMPointerType::get(ctx, 1);

  // Add the pointer type to the function type.
  // The synclog pointer will be the second last argument to the kernel (last
  // argument is the global scratch memory pointer).
  auto funcTy = kernelFunc.getFunctionType();
  llvm::SmallVector<mlir::Type, 4> argTypes(funcTy.getParams().begin(),
                                            funcTy.getParams().end());
  auto insertionPoint = argTypes.end() - 1;
  argTypes.insert(insertionPoint, synclogPtrTy);

  // Create the new function.
  auto newKernelFuncTy =
      LLVM::LLVMFunctionType::get(funcTy.getReturnType(), argTypes);
  rewriter.setInsertionPoint(kernelFunc);
  auto newKernelFunc = rewriter.create<LLVM::LLVMFuncOp>(
      mod.getLoc(), kernelFunc.getName(), newKernelFuncTy);

  // Modify attributes to add the new argument.
  SmallVector<NamedAttribute> amendedAttrs;
  for (const auto &attr : kernelFunc->getAttrs()) {
    if (attr.getName() == "function_type") {
      continue;
    }
    amendedAttrs.push_back(attr);
  }
  if (auto argAttrs = kernelFunc.getAllArgAttrs()) {
    llvm::SmallVector<mlir::Attribute> amendedArgAttrs(argAttrs.begin(),
                                                       argAttrs.end());
    amendedArgAttrs.emplace_back(DictionaryAttr::get(ctx));
    amendedAttrs.push_back(
        rewriter.getNamedAttr(kernelFunc.getArgAttrsAttrName(),
                              rewriter.getArrayAttr(amendedArgAttrs)));
  }
  newKernelFunc->setAttrs(amendedAttrs);

  // Add the new argument to the function body and inline.
  auto &region = kernelFunc.getBody();
  auto numArgs = region.getNumArguments();
  region.insertArgument(numArgs - 1, synclogPtrTy, loc);
  rewriter.inlineRegionBefore(region, newKernelFunc.getBody(),
                              newKernelFunc.end());
  rewriter.eraseOp(kernelFunc);

  return newKernelFunc;
}

// Helper template to recursively chain AND operations.
template <typename First, typename... Rest>
Value andChain(TritonLLVMOpBuilder &llvmBuilder, First first, Rest... rest) {
  if constexpr (sizeof...(rest) == 0) {
    return first.getResult();
  } else {
    return llvmBuilder.and_(first.getResult(), andChain(llvmBuilder, rest...));
  }
}

// Helper to write synclog to buffer for a given async operation.
void writeLogToBuffer(IRRewriter &rewriter, Operation *op, Value synclogBuffer,
                      int headerNumber) {
  auto ctx = rewriter.getContext();
  auto loc = op->getLoc();
  TritonLLVMOpBuilder llvmBuilder(loc, rewriter);
  auto i32Type = type::i32Ty(ctx);
  auto i64Type = type::i64Ty(ctx);
  auto globalPtrTy = LLVM::LLVMPointerType::get(ctx, 1);

  rewriter.setInsertionPoint(op);
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
  auto andAll = andChain(llvmBuilder, cond1, cond2, cond3, cond4, cond5, cond6);

  auto *prevBlock = rewriter.getInsertionBlock();
  auto *ifBlock = rewriter.splitBlock(prevBlock, rewriter.getInsertionPoint());
  auto *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
  rewriter.setInsertionPointToEnd(prevBlock);
  rewriter.create<LLVM::CondBrOp>(loc, andAll, ifBlock, thenBlock);
  rewriter.setInsertionPointToEnd(ifBlock);
  rewriter.create<LLVM::BrOp>(loc, thenBlock);
  rewriter.setInsertionPointToStart(ifBlock);

  // Atomically allocate space in synclog buffer.
  auto numArgs = std::min(
      5, static_cast<int>(llvm::count_if(op->getOperands(), [](auto arg) {
        return isa<LLVM::LLVMPointerType>(arg.getType()) ||
               arg.getType().isInteger();
      })));
  auto constLength = llvmBuilder.i32_val(11 + numArgs);
  auto last = rewriter.create<LLVM::AtomicRMWOp>(
      loc, LLVM::AtomicBinOp::add, synclogBuffer, constLength,
      LLVM::AtomicOrdering::monotonic);
  auto lastPlusConstLength = llvmBuilder.add(last, constLength);
  auto synclogCapValue = llvmBuilder.i32_val(synclogCap);
  auto lastPlusConstLengthLessThanCap =
      llvmBuilder.icmp_ult(lastPlusConstLength, synclogCapValue);

  auto *writeBufferBlock =
      rewriter.splitBlock(ifBlock, rewriter.getInsertionPoint());
  auto *bufferFullBlock =
      rewriter.splitBlock(writeBufferBlock, rewriter.getInsertionPoint());
  rewriter.setInsertionPointToEnd(ifBlock);
  rewriter.create<LLVM::CondBrOp>(loc, lastPlusConstLengthLessThanCap,
                                  writeBufferBlock, bufferFullBlock);
  rewriter.setInsertionPointToEnd(writeBufferBlock);
  rewriter.create<LLVM::BrOp>(loc, thenBlock);

  rewriter.setInsertionPointToStart(writeBufferBlock);
  auto const1 = llvmBuilder.i32_val(1);
  auto offset = llvmBuilder.add(const1, last);
  auto synclogPtr =
      llvmBuilder.gep(globalPtrTy, i32Type, synclogBuffer, offset.getResult());

  // Write to buffer. Information is stored in the following order:
  // 0: header number
  // 1: number of arguments stored
  // 2: argument 0
  // 3: ...
  // 2 + numArgsStored: time_lo
  // 3 + numArgsStored: time_hi
  // 4 + numArgsStored: threadIdx_x
  // 5 + numArgsStored: threadIdx_y
  // 6 + numArgsStored: threadIdx_z
  // 7 + numArgsStored: blockIdx_x
  // 8 + numArgsStored: blockIdx_y
  // 9 + numArgsStored: blockIdx_z
  // 10 + numArgsStored: ctaRank
  auto storeInBuffer = [&](Value valToStore, int index) {
    auto ptr = llvmBuilder.gep(globalPtrTy, i32Type, synclogPtr,
                               llvmBuilder.i32_val(index));
    llvmBuilder.store(valToStore, ptr);
  };
  auto headerNumberValue = llvmBuilder.i32_val(headerNumber);
  storeInBuffer(headerNumberValue, 0);
  storeInBuffer(llvmBuilder.i32_val(numArgs), 1);
  size_t argIndex = 2;
  for (const auto &arg : op->getOperands()) {
    auto argType = arg.getType();
    // Store only pointers and integers.
    if (isa<LLVM::LLVMPointerType>(argType)) {
      storeInBuffer(llvmBuilder.ptrtoint(i32Type, arg), argIndex);
      argIndex++;
    } else if (argType.isInteger()) {
      storeInBuffer(arg, argIndex);
      argIndex++;
    }
    // Store at most 5 arguments.
    if (argIndex >= 7) {
      break;
    }
  }

  auto afterArgsOffset = llvmBuilder.i32_val(2 + numArgs);
  synclogPtr =
      llvmBuilder.gep(globalPtrTy, i32Type, synclogPtr, afterArgsOffset);
  storeInBuffer(time_lo, 0);
  storeInBuffer(time_hi, 1);
  storeInBuffer(threadIdx_x, 2);
  storeInBuffer(threadIdx_y, 3);
  storeInBuffer(threadIdx_z, 4);
  storeInBuffer(blockIdx_x, 5);
  storeInBuffer(blockIdx_y, 6);
  storeInBuffer(blockIdx_z, 7);
  storeInBuffer(ctaRank, 8);

  // If the buffer is full, move the pointer back n places.
  rewriter.setInsertionPointToStart(bufferFullBlock);
  rewriter.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::sub, synclogBuffer,
                                     constLength,
                                     LLVM::AtomicOrdering::monotonic);
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
    kernelFunc = addSynclogPtr(rewriter, mod, kernelFunc);
    auto synclogBuffer = *(kernelFunc.getArguments().end() - 2);

    auto synclogHeader = [&](std::string &opName) -> int {
      auto it = llvm::find_if(opNames, [&](const auto &elem) {
        return opName.find(elem.first) != std::string::npos;
      });
      if (it == opNames.end()) {
        return -1;
      }
      return it->second;
    };

    llvm::SmallVector<std::pair<Operation *, int>> asyncOps;
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
      auto header = synclogHeader(instrString);
      if (header != -1) {
        asyncOps.push_back(std::make_pair(op, header));
      }
    });

    for (const auto &[op, header] : asyncOps) {
      writeLogToBuffer(rewriter, op, synclogBuffer, header);
    }
  }
};
} // namespace
