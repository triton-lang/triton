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

// Array to map NVVM/PTX ops to unique header numbers.
// This allows us to identify the operation in the synclog.
std::array<const char *, 25> opNames{"barrier.sync",
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

void emitSynclog(IRRewriter &rewriter, Operation *op,
                 NVIDIA::TargetInfo &targetInfo, std::string &opName) {
  rewriter.setInsertionPoint(op);
  auto ctx = rewriter.getContext();
  auto loc = op->getLoc();
  auto i32Type = type::i32Ty(ctx);
  auto i64Type = type::i64Ty(ctx);

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
  auto blockIdx_x = rewriter.create<NVVM::BlockIdXOp>(loc, i32Type);
  auto threadIdx_x = rewriter.create<NVVM::ThreadIdXOp>(loc, i32Type);
  auto blockIdx_y = rewriter.create<NVVM::BlockIdYOp>(loc, i32Type);
  auto threadIdx_y = rewriter.create<NVVM::ThreadIdYOp>(loc, i32Type);
  auto blockIdx_z = rewriter.create<NVVM::BlockIdZOp>(loc, i32Type);
  auto threadIdx_z = rewriter.create<NVVM::ThreadIdZOp>(loc, i32Type);
  auto ctaRank = rewriter.create<NVVM::ClusterId>(loc, i32Type);

  std::string formatStr;
  llvm::raw_string_ostream os(formatStr);
  os << opName << " time=%lu thread=%u,%u,%u block=%u,%u,%u cta_rank=%u ";

  // Remove invalid characters from the format string.
  formatStr.erase(std::remove_if(formatStr.begin(), formatStr.end(),
                                 [](char c) { return c == '\t'; }),
                  formatStr.end());

  llvm::SmallVector<Value> args{time64,      threadIdx_x, threadIdx_y,
                                threadIdx_z, blockIdx_x,  blockIdx_y,
                                blockIdx_z,  ctaRank};

  size_t numArgsPrinted = 0;
  for (size_t i = 0; i < op->getNumOperands() && numArgsPrinted < 5; i++) {
    // Cap to 5 operands for now. This is a temporary solution to avoid
    // printing out tensor values.
    auto operand = op->getOperand(i);
    // By default, print integer operands as signed to avoid overflows.
    auto formatSubstr = getFormatSubstr(operand, false, std::nullopt, true);
    if (!formatSubstr.has_value()) {
      continue;
    }
    args.push_back(operand);
    os << *formatSubstr << " ";
    numArgsPrinted++;
  }
  targetInfo.printf(rewriter, formatStr, args);
}

namespace {
struct InsertSynclogs
    : public mlir::triton::impl::InsertSynclogsBase<InsertSynclogs> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());
    // Using hardcoded compute capability and ptx version since the printing is
    // independent of them.
    NVIDIA::TargetInfo targetInfo(/*computeCapability=*/80, /*ptxVersion=*/80);
    auto shouldEmitSynclog = [&](std::string &opName) -> int {
      return llvm::any_of(opNames, [&](const auto &elem) {
        return opName.find(elem) != std::string::npos;
      });
    };
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
        emitSynclog(rewriter, op, targetInfo, instrString);
      }
    });
  }
};
} // namespace
