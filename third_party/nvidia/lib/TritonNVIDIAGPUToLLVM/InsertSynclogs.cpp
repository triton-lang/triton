#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir::triton {
#define GEN_PASS_DEF_INSERTSYNCLOGS
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// Array to map NVVM/PTX ops to unique header numbers.
// This allows us to identify the operation in the synclog.
std::array<const char *, 23> opNames{"barrier.sync",
                                     "nvvm.barrier0",
                                     "cp.async.bulk.commit.group",
                                     "cp.async.commit.group",
                                     "cp.async.bulk.wait.group",
                                     "cp.async.wait.group",
                                     "cp.async.bulk.tensor",
                                     "cp.reduce.async.bulk.tensor",
                                     "fence.proxy",
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

  std::string formatStr;
  llvm::raw_string_ostream os(formatStr);
  os << opName << " time=%lu thread=%u,%u,%u block=%u,%u,%u ";

  targetInfo.printf(rewriter, formatStr,
                    {time64, blockIdx_x, threadIdx_x, blockIdx_y, threadIdx_y,
                     blockIdx_z, threadIdx_z});
}

namespace {
struct InsertSynclogs
    : public mlir::triton::impl::InsertSynclogsBase<InsertSynclogs> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());
    NVIDIA::TargetInfo targetInfo(/*computeCapability=*/100, /*ptxVersion=*/87);
    auto shouldEmitSynclog = [&](std::string &opName) -> int {
      return llvm::any_of(opNames, [&](const auto &elem) {
        return opName.find(elem) != std::string::npos;
      });
    };
    mod.walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .Case<LLVM::InlineAsmOp>([&](auto inlineAsmOp) {
            auto asmString = inlineAsmOp.getAsmString().str();
            if (shouldEmitSynclog(asmString)) {
              emitSynclog(rewriter, op, targetInfo, asmString);
            }
          })
          .Case<LLVM::CallIntrinsicOp>([&](auto callIntrinsicOp) {
            auto asmString = callIntrinsicOp.getIntrin().str();
            if (shouldEmitSynclog(asmString)) {
              emitSynclog(rewriter, op, targetInfo, asmString);
            }
          })
          .Default([&](auto) {
            auto opName = op->getName().getStringRef().str();
            if (shouldEmitSynclog(opName)) {
              emitSynclog(rewriter, op, targetInfo, opName);
            }
          });
    });
  }
};
} // namespace
