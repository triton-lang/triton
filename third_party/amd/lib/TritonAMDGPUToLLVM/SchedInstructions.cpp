#include "TritonAMDGPUToLLVM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_INSERTINSTRUCTIONSCHEDHINTS
#define GEN_PASS_DEF_LOWERINSTRUCTIONSCHEDHINTS
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

// The bitmask that encodes kinds of the instructions from AMD ISA.
// The bitmask is used for providing instruction scheduling hints.
enum InstructionKindMask {
  NONE = 0x0000000,
  ALL_ALU = 0x00000001,
  VALU = 0x00000002,
  SALU = 0x00000004,
  MFMA = 0x00000008,
  ALL_VMEM = 0x00000010,
  VMEM_READ = 0x00000020,
  VMEM_WRITE = 0x00000040,
  ALL_DS = 0x00000080,
  DS_READ = 0x00000100,
  DS_WRITE = 0x00000200
};

// Create an intrinsic to control how different instruction kinds should
// interleave for better ILP.
void createSchedGroupBarrier(PatternRewriter &rewriter, Location loc,
                             InstructionKindMask maskValue, int sizeValue,
                             int groupIdValue) {
  MLIRContext *ctx = rewriter.getContext();
  auto intrinsicName = str_attr("llvm.amdgcn.sched.group.barrier");

  Value mask =
      LLVM::createConstantI32(loc, rewriter, static_cast<int32_t>(maskValue));
  Value size =
      LLVM::createConstantI32(loc, rewriter, static_cast<int32_t>(sizeValue));
  Value groupId = LLVM::createConstantI32(loc, rewriter,
                                          static_cast<int32_t>(groupIdValue));

  LLVM::FastmathFlagsAttr defaultFlags{};
  rewriter.create<LLVM::CallIntrinsicOp>(loc, TypeRange{}, intrinsicName,
                                         ValueRange{mask, size, groupId},
                                         defaultFlags);
}

// Insert intrinsic that controls the types of instructions that may be
// allowed to cross the intrinsic during instruction scheduling
Operation *createSchedBarrier(PatternRewriter &rewriter, Location loc,
                              int64_t maskValue) {
  MLIRContext *ctx = rewriter.getContext();
  auto intrinsicName = str_attr("llvm.amdgcn.sched.barrier");
  LLVM::FastmathFlagsAttr defaultFlags{};

  Value mask =
      LLVM::createConstantI32(loc, rewriter, static_cast<int32_t>(maskValue));
  return rewriter.create<LLVM::CallIntrinsicOp>(loc, TypeRange{}, intrinsicName,
                                                ValueRange{mask}, defaultFlags);
}

// Insert an experimental intrinsic for instruction group level parallelism.
// The intrinsic takes a value that specifies the strategy.
Operation *createIglpOpt(PatternRewriter &rewriter, Location loc, int value) {
  MLIRContext *ctx = rewriter.getContext();
  auto intrinsicName = str_attr("llvm.amdgcn.iglp.opt");
  LLVM::FastmathFlagsAttr defaultFlags{};
  Value iglpValue =
      LLVM::createConstantI32(loc, rewriter, static_cast<int32_t>(value));
  return rewriter.create<LLVM::CallIntrinsicOp>(
      loc, TypeRange{}, intrinsicName, ValueRange{iglpValue}, defaultFlags);
}

struct InstructionSchedHintsRewriter
    : public OpRewritePattern<triton::amdgpu::InstructionSchedHint> {

  InstructionSchedHintsRewriter(mlir::MLIRContext *ctx, uint32_t variant)
      : OpRewritePattern(ctx) {
    assert(variant < static_cast<uint32_t>(SchedulingType::COUNT) &&
           "instruction scheduling hint must have a valid value");
    schedulingType = static_cast<SchedulingType>(variant);
  }

  enum class SchedulingType : uint32_t { NONE = 0, IGLP_0, IGLP_1, COUNT };

  LogicalResult
  matchAndRewrite(triton::amdgpu::InstructionSchedHint instructionSchedHint,
                  PatternRewriter &rewriter) const override {

    // The switch controls whether instructions are allowed to cross the basic
    // block boundaries at the very top and at the very bottom. Note, this is
    // not supposed to be used together with IGLP OPT according to the AMDGPU
    // backend documentation.
    const bool limitSchedulingRange =
        !(schedulingType == SchedulingType::IGLP_0 ||
          schedulingType == SchedulingType::IGLP_1);
    Location loc = instructionSchedHint->getLoc();
    Block *block = instructionSchedHint->getBlock();
    if (limitSchedulingRange) {
      rewriter.setInsertionPointToStart(block);
      createSchedBarrier(rewriter, loc, InstructionKindMask::NONE);
    }

    rewriter.setInsertionPoint(block, std::prev(block->end()));

    switch (schedulingType) {
    case SchedulingType::IGLP_0:
      [[fallthrough]];
    case SchedulingType::IGLP_1: {
      createIglpOpt(rewriter, loc, static_cast<int>(schedulingType) - 1);
      break;
    }
    case SchedulingType::NONE:
      [[fallthrough]];
    default: {
      break;
    }
    }

    if (limitSchedulingRange)
      createSchedBarrier(rewriter, loc, InstructionKindMask::NONE);

    rewriter.eraseOp(instructionSchedHint);
    return mlir::success();
  }

private:
  SchedulingType schedulingType;
};

struct LowerInstructionSchedHints
    : public triton::impl::LowerInstructionSchedHintsBase<
          LowerInstructionSchedHints> {

  explicit LowerInstructionSchedHints(uint32_t variant) {
    this->variant = variant;
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalOp<triton::amdgpu::InstructionSchedHint>();

    RewritePatternSet patterns(ctx);
    patterns.add<InstructionSchedHintsRewriter>(ctx, this->variant);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct InsertInstructionSchedHints
    : public triton::impl::InsertInstructionSchedHintsBase<
          InsertInstructionSchedHints> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();

    mod->walk([ctx](triton::DotOp dot) {
      if (dyn_cast<mlir::scf::ForOp>(dot->getParentOp())) {
        mlir::OpBuilder rewriter(ctx);
        rewriter.setInsertionPointAfter(dot);
        rewriter.create<triton::amdgpu::InstructionSchedHint>(dot->getLoc());
      }
    });
  }
};
} // namespace

namespace mlir::triton {
std::unique_ptr<OperationPass<ModuleOp>>
createLowerInstructionSchedHintsPass(uint32_t variant) {
  return std::make_unique<LowerInstructionSchedHints>(variant);
}

std::unique_ptr<OperationPass<ModuleOp>>
createInsertInstructionSchedHintsPass() {
  return std::make_unique<InsertInstructionSchedHints>();
}
} // namespace mlir::triton
