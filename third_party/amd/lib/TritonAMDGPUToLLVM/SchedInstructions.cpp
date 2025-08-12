#include "TritonAMDGPUToLLVM/Passes.h"
#include "Utility.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONAMDGPUINSERTINSTRUCTIONSCHEDHINTS
#define GEN_PASS_DEF_TRITONAMDGPULOWERINSTRUCTIONSCHEDHINTS
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

#undef DEBUG_TYPE
#define DEBUG_TYPE "lower-insert-instruction-sched-hints"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using ::mlir::LLVM::AMD::isChainDotHead;

// TODO: The following passes/algorithms are applicable only for a single
// `tt.dot` op in a `scf.for` block -i.e., a single schedule hint op per block.
// Note, we need to relax this assumption in the future and extend the current
// implementation.

namespace {

// Insert intrinsic that controls the types of instructions that may be
// allowed to cross the intrinsic during instruction scheduling.
Operation *createSchedBarrier(PatternRewriter &rewriter, Location loc,
                              mlir::amdgpu::sched_barrier_opt_enum maskValue) {
  IntegerAttr mask =
      rewriter.getI32IntegerAttr(static_cast<int32_t>(maskValue));
  return rewriter.create<ROCDL::SchedBarrier>(loc, mask);
}

// Insert an experimental intrinsic for instruction group level parallelism.
// The intrinsic takes a value that specifies the strategy.
Operation *createIglpOpt(PatternRewriter &rewriter, Location loc, int value) {
  IntegerAttr iglpValue =
      rewriter.getI32IntegerAttr(static_cast<int32_t>(value));
  return rewriter.create<ROCDL::IglpOpt>(loc, iglpValue);
}

struct InstructionSchedHintsRewriter
    : public OpRewritePattern<triton::amdgpu::InstructionSchedHint> {

  InstructionSchedHintsRewriter(MLIRContext *ctx, StringRef arch,
                                int32_t numStages)
      : OpRewritePattern(ctx), numStages(numStages) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::InstructionSchedHint instructionSchedHint,
                  PatternRewriter &rewriter) const override {
    auto schedVariant = instructionSchedHint.getVariant();
    if (schedVariant == mlir::triton::amdgpu::SchedHint::none) {
      rewriter.eraseOp(instructionSchedHint);
      return success();
    }

    // The switch controls whether instructions are allowed to cross the basic
    // block boundaries at the very top and at the very bottom. Note, this is
    // not supposed to be used together with IGLP OPT according to the AMDGPU
    // backend documentation.
    const bool limitSchedulingRange =
        schedVariant == mlir::triton::amdgpu::SchedHint::attention;
    ;
    Location loc = instructionSchedHint->getLoc();
    Block *block = instructionSchedHint->getBlock();
    if (limitSchedulingRange) {
      rewriter.setInsertionPointToStart(block);
      createSchedBarrier(rewriter, loc,
                         mlir::amdgpu::sched_barrier_opt_enum::none);
    }

    rewriter.setInsertionPoint(block, std::prev(block->end()));

    switch (schedVariant) {
    case mlir::triton::amdgpu::SchedHint::attention:
      createIglpOpt(rewriter, loc, 2);
      break;
    case mlir::triton::amdgpu::SchedHint::none:
    default:
      break;
    }

    if (limitSchedulingRange)
      createSchedBarrier(rewriter, loc,
                         mlir::amdgpu::sched_barrier_opt_enum::none);

    rewriter.eraseOp(instructionSchedHint);
    return success();
  }

private:
  int32_t numStages;
};

struct TritonAMDGPULowerInstructionSchedHints
    : public triton::impl::TritonAMDGPULowerInstructionSchedHintsBase<
          TritonAMDGPULowerInstructionSchedHints> {

  explicit TritonAMDGPULowerInstructionSchedHints(StringRef arch,
                                                  int32_t numStages) {
    this->arch = std::move(arch.str());
    this->numStages = numStages;
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalOp<triton::amdgpu::InstructionSchedHint>();
    target.addLegalOp<ROCDL::SchedBarrier>();
    target.addLegalOp<ROCDL::IglpOpt>();
    target.addLegalOp<ROCDL::SchedGroupBarrier>();

    RewritePatternSet patterns(ctx);

    patterns.add<InstructionSchedHintsRewriter>(ctx, this->arch,
                                                this->numStages);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {

      signalPassFailure();
    }
  }
};

struct TritonAMDGPUInsertInstructionSchedHints
    : public triton::impl::TritonAMDGPUInsertInstructionSchedHintsBase<
          TritonAMDGPUInsertInstructionSchedHints> {

  explicit TritonAMDGPUInsertInstructionSchedHints(StringRef variant) {
    this->variant = std::move(variant.str());
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();

    auto schedHint = mlir::triton::amdgpu::SchedHint::none;
    std::transform(variant.begin(), variant.end(), variant.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (auto maybeSchedHint = triton::amdgpu::symbolizeSchedHint(variant))
      schedHint = maybeSchedHint.value();
    else {
      LDBG("ignoring instruction scheduling because "
           "unknown instruction scheduling variant has been provided");
      return;
    }

    switch (schedHint) {
    case mlir::triton::amdgpu::SchedHint::attention:
      mod.walk([&](scf::ForOp forOp) {
        // The attention schedule hint is inserted to the beginning of a
        // for-loop with chained dots.
        auto result = forOp->walk([](triton::DotOp op) {
          if (isChainDotHead(op))
            return WalkResult::interrupt();
          return WalkResult::advance();
        });

        if (result.wasInterrupted()) {
          OpBuilder rewriter(ctx);
          rewriter.setInsertionPointToStart(forOp.getBody());
          rewriter.create<triton::amdgpu::InstructionSchedHint>(forOp->getLoc(),
                                                                schedHint);
        }
      });
      break;
    case mlir::triton::amdgpu::SchedHint::none:
    default:
      break;
    }
  }
};
} // namespace

namespace mlir::triton {
std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPULowerInstructionSchedHintsPass(StringRef arch,
                                                 int32_t numStages) {
  return std::make_unique<TritonAMDGPULowerInstructionSchedHints>(arch,
                                                                  numStages);
}

std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPUInsertInstructionSchedHintsPass(StringRef variant) {
  return std::make_unique<TritonAMDGPUInsertInstructionSchedHints>(variant);
}
} // namespace mlir::triton
