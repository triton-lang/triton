#include "SchedInstructions.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
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

// TODO: The following passes/algorithms are applicable only for a single
// `tt.dot` op in a `scf.for` block -i.e., a single schedule hint op per block.
// Note, we need to relax this assumption in the future and extend the current
// implementation.

namespace mlir::triton {
void setNumGeneratedMMAs(DotOp op, size_t mmaCount, unsigned m, unsigned n,
                         unsigned k, Type elementType) {
  auto *ctx = op->getContext();
  auto mmaType = RankedTensorType::get({m, n, k}, elementType);
  auto counterAttr =
      triton::amdgpu::InstCounterAttr::get(ctx, mmaCount, mmaType);

  op->getBlock()->walk([&](triton::amdgpu::InstructionSchedHint schedHint) {
    schedHint.setNumMMAsAttr(counterAttr);
  });
}

template <typename LoadOpType>
void setNumGeneratedGlobalLoads(LoadOpType op, size_t globalLoadsCount,
                                Type type) {
  MLIRContext *ctx = op->getContext();
  auto counterAttr =
      triton::amdgpu::InstCounterAttr::get(ctx, globalLoadsCount, type);

  op->getBlock()->walk([&](triton::amdgpu::InstructionSchedHint schedHint) {
    if (auto opIdxAttr = op->template getAttrOfType<triton::amdgpu::OpIdxAttr>(
            triton::amdgpu::OpIdxAttr::getMnemonic())) {
      assert(opIdxAttr.getValue() < 2);
      const bool isBufferLoadOp =
          std::is_same_v<LoadOpType, triton::amdgpu::BufferLoadOp>;
      if (opIdxAttr.getValue() == 0) {
        schedHint.setNumGlobalLoadsAAttr(counterAttr);
        schedHint.setIsBufferLoadsAEnabled(isBufferLoadOp);
      } else {
        schedHint.setNumGlobalLoadsBAttr(counterAttr);
        schedHint.setIsBufferLoadsBEnabled(isBufferLoadOp);
      }
    }
  });
}
template void setNumGeneratedGlobalLoads(triton::amdgpu::BufferLoadOp op,
                                         size_t globalLoadsCount, Type type);
template void setNumGeneratedGlobalLoads(triton::LoadOp op,
                                         size_t globalLoadsCount, Type type);

void setNumGeneratedDsReads(gpu::LocalLoadOp op, size_t dsReadsCount,
                            Type type) {
  auto *ctx = op->getContext();
  auto counterAttr =
      triton::amdgpu::InstCounterAttr::get(ctx, dsReadsCount, type);

  op->getBlock()->walk([&](triton::amdgpu::InstructionSchedHint schedHint) {
    Value dst = op.getResult();
    auto dstTensorTy = cast<RankedTensorType>(dst.getType());
    auto dotOperandLayout =
        cast<DotOperandEncodingAttr>(dstTensorTy.getEncoding());
    const size_t opIdx = dotOperandLayout.getOpIdx();
    assert(opIdx < 2);
    if (opIdx == 0)
      schedHint.setNumDsReadsAAttr(counterAttr);
    else
      schedHint.setNumDsReadsBAttr(counterAttr);
  });
}

void storeOpConversionCallback(triton::gpu::LocalStoreOp op,
                               size_t localStoreOpCount, Type type) {
  MLIRContext *ctx = op->getContext();
  auto counterAttr =
      triton::amdgpu::InstCounterAttr::get(ctx, localStoreOpCount, type);

  op->getBlock()->walk([&](triton::amdgpu::InstructionSchedHint schedHint) {
    if (auto opIdxAttr = op->getAttrOfType<triton::amdgpu::OpIdxAttr>(
            triton::amdgpu::OpIdxAttr::getMnemonic())) {
      assert(opIdxAttr.getValue() < 2);
      if (opIdxAttr.getValue() == 0)
        schedHint.setNumDsWritesAAttr(counterAttr);
      else
        schedHint.setNumDsWritesBAttr(counterAttr);
    }
  });
}

triton::DotOp getSingleDotOpIfExists(scf::ForOp forOp) {
  triton::DotOp dotOp = nullptr;
  size_t dotCounter = 0;
  forOp->walk(
      [&dotOp, &dotCounter](triton::DotOp op) { dotOp = op, ++dotCounter; });

  return (dotCounter == 1) ? dotOp : nullptr;
}
} // namespace mlir::triton

namespace {

// Create an intrinsic to control how different instruction kinds should
// interleave for better ILP.
void createSchedGroupBarrier(PatternRewriter &rewriter, Location loc,
                             mlir::amdgpu::sched_barrier_opt_enum maskValue,
                             int sizeValue, int groupIdValue) {
  IntegerAttr mask =
      rewriter.getI32IntegerAttr(static_cast<int32_t>(maskValue));
  IntegerAttr size =
      rewriter.getI32IntegerAttr(static_cast<int32_t>(sizeValue));
  IntegerAttr groupId =
      rewriter.getI32IntegerAttr(static_cast<int32_t>(groupIdValue));
  rewriter.create<ROCDL::SchedGroupBarrier>(loc, mask, size, groupId);
}

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

  InstructionSchedHintsRewriter(MLIRContext *ctx, int32_t numStages,
                                std::string variant)
      : OpRewritePattern(ctx), numStages(numStages) {
    std::transform(variant.begin(), variant.end(), variant.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    this->schedulingType = llvm::StringSwitch<SchedulingType>(variant)
                               .Case("none", SchedulingType::NONE)
                               .Case("iglp0", SchedulingType::IGLP0)
                               .Case("iglp1", SchedulingType::IGLP1)
                               .Case("ck_v3", SchedulingType::CK_V3)
                               .Default(SchedulingType::UNKNOWN);

    if (this->numStages < 2) {
      this->schedulingType = SchedulingType::NONE;
      LDBG("ignoring instruction scheduling due to a very low num. "
           "stages value. Must be >= 2");
    }
  }

  enum class SchedulingType : uint32_t {
    NONE = 0,
    IGLP0,
    IGLP1,
    CK_V3,
    UNKNOWN
  };

  // This is the implementation of the CK's V3 pipelining (see
  // see ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3.hpp).
  // This scheduling requires 1x register and 1x LDS buffers combined with the
  // local (LDS to registers) and global (HBM to registers) data prefetching.
  // see:
  // include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3.h
  void
  createCKV3Schedule(PatternRewriter &rewriter, Location loc,
                     triton::amdgpu::InstructionSchedHint schedHint) const {

    if (!(schedHint.getIsBufferLoadsAEnabled() &&
          schedHint.getIsBufferLoadsBEnabled())) {
      LDBG("Skipping instruction scheduling because `ck_v3` "
           "scheduling can be used only with `buffer_load` instructions.");
      return;
    }

    const uint32_t numDsReadInstA = schedHint.getNumDsReadsA().getValue();
    const uint32_t numDsReadInstB = schedHint.getNumDsReadsB().getValue();

    const uint32_t numDsWriteInstA = schedHint.getNumDsWritesA().getValue();
    const uint32_t numDsWriteInstB = schedHint.getNumDsWritesB().getValue();

    const uint32_t numBufferLoadInstA =
        schedHint.getNumGlobalLoadsA().getValue();
    const uint32_t numBufferLoadInstB =
        schedHint.getNumGlobalLoadsB().getValue();

    if (numBufferLoadInstA == 0)
      schedHint.emitError("buffer load count for tile A must be initialized");

    if (numBufferLoadInstB == 0)
      schedHint.emitError("buffer load count for tile B must be initialized");

    const uint32_t numMfmaInst = schedHint.getNumMMAs().getValue();

    auto mfmaType = cast<RankedTensorType>(schedHint.getNumMMAs().getType());
    const uint32_t nPerXDL = mfmaType.getShape()[1];
    const uint32_t mfmaCycle = nPerXDL == 16 ? 16 : 32;

    auto dsReadsAType = cast<VectorType>(schedHint.getNumDsReadsA().getType());
    auto dsReadsBType = cast<VectorType>(schedHint.getNumDsReadsB().getType());

    const uint32_t dsReadAIssueCycle = dsReadsAType.getShape()[0] == 16 ? 8 : 4;
    const uint32_t dsReadBIssueCycle = dsReadsBType.getShape()[0] == 16 ? 8 : 4;

    const auto dsReadAMfmaRate =
        (mfmaCycle - 4 + 2 * dsReadAIssueCycle - 1) / (2 * dsReadAIssueCycle);
    const auto dsReadBMfmaRate =
        (mfmaCycle - 4 + 2 * dsReadBIssueCycle - 1) / (2 * dsReadBIssueCycle);

    const auto numDsreadAMfma =
        (numDsReadInstA + dsReadAMfmaRate - 1) / dsReadAMfmaRate;
    const auto numDsreadBMfma =
        (numDsReadInstB + dsReadBMfmaRate - 1) / dsReadBMfmaRate;

    // stage 1
    const auto numMfmaStage1 = numMfmaInst - (numDsreadAMfma + numDsreadBMfma);
    const auto numMfmaPerIssue =
        numMfmaStage1 / (numBufferLoadInstA + numBufferLoadInstB);

    const auto numDswritePerIssueA = numDsWriteInstA / numBufferLoadInstA;
    const auto numDswritePerIssueB = numDsWriteInstB / numBufferLoadInstB;

    for (size_t i = 0; i < numBufferLoadInstA; ++i) {
      for (size_t idswrite = 0; idswrite < numDswritePerIssueA; ++idswrite) {
        createSchedGroupBarrier(rewriter, loc,
                                mlir::amdgpu::sched_barrier_opt_enum::ds_write,
                                1, 0);
        createSchedGroupBarrier(rewriter, loc,
                                mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma,
                                1, 0);
      }
      createSchedGroupBarrier(
          rewriter, loc, mlir::amdgpu::sched_barrier_opt_enum::vmem_read, 1, 0);
      createSchedGroupBarrier(rewriter, loc,
                              mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma,
                              numMfmaPerIssue - numDswritePerIssueA, 0);
    }

    for (size_t i = 0; i < numBufferLoadInstB; ++i) {
      for (size_t idswrite = 0; idswrite < numDswritePerIssueB; ++idswrite) {
        createSchedGroupBarrier(rewriter, loc,
                                mlir::amdgpu::sched_barrier_opt_enum::ds_write,
                                1, 0);
        createSchedGroupBarrier(rewriter, loc,
                                mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma,
                                1, 0);
      }
      createSchedGroupBarrier(
          rewriter, loc, mlir::amdgpu::sched_barrier_opt_enum::vmem_read, 1, 0);
      createSchedGroupBarrier(rewriter, loc,
                              mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma,
                              numMfmaPerIssue - numDswritePerIssueB, 0);
    }

    // stage 2
    for (size_t i = 0; i < numDsreadAMfma; ++i) {
      if ((numDsReadInstA - (i + 1) * dsReadAMfmaRate) >= dsReadAMfmaRate) {
        createSchedGroupBarrier(rewriter, loc,
                                mlir::amdgpu::sched_barrier_opt_enum::ds_read,
                                dsReadAMfmaRate, 0);
      } else {
        createSchedGroupBarrier(
            rewriter, loc, mlir::amdgpu::sched_barrier_opt_enum::ds_read,
            numDsReadInstA - (numDsreadAMfma - 1) * dsReadAMfmaRate, 0);
      }
      createSchedGroupBarrier(
          rewriter, loc, mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma, 1, 0);
    }

    for (size_t i = 0; i < numDsreadBMfma; ++i) {
      if ((numDsReadInstB - (i + 1) * dsReadBMfmaRate) >= dsReadBMfmaRate) {
        createSchedGroupBarrier(rewriter, loc,
                                mlir::amdgpu::sched_barrier_opt_enum::ds_read,
                                dsReadBMfmaRate, 0);
      } else {
        createSchedGroupBarrier(
            rewriter, loc, mlir::amdgpu::sched_barrier_opt_enum::ds_read,
            numDsReadInstB - (numDsreadBMfma - 1) * dsReadBMfmaRate, 0);
      }
      createSchedGroupBarrier(
          rewriter, loc, mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma, 1, 0);
    }

    // The AMDGPU compiler backend can fold consecutive `ds_read/ds_write`
    // instructions into wider variants as a part of its load/store optimization
    // during the instruction selection pass. If it happens, then it means that
    // we are overestimated these types of instructions at the current level of
    // the IR. In this scenario, the inserted `sched.group.barriers` will result
    // in "fooling" the scheduling solver which can mess up the final assembly.
    // To avoid this, we switch off the backend load/store folding optimization
    // which is going to prevent instructions folding. In this case, the
    // instruction widths of `ds_read/ds_write` instructions are going to match
    // their LLVM representations. This is implemented as follows.

    // TODO: The current implementation disables `ds_read/ds_write` folding for
    // all basic blocks in the currently processed function. We should try to
    // avoid it. The compiler backend team proposed to play we the load/store
    // alignment values within the currently processed basic block as an
    // alternative solution.
    auto funcOp = schedHint->getParentOfType<LLVM::LLVMFuncOp>();
    MLIRContext *ctx = schedHint->getContext();
    llvm::SmallVector<StringAttr> targetFeatures;
    if (auto attr = funcOp.getTargetFeatures()) {
      llvm::copy(attr->getFeatures(), std::back_inserter(targetFeatures));
    }
    targetFeatures.push_back(str_attr("-load-store-opt"));
    funcOp.setTargetFeaturesAttr(
        ::mlir::LLVM::TargetFeaturesAttr::get(ctx, targetFeatures));
  }

  LogicalResult
  matchAndRewrite(triton::amdgpu::InstructionSchedHint instructionSchedHint,
                  PatternRewriter &rewriter) const override {
    if (this->schedulingType == SchedulingType::NONE) {
      rewriter.eraseOp(instructionSchedHint);
      return success();
    }

    if (this->schedulingType == SchedulingType::UNKNOWN) {
      instructionSchedHint.emitError(
          "unknown instruction scheduling variant has been provided");
      return failure();
    }

    // The switch controls whether instructions are allowed to cross the basic
    // block boundaries at the very top and at the very bottom. Note, this is
    // not supposed to be used together with IGLP OPT according to the AMDGPU
    // backend documentation.
    const bool limitSchedulingRange =
        !(schedulingType == SchedulingType::NONE ||
          schedulingType == SchedulingType::IGLP0 ||
          schedulingType == SchedulingType::IGLP1);
    Location loc = instructionSchedHint->getLoc();
    Block *block = instructionSchedHint->getBlock();
    if (limitSchedulingRange) {
      rewriter.setInsertionPointToStart(block);
      createSchedBarrier(rewriter, loc,
                         mlir::amdgpu::sched_barrier_opt_enum::none);
    }

    rewriter.setInsertionPoint(block, std::prev(block->end()));

    switch (schedulingType) {
    case SchedulingType::IGLP0:
      [[fallthrough]];
    case SchedulingType::IGLP1: {
      createIglpOpt(rewriter, loc, static_cast<int>(schedulingType) - 1);
      break;
    }
    case SchedulingType::CK_V3: {
      createCKV3Schedule(rewriter, loc, instructionSchedHint);
      break;
    }
    case SchedulingType::NONE:
      [[fallthrough]];
    default: {
      break;
    }
    }

    if (limitSchedulingRange)
      createSchedBarrier(rewriter, loc,
                         mlir::amdgpu::sched_barrier_opt_enum::none);

    rewriter.eraseOp(instructionSchedHint);
    return success();
  }

private:
  int32_t numStages;
  SchedulingType schedulingType;
};

struct TritonAMDGPULowerInstructionSchedHints
    : public triton::impl::TritonAMDGPULowerInstructionSchedHintsBase<
          TritonAMDGPULowerInstructionSchedHints> {

  explicit TritonAMDGPULowerInstructionSchedHints(int32_t numStages,
                                                  std::string variant) {
    this->numStages = numStages;
    this->variant = variant;
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

    patterns.add<InstructionSchedHintsRewriter>(ctx, this->numStages,

                                                this->variant);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {

      signalPassFailure();
    }
  }
};

struct TritonAMDGPUInsertInstructionSchedHints
    : public triton::impl::TritonAMDGPUInsertInstructionSchedHintsBase<
          TritonAMDGPUInsertInstructionSchedHints> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();

    mod.walk([this, ctx](scf::ForOp forOp) {
      // Note, instruction schedule barriers are inserted only in the case of
      // a single `tt.dot` op in a `scf::ForOp` scope in the current
      // implementation.
      if (auto dotOp = getSingleDotOpIfExists(forOp)) {
        OpBuilder rewriter(ctx);
        rewriter.setInsertionPointAfter(dotOp);
        rewriter.create<triton::amdgpu::InstructionSchedHint>(dotOp->getLoc());
      }
    });
  }
};
} // namespace

namespace mlir::triton {
std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPULowerInstructionSchedHintsPass(int32_t numStages,
                                                 std::string variant) {
  return std::make_unique<TritonAMDGPULowerInstructionSchedHints>(numStages,
                                                                  variant);
}

std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPUInsertInstructionSchedHintsPass() {
  return std::make_unique<TritonAMDGPUInsertInstructionSchedHints>();
}
} // namespace mlir::triton
