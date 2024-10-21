#include "TritonAMDGPUToLLVM/Passes.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONAMDGPUINSERTINSTRUCTIONSCHEDHINTS
#define GEN_PASS_DEF_TRITONAMDGPULOWERINSTRUCTIONSCHEDHINTS
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

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

void setNumGeneratedGlobalLoads(triton::LoadOp op, size_t globalLoadsCount,
                                Type type) {
  MLIRContext *ctx = op->getContext();
  auto counterAttr =
      triton::amdgpu::InstCounterAttr::get(ctx, globalLoadsCount, type);

  op->getBlock()->walk([&](triton::amdgpu::InstructionSchedHint schedHint) {
    auto opIdxAttr = cast<triton::amdgpu::OpIdxAttr>(
        op->getAttr(triton::amdgpu::OpIdxAttr::getMnemonic()));
    assert(opIdxAttr.getValue() < 2);
    if (opIdxAttr.getValue() == 0)
      schedHint.setNumGlobalLoadsAAttr(counterAttr);
    else
      schedHint.setNumGlobalLoadsBAttr(counterAttr);
  });
}

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
    auto opIdxAttr = op->getAttrOfType<triton::amdgpu::OpIdxAttr>(
        triton::amdgpu::OpIdxAttr::getMnemonic());
    assert(opIdxAttr.getValue() < 2);
    if (opIdxAttr.getValue() == 0)
      schedHint.setNumDsWritesAAttr(counterAttr);
    else
      schedHint.setNumDsWritesBAttr(counterAttr);
  });
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
  rewriter.create<mlir::ROCDL::SchedGroupBarrier>(loc, mask, size, groupId);
}

// Insert intrinsic that controls the types of instructions that may be
// allowed to cross the intrinsic during instruction scheduling.
Operation *createSchedBarrier(PatternRewriter &rewriter, Location loc,
                              mlir::amdgpu::sched_barrier_opt_enum maskValue) {
  IntegerAttr mask =
      rewriter.getI32IntegerAttr(static_cast<int32_t>(maskValue));
  return rewriter.create<mlir::ROCDL::SchedBarrier>(loc, mask);
}

// Insert an experimental intrinsic for instruction group level parallelism.
// The intrinsic takes a value that specifies the strategy.
Operation *createIglpOpt(PatternRewriter &rewriter, Location loc, int value) {
  IntegerAttr iglpValue =
      rewriter.getI32IntegerAttr(static_cast<int32_t>(value));
  return rewriter.create<mlir::ROCDL::IglpOpt>(loc, iglpValue);
}

struct InstructionSchedHintsRewriter
    : public OpRewritePattern<triton::amdgpu::InstructionSchedHint> {

  InstructionSchedHintsRewriter(mlir::MLIRContext *ctx, std::string variant)
      : OpRewritePattern(ctx) {
    std::transform(variant.begin(), variant.end(), variant.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    this->schedulingType = llvm::StringSwitch<SchedulingType>(variant)
                               .Case("default", SchedulingType::NONE)
                               .Case("iglp0", SchedulingType::IGLP0)
                               .Case("iglp1", SchedulingType::IGLP1)
                               .Case("ck_v3", SchedulingType::CK_V3)
                               .Default(SchedulingType::UNKNOWN);
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
  // local (LDS to registers) and global (HBN to registers) data prefetching.
  // see:
  // include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3.h
  void
  createCKV3Schedule(PatternRewriter &rewriter, Location loc,
                     triton::amdgpu::InstructionSchedHint schedHint) const {
    const uint32_t numDsReadInstA = schedHint.getNumDsReadsA().getValue();
    const uint32_t numDsReadInstB = schedHint.getNumDsReadsB().getValue();

    const uint32_t numDsWriteInstA = schedHint.getNumDsWritesA().getValue();
    const uint32_t numDsWriteInstB = schedHint.getNumDsWritesB().getValue();

    const uint32_t numBufferLoadInstA =
        schedHint.getNumGlobalLoadsA().getValue();
    const uint32_t numBufferLoadInstB =
        schedHint.getNumGlobalLoadsB().getValue();

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
    const auto num_mfma_per_issue =
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
                              num_mfma_per_issue - numDswritePerIssueA, 0);
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
                              num_mfma_per_issue - numDswritePerIssueB, 0);
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
  }

  LogicalResult
  matchAndRewrite(triton::amdgpu::InstructionSchedHint instructionSchedHint,
                  PatternRewriter &rewriter) const override {

    if (this->schedulingType == SchedulingType::UNKNOWN) {
      llvm::dbgs()
          << "[" << getDebugName() << "]: "
          << "unknown instruction scheduling variant has been provided\n";
      return mlir::failure();
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
    return mlir::success();
  }

private:
  SchedulingType schedulingType;
};

struct TritonAMDGPULowerInstructionSchedHints
    : public triton::impl::TritonAMDGPULowerInstructionSchedHintsBase<
          TritonAMDGPULowerInstructionSchedHints> {

  explicit TritonAMDGPULowerInstructionSchedHints(std::string variant) {
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
    patterns.add<InstructionSchedHintsRewriter>(ctx, this->variant);

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
      triton::DotOp dot = nullptr;
      size_t dotCounter = 0;
      forOp->walk([&dot, &dotCounter](triton::DotOp op) {
        dot = op;
        ++dotCounter;
      });
      // Note, instruction schedule barriers are inserted only in the case of
      // a single `tt.dot` op in a `scf::ForOp` scope in the current
      // implementation.
      if (dotCounter == 1) {
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
createTritonAMDGPULowerInstructionSchedHintsPass(std::string variant) {
  return std::make_unique<TritonAMDGPULowerInstructionSchedHints>(variant);
}

std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPUInsertInstructionSchedHintsPass() {
  return std::make_unique<TritonAMDGPUInsertInstructionSchedHints>();
}
} // namespace mlir::triton
