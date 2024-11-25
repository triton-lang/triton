#include "SchedInstructions.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "Utility.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONAMDGPUINSERTINSTRUCTIONCONTROLLOGIC
#define GEN_PASS_DEF_TRITONAMDGPULOWERINSTRUCTIONSCHEDHINTS
#define GEN_PASS_DEF_TRITONAMDGPULOWERINSTRUCTIONSCHEDGUARDS
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

// The following structs represent in-source database regarding a target
// machine. It provides instructions execution and issue cycles needed for
// scheduling.
struct MachineDescr {
  virtual ~MachineDescr() = default;
  virtual uint32_t getDsReadIssueCycle(uint32_t instrWidth) = 0;
  virtual FailureOr<uint32_t> getMmaExecCycle(llvm::ArrayRef<int64_t> dims) = 0;
  virtual uint32_t getMmaIssueCycle() = 0;
  virtual uint32_t getNumLdsDataPaths() = 0;
  static std::unique_ptr<MachineDescr> get(StringRef arch);
};

template <typename Derived> struct MachineDescrImpl : MachineDescr {
  uint32_t getDsReadIssueCycle(uint32_t instrWidth) final {
    return instrWidth == 16 ? 8 : 4;
  }

  FailureOr<uint32_t> getMmaExecCycle(llvm::ArrayRef<int64_t> dims) final {
    if (dims.size() != 3)
      return failure();
    auto it =
        Derived::mmaTable.find(std::make_tuple(dims[0], dims[1], dims[2]));
    if (it != Derived::mmaTable.end())
      return it->second;
    return failure();
  }

  uint32_t getMmaIssueCycle() final { return Derived::mmaIssueCycle; };
  uint32_t getNumLdsDataPaths() final { return Derived::numLdsDataPaths; }

  using MmaTable =
      llvm::DenseMap<std::tuple<int64_t, int64_t, int64_t>, uint32_t>;
};

struct CDNA2Kind : public MachineDescrImpl<CDNA2Kind> {
  static const inline MmaTable mmaTable{{{32, 32, 8}, 64}, {{16, 16, 16}, 32}};
  static const inline uint32_t mmaIssueCycle{4};
  static const inline uint32_t numLdsDataPaths{2};
};

struct CDNA3Kind : public MachineDescrImpl<CDNA3Kind> {
  static const inline MmaTable mmaTable{{{32, 32, 8}, 32}, {{16, 16, 16}, 16}};
  static const inline uint32_t mmaIssueCycle{4};
  static const inline uint32_t numLdsDataPaths{2};
};

std::unique_ptr<MachineDescr> MachineDescr::get(StringRef arch) {
  AMD::ISAFamily family = AMD::deduceISAFamily(arch);
  switch (family) {
  case AMD::ISAFamily::CDNA3: {
    return std::make_unique<MachineDescrImpl<CDNA3Kind>>();
  }
  case AMD::ISAFamily::CDNA2: {
    return std::make_unique<MachineDescrImpl<CDNA2Kind>>();
  }
  default: {
    return nullptr;
  }
  }
  return nullptr;
}

struct InstructionSchedHintsRewriter
    : public OpRewritePattern<triton::amdgpu::InstructionSchedHint> {

  InstructionSchedHintsRewriter(MLIRContext *ctx, StringRef arch,
                                int32_t numStages)
      : OpRewritePattern(ctx), numStages(numStages) {
    this->machineDescr = MachineDescr::get(arch);
  }

  // The following is inspired by ROCm Composable Kernel library's V3 pipelining
  // (see ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3.hpp).
  // This scheduling requires 1x register and 1x LDS buffers combined with the
  // local (LDS to registers) and global (HBM to registers) data prefetching.
  LogicalResult createLocalPrefetchSchedule(
      PatternRewriter &rewriter, Location loc,
      triton::amdgpu::InstructionSchedHint schedHint) const {

    if (!(schedHint.getIsBufferLoadsAEnabled() &&
          schedHint.getIsBufferLoadsBEnabled())) {
      LDBG("skipping `local_prefetch` scheduling given it needs `buffer_load` "
           "instructions.");
      return failure();
    }

    if (!machineDescr) {
      schedHint.emitError("unknown target architecture detected");
      return failure();
    }

    const uint32_t numDsReadInstA = schedHint.getNumDsReadsA().getValue();
    const uint32_t numDsReadInstB = schedHint.getNumDsReadsB().getValue();

    const uint32_t numDsWriteInstA = schedHint.getNumDsWritesA().getValue();
    const uint32_t numDsWriteInstB = schedHint.getNumDsWritesB().getValue();

    const uint32_t numBufferLoadInstA =
        schedHint.getNumGlobalLoadsA().getValue();
    const uint32_t numBufferLoadInstB =
        schedHint.getNumGlobalLoadsB().getValue();

    if (numBufferLoadInstA == 0) {
      schedHint.emitError("buffer load count for tile A must be initialized");
      return failure();
    }

    if (numBufferLoadInstB == 0) {
      schedHint.emitError("buffer load count for tile B must be initialized");
      return failure();
    }

    const uint32_t numMmaInst = schedHint.getNumMMAs().getValue();

    auto mmaType = cast<RankedTensorType>(schedHint.getNumMMAs().getType());
    auto maybeMmaExecCycle = machineDescr->getMmaExecCycle(mmaType.getShape());
    if (llvm::failed(maybeMmaExecCycle)) {
      schedHint.emitError("unknown mma instruction type");
      return failure();
    }
    const uint32_t mmaExecCycle = maybeMmaExecCycle.value();

    auto dsReadsAType = cast<VectorType>(schedHint.getNumDsReadsA().getType());
    auto dsReadsBType = cast<VectorType>(schedHint.getNumDsReadsB().getType());

    const uint32_t dsReadAIssueCycle =
        machineDescr->getDsReadIssueCycle(dsReadsAType.getShape()[0]);
    const uint32_t dsReadBIssueCycle =
        machineDescr->getDsReadIssueCycle(dsReadsBType.getShape()[0]);

    const uint32_t mmaIssueCycle = this->machineDescr->getMmaIssueCycle();
    const uint32_t numLdsDataPaths = this->machineDescr->getNumLdsDataPaths();

    const auto dsReadAMmaRate = (mmaExecCycle - mmaIssueCycle +
                                 numLdsDataPaths * dsReadAIssueCycle - 1) /
                                (numLdsDataPaths * dsReadAIssueCycle);
    const auto dsReadBMmaRate = (mmaExecCycle - mmaIssueCycle +
                                 numLdsDataPaths * dsReadBIssueCycle - 1) /
                                (numLdsDataPaths * dsReadBIssueCycle);

    const auto numDsreadAMma =
        (numDsReadInstA + dsReadAMmaRate - 1) / dsReadAMmaRate;
    const auto numDsreadBMma =
        (numDsReadInstB + dsReadBMmaRate - 1) / dsReadBMmaRate;

    // stage 1
    const auto numMmaStage1 = numMmaInst - (numDsreadAMma + numDsreadBMma);
    const auto numMmaPerIssue =
        numMmaStage1 / (numBufferLoadInstA + numBufferLoadInstB);

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
                              numMmaPerIssue - numDswritePerIssueA, 0);
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
                              numMmaPerIssue - numDswritePerIssueB, 0);
    }

    // stage 2
    for (size_t i = 0; i < numDsreadAMma; ++i) {
      if ((numDsReadInstA - (i + 1) * dsReadAMmaRate) >= dsReadAMmaRate) {
        createSchedGroupBarrier(rewriter, loc,
                                mlir::amdgpu::sched_barrier_opt_enum::ds_read,
                                dsReadAMmaRate, 0);
      } else {
        createSchedGroupBarrier(
            rewriter, loc, mlir::amdgpu::sched_barrier_opt_enum::ds_read,
            numDsReadInstA - (numDsreadAMma - 1) * dsReadAMmaRate, 0);
      }
      createSchedGroupBarrier(
          rewriter, loc, mlir::amdgpu::sched_barrier_opt_enum::mfma_wmma, 1, 0);
    }

    for (size_t i = 0; i < numDsreadBMma; ++i) {
      if ((numDsReadInstB - (i + 1) * dsReadBMmaRate) >= dsReadBMmaRate) {
        createSchedGroupBarrier(rewriter, loc,
                                mlir::amdgpu::sched_barrier_opt_enum::ds_read,
                                dsReadBMmaRate, 0);
      } else {
        createSchedGroupBarrier(
            rewriter, loc, mlir::amdgpu::sched_barrier_opt_enum::ds_read,
            numDsReadInstB - (numDsreadBMma - 1) * dsReadBMmaRate, 0);
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
    return success();
  }

  LogicalResult
  matchAndRewrite(triton::amdgpu::InstructionSchedHint instructionSchedHint,
                  PatternRewriter &rewriter) const override {

    triton::amdgpu::SchedHint schedulingType =
        instructionSchedHint.getSchedVariant();
    if (this->numStages < 2) {
      LDBG(
          "rewriting advanced scheduling hint to 'none' given unpipelined loop "
          "due to num_stages < 2");
      schedulingType = triton::amdgpu::SchedHint::none;
    }

    // The switch controls whether instructions are allowed to cross the basic
    // block boundaries at the very top and at the very bottom. Note, this is
    // not supposed to be used together with IGLP OPT according to the AMDGPU
    // backend documentation.
    bool limitSchedulingRange =
        schedulingType == triton::amdgpu::SchedHint::local_prefetch;

    Location loc = instructionSchedHint->getLoc();
    Block *block = instructionSchedHint->getBlock();
    rewriter.setInsertionPoint(block, std::prev(block->end()));

    switch (schedulingType) {
    case triton::amdgpu::SchedHint::llvm_iglp_0:
      [[fallthrough]];
    case triton::amdgpu::SchedHint::llvm_iglp_1: {
      createIglpOpt(rewriter, loc, static_cast<int>(schedulingType) - 1);
      break;
    }
    case triton::amdgpu::SchedHint::local_prefetch: {
      LogicalResult result =
          createLocalPrefetchSchedule(rewriter, loc, instructionSchedHint);
      if (failed(result))
        limitSchedulingRange = false;
      break;
    }
    default: {
      break;
    }
    }

    auto scanResult = block->walk([](triton::amdgpu::InstructionSchedGuard) {
      return WalkResult::interrupt();
    });
    const bool isRegionAlreadyGuarded = scanResult.wasInterrupted();

    if (limitSchedulingRange && !isRegionAlreadyGuarded)
      rewriter.create<triton::amdgpu::InstructionSchedGuard>(loc);

    rewriter.eraseOp(instructionSchedHint);
    return success();
  }

private:
  int32_t numStages;
  std::unique_ptr<MachineDescr> machineDescr;
};

struct TritonAMDGPULowerInstructionSchedHints
    : public triton::impl::TritonAMDGPULowerInstructionSchedHintsBase<
          TritonAMDGPULowerInstructionSchedHints> {

  explicit TritonAMDGPULowerInstructionSchedHints(StringRef arch,
                                                  int32_t numStages) {
    this->arch = arch.str();
    this->numStages = numStages;
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalOp<triton::amdgpu::InstructionSchedHint>();
    target.addLegalOp<triton::amdgpu::InstructionSchedGuard>();
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

struct TritonAMDGPUInsertInstructionControlLogic
    : public triton::impl::TritonAMDGPUInsertInstructionControlLogicBase<
          TritonAMDGPUInsertInstructionControlLogic> {

  explicit TritonAMDGPUInsertInstructionControlLogic(
      StringRef schedVariant, bool useInstructionSchedGuards) {
    this->schedVariant = schedVariant.str();
    this->useInstructionSchedGuards = useInstructionSchedGuards;
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();

    // use the global kernel parameter to decide whether to insert instruction
    // scheduling guards
    if (this->useInstructionSchedGuards) {
      mod.walk([&](triton::FuncOp funcOp) {
        SmallVector<scf::ForOp> leafForOps = AMD::getLeafForOps(funcOp);
        for (auto forOp : leafForOps) {
          OpBuilder builder(forOp->getContext());
          Block *block = forOp.getBody();
          builder.setInsertionPoint(block, std::prev(block->end()));
          builder.create<triton::amdgpu::InstructionSchedGuard>(forOp.getLoc());
        }
      });
    }

    std::string allSchedVariants;
    llvm::raw_string_ostream os(allSchedVariants);
    constexpr auto maxNumVariants = triton::amdgpu::getMaxEnumValForSchedHint();
    for (size_t i = 0; i < maxNumVariants; ++i)
      os << triton::amdgpu::symbolizeSchedHint(i) << ", ";
    os << triton::amdgpu::symbolizeSchedHint(maxNumVariants);

    auto maybeSchedHint =
        triton::amdgpu::symbolizeSchedHint(this->schedVariant);
    if (!maybeSchedHint) {
      LDBG("skipping instruction scheduling: unknown scheduling hint. "
           "supported hints: "
           << os.str());
      return;
    }

    triton::amdgpu::SchedHint schedHint = maybeSchedHint.value();
    if (schedHint == triton::amdgpu::SchedHint::none)
      return;

    mod.walk([this, ctx, schedHint](scf::ForOp forOp) {
      // Note, instruction schedule barriers are inserted only in the case of
      // a single `tt.dot` op in a `scf::ForOp` scope in the current
      // implementation.
      if (auto dotOp = getSingleDotOpIfExists(forOp)) {
        OpBuilder rewriter(ctx);
        rewriter.setInsertionPointAfter(dotOp);
        rewriter.create<triton::amdgpu::InstructionSchedHint>(dotOp->getLoc(),
                                                              schedHint);
      }
    });
  }
};

struct InstructionSchedGuardsRewriter
    : public OpRewritePattern<triton::amdgpu::InstructionSchedGuard> {
  InstructionSchedGuardsRewriter(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::InstructionSchedGuard instructionSchedGuard,
                  PatternRewriter &rewriter) const override {

    Location loc = instructionSchedGuard->getLoc();
    Block *block = instructionSchedGuard->getBlock();
    rewriter.setInsertionPointToStart(block);
    createSchedBarrier(rewriter, loc,
                       mlir::amdgpu::sched_barrier_opt_enum::none);

    rewriter.setInsertionPoint(block, std::prev(block->end()));
    createSchedBarrier(rewriter, loc,
                       mlir::amdgpu::sched_barrier_opt_enum::none);
    rewriter.eraseOp(instructionSchedGuard);
    return success();
  }
};

struct TritonAMDGPULowerInstructionSchedGuards
    : public triton::impl::TritonAMDGPULowerInstructionSchedGuardsBase<
          TritonAMDGPULowerInstructionSchedGuards> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<ROCDL::ROCDLDialect>();
    target.addIllegalOp<triton::amdgpu::InstructionSchedGuard>();

    RewritePatternSet patterns(ctx);
    patterns.add<InstructionSchedGuardsRewriter>(ctx);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {

      signalPassFailure();
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
createTritonAMDGPUInsertInstructionControlLogicPass(
    StringRef schedVariant, bool useInstructionSchedGuards) {
  return std::make_unique<TritonAMDGPUInsertInstructionControlLogic>(
      schedVariant, useInstructionSchedGuards);
}

std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPULowerInstructionSchedGuardsPass() {
  return std::make_unique<TritonAMDGPULowerInstructionSchedGuards>();
}
} // namespace mlir::triton
