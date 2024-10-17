#include "TritonAMDGPUToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
  auto counterAttr = amdgpu::InstCounterAttr::get(ctx, mmaCount, mmaType);

  op->getBlock()->walk([&](amdgpu::InstructionSchedHint schedHint) {
    schedHint.setNumMMAsAttr(counterAttr);
  });
}

void setNumGeneratedGlobalLoads(triton::LoadOp op, size_t globalLoadsCount,
                                Type type) {
  MLIRContext *ctx = op->getContext();
  auto counterAttr = amdgpu::InstCounterAttr::get(ctx, globalLoadsCount, type);

  op->getBlock()->walk([&](amdgpu::InstructionSchedHint schedHint) {
    auto opIdxAttr =
        cast<amdgpu::OpIdxAttr>(op->getAttr(amdgpu::OpIdxAttr::getMnemonic()));
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
  auto counterAttr = amdgpu::InstCounterAttr::get(ctx, dsReadsCount, type);

  op->getBlock()->walk([&](amdgpu::InstructionSchedHint schedHint) {
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
  auto counterAttr = amdgpu::InstCounterAttr::get(ctx, localStoreOpCount, type);

  op->getBlock()->walk([&](amdgpu::InstructionSchedHint schedHint) {
    auto opIdxAttr =
        op->getAttrOfType<amdgpu::OpIdxAttr>(amdgpu::OpIdxAttr::getMnemonic());
    assert(opIdxAttr.getValue() < 2);
    if (opIdxAttr.getValue() == 0)
      schedHint.setNumDsWritesAAttr(counterAttr);
    else
      schedHint.setNumDsWritesBAttr(counterAttr);
  });
}
} // namespace mlir::triton

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
  const char *intrinsicName = "llvm.amdgcn.sched.group.barrier";

  Value mask =
      LLVM::createConstantI32(loc, rewriter, static_cast<int32_t>(maskValue));
  Value size =
      LLVM::createConstantI32(loc, rewriter, static_cast<int32_t>(sizeValue));
  Value groupId = LLVM::createConstantI32(loc, rewriter,
                                          static_cast<int32_t>(groupIdValue));

  LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsicName, TypeRange{},
                                  ValueRange{mask, size, groupId});
}

// Insert intrinsic that controls the types of instructions that may be
// allowed to cross the intrinsic during instruction scheduling.
Operation *createSchedBarrier(PatternRewriter &rewriter, Location loc,
                              int64_t maskValue) {
  MLIRContext *ctx = rewriter.getContext();
  const char *intrinsicName = "llvm.amdgcn.sched.barrier";
  LLVM::FastmathFlagsAttr defaultFlags{};

  Value mask =
      LLVM::createConstantI32(loc, rewriter, static_cast<int32_t>(maskValue));
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsicName,
                                         TypeRange{}, ValueRange{mask});
}

// Insert an experimental intrinsic for instruction group level parallelism.
// The intrinsic takes a value that specifies the strategy.
Operation *createIglpOpt(PatternRewriter &rewriter, Location loc, int value) {
  MLIRContext *ctx = rewriter.getContext();
  const char *intrinsicName = "llvm.amdgcn.iglp.opt";
  LLVM::FastmathFlagsAttr defaultFlags{};
  Value iglpValue =
      LLVM::createConstantI32(loc, rewriter, static_cast<int32_t>(value));
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsicName,
                                         TypeRange{}, ValueRange{iglpValue});
}

struct InstructionSchedHintsRewriter
    : public OpRewritePattern<amdgpu::InstructionSchedHint> {

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
  void createCKV3Schedule(PatternRewriter &rewriter, Location loc,
                          amdgpu::InstructionSchedHint schedHint) const {
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
        createSchedGroupBarrier(rewriter, loc, InstructionKindMask::DS_WRITE, 1,
                                0);
        createSchedGroupBarrier(rewriter, loc, InstructionKindMask::MFMA, 1, 0);
      }
      createSchedGroupBarrier(rewriter, loc, InstructionKindMask::VMEM_READ, 1,
                              0);
      createSchedGroupBarrier(rewriter, loc, InstructionKindMask::MFMA,
                              num_mfma_per_issue - numDswritePerIssueA, 0);
    }

    for (size_t i = 0; i < numBufferLoadInstB; ++i) {
      for (size_t idswrite = 0; idswrite < numDswritePerIssueB; ++idswrite) {
        createSchedGroupBarrier(rewriter, loc, InstructionKindMask::DS_WRITE, 1,
                                0);
        createSchedGroupBarrier(rewriter, loc, InstructionKindMask::MFMA, 1, 0);
      }
      createSchedGroupBarrier(rewriter, loc, InstructionKindMask::VMEM_READ, 1,
                              0);
      createSchedGroupBarrier(rewriter, loc, InstructionKindMask::MFMA,
                              num_mfma_per_issue - numDswritePerIssueB, 0);
    }

    // stage 2
    for (size_t i = 0; i < numDsreadAMfma; ++i) {
      if ((numDsReadInstA - (i + 1) * dsReadAMfmaRate) >= dsReadAMfmaRate) {
        createSchedGroupBarrier(rewriter, loc, InstructionKindMask::DS_READ,
                                dsReadAMfmaRate, 0);
      } else {
        createSchedGroupBarrier(
            rewriter, loc, InstructionKindMask::DS_READ,
            numDsReadInstA - (numDsreadAMfma - 1) * dsReadAMfmaRate, 0);
      }
      createSchedGroupBarrier(rewriter, loc, InstructionKindMask::MFMA, 1, 0);
    }

    for (size_t i = 0; i < numDsreadBMfma; ++i) {
      if ((numDsReadInstB - (i + 1) * dsReadBMfmaRate) >= dsReadBMfmaRate) {
        createSchedGroupBarrier(rewriter, loc, InstructionKindMask::DS_READ,
                                dsReadBMfmaRate, 0);
      } else {
        createSchedGroupBarrier(
            rewriter, loc, InstructionKindMask::DS_READ,
            numDsReadInstB - (numDsreadBMfma - 1) * dsReadBMfmaRate, 0);
      }
      createSchedGroupBarrier(rewriter, loc, InstructionKindMask::MFMA, 1, 0);
    }
  }

  LogicalResult
  matchAndRewrite(amdgpu::InstructionSchedHint instructionSchedHint,
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
      createSchedBarrier(rewriter, loc, InstructionKindMask::NONE);
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
      createSchedBarrier(rewriter, loc, InstructionKindMask::NONE);

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
    target.addIllegalOp<amdgpu::InstructionSchedHint>();

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
        rewriter.create<amdgpu::InstructionSchedHint>(dot->getLoc());
        annotateDotUsageOnLoadStore(forOp);
      }
    });
  }

  template <typename Type> bool isOf(Operation *op) const {
    return llvm::isa<Type>(op);
  }

  template <typename... Types>
  llvm::SmallVector<Operation *> getUsersOfTypes(Value value) const {
    llvm::SmallVector<Operation *> concreteUsers;
    for (auto user : value.getUsers()) {
      std::vector<bool> values = {(isOf<Types>(user), ...)};
      if (llvm::any_of(values, [](bool value) { return value; }))
        concreteUsers.push_back(user);
    }
    return concreteUsers;
  }

  template <typename Type>
  llvm::SmallVector<Type> getUsersOfType(Value value) const {
    auto users = getUsersOfTypes<Type>(value);
    llvm::SmallVector<Type> concreteUsers;
    for (auto user : getUsersOfTypes<Type>(value)) {
      concreteUsers.push_back(cast<Type>(user));
    }
    return concreteUsers;
  }

  // Go through a single use chain of `convert_layout` and/or `fp_to_fp` Ops to
  // get the final value after all conversions
  Value rewindUnaryOps(Value value) const {
    auto unaryOps =
        getUsersOfTypes<triton::gpu::ConvertLayoutOp, triton::FpToFpOp>(value);
    while (!unaryOps.empty()) {
      assert(unaryOps.size() == 1);
      value = unaryOps[0]->getResult(0);
      unaryOps =
          getUsersOfTypes<triton::gpu::ConvertLayoutOp, triton::FpToFpOp>(
              value);
    }
    return value;
  }

  // Given a `scf::ForOp`, the method finds and annotates all Ops which produce
  // input values for the `tt.dot` operation. The algorithm handles software
  // pipelining. Therefore, we start by tracking `tt.load` ops and unwind the
  // data flow by looking up to the yielded values and iteration arguments of a
  // given `scf::ForOp` till we find `ttg.local_store` Op. Once a
  // `ttg.local_store` Op is found, we need a single yielded-arguments lookup to
  // find the corresponding `ttg.local_load` Op from which we have a direct data
  // flow path to the target `tt.dot` op. At this point, we can annotate all
  // found Ops (i.e., `tt.load`, `ttg.local_store`) with the input argument
  // index of the data to `tt.dot` Op. Here is an example of the resulting
  // annotated TTGIR:
  //
  //  %13:8 = scf.for %arg11 = %c0_i32 to %0 step %c1_i32 iter_args(
  //    %arg0 = %cst_1, %arg1 = %in_0, %arg2 = %in_1, %arg3 = %c0_i32,
  //    %arg4 = %in_2, %arg5 = %in_3, %arg6 = %in_4, %arg7 = %in_5)
  //    -> (...)  : i32 {
  //    %1 = triton_gpu.local_load %arg4 : {OpIdx = 0}
  //    %2 = triton_gpu.local_load %arg5 : {OpIdx = 1}
  //    %3 = tt.dot %1, %2, %arg0
  //    %4 = tt.addptr %arg1, %cst
  //    %5 = tt.addptr %arg2, %cst_0
  //    %6 = tt.load %4 : {OpIdx = 0}
  //    %7 = tt.load %5 : {OpIdx = 1}
  //    %8 = arith.addi %arg3, %c1_i32
  //    %9 = arith.cmpi slt, %8, %c2_i32
  //    %10 = arith.select %9, %8, %c0_i32
  //    %11 = triton_gpu.memdesc_subview %56[%10, %c0_i32, %c0_i32]
  //    triton_gpu.local_store %arg6, %11 : {OpIdx = 0}
  //    %12 = triton_gpu.memdesc_subview %57[%10, %c0_i32, %c0_i32]
  //    triton_gpu.local_store %arg7, %12 : {OpIdx = 1}
  //    scf.yield %3, %4, %5, %10, %11, %12, %6, %7 : (...)
  //  }
  //
  // Note, this is required for counting issued `llvm` instructions during
  // lowering from TTGIR to LLVM dialects to perform advanced instruction
  // scheduling.
  void annotateDotUsageOnLoadStore(scf::ForOp forOp) const {
    llvm::SmallVector<triton::LoadOp> loadOps;
    forOp.walk(
        [&loadOps](triton::LoadOp loadOp) { loadOps.push_back(loadOp); });

    ValueRange yieldedValues = forOp.getYieldedValues();
    auto initArgs = forOp.getRegionIterArgs();

    MLIRContext *ctx = forOp->getContext();
    mlir::OpBuilder rewriter(ctx);

    for (auto loadOp : loadOps) {
      Value loadResult = loadOp.getResult();

      // Unwind till the first carried loop iteration regarding `tt.load`.
      Value loopCarriedLoadValue = loadResult;
      bool foundFirstCarriedLoopIteration = false;
      while (!foundFirstCarriedLoopIteration) {
        auto it = llvm::find(yieldedValues, loopCarriedLoadValue);
        if (it != yieldedValues.end()) {
          size_t idx = std::distance(yieldedValues.begin(), it);
          loopCarriedLoadValue = initArgs[idx];
        } else {
          foundFirstCarriedLoopIteration = true;
        }
      }

      loopCarriedLoadValue = rewindUnaryOps(loopCarriedLoadValue);
      assert(loopCarriedLoadValue.hasOneUse());

      // Handle pipelining - i.e., `local_store`, `memdesc_subview`,
      // `local_load` ops.
      triton::gpu::LocalLoadOp localLoadOp = nullptr;
      auto loadOpUser = *(loopCarriedLoadValue.user_begin());
      auto localStoreOp = llvm::dyn_cast<triton::gpu::LocalStoreOp>(loadOpUser);
      if (localStoreOp) {
        auto subviewOp = localStoreOp.getDst()
                             .getDefiningOp<triton::gpu::MemDescSubviewOp>();
        Value subviewResult = subviewOp.getResult();
        auto it = llvm::find(yieldedValues, subviewResult);
        if (it != yieldedValues.end()) {
          size_t idx = std::distance(yieldedValues.begin(), it);
          Value loopCarriedSubviewValue = initArgs[idx];

          auto subviewLoadOps =
              getUsersOfType<triton::gpu::LocalLoadOp>(loopCarriedSubviewValue);
          assert(subviewLoadOps.size() == 1);
          localLoadOp = *subviewLoadOps.begin();

          loopCarriedLoadValue = localLoadOp.getResult();
        } else {
          auto localLoadOps =
              getUsersOfType<triton::gpu::LocalLoadOp>(subviewResult);
          assert(localLoadOps.size() == 1);
          localLoadOp = *localLoadOps.begin();
          auto it = llvm::find(yieldedValues, localLoadOp.getResult());
          assert(it != yieldedValues.end());
          size_t idx = std::distance(yieldedValues.begin(), it);
          loopCarriedLoadValue = initArgs[idx];
        }
        loopCarriedLoadValue = rewindUnaryOps(loopCarriedLoadValue);
      }

      // Find the corresponding `DotOp`.
      auto dots = getUsersOfType<triton::DotOp>(loopCarriedLoadValue);
      assert(dots.size() == 1);

      // Find which `DotOp` argument the current `loadOp` belongs to.
      auto dotOperands = dots.begin()->getOperands();
      auto it = llvm::find(dotOperands, loopCarriedLoadValue);
      assert(it != dotOperands.end());
      size_t opIdx = std::distance(dotOperands.begin(), it);

      // Set `OpIdx` attributes.
      auto opIdxAttr = amdgpu::OpIdxAttr::get(ctx, opIdx);

      loadOp->setAttr(amdgpu::OpIdxAttr::getMnemonic(), opIdxAttr);
      if (localStoreOp)
        localStoreOp->setAttr(amdgpu::OpIdxAttr::getMnemonic(), opIdxAttr);
    }
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
