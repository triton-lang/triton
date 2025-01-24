#include "TritonAMDGPUToLLVM/Passes.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
// #include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
// #include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
// #include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
// #include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
// #include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
// #include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
// #include "triton/Analysis/Allocation.h"
// #include "triton/Analysis/AxisInfo.h"
// #include "triton/Analysis/Membar.h"
// #include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
// #include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/include/TritonAMDGPUTransforms/MfmaGroup.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONAMDGPUREFINEOPS
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

// TODO: take the implementation from `ReorderInstructions.cpp`
static SmallVector<scf::ForOp> getLeafForOps(triton::FuncOp funcOp) {
  SmallVector<scf::ForOp> allOps;
  funcOp->walk([&](scf::ForOp forOp) { allOps.push_back(forOp); });

  SmallVector<scf::ForOp> leafOps;
  for (scf::ForOp forOp : allOps) {
    auto searchResult = forOp.getBody()->walk(
        [](scf::ForOp) { return WalkResult::interrupt(); });
    if (!searchResult.wasInterrupted())
      leafOps.push_back(forOp);
  }
  return leafOps;
}

// TODO: refactor. We should rely on some other indices (i.e., don't use OpIdx
// for this purpose)
using LocalAllocTable =
    DenseMap<Operation *, std::pair<ttg::LocalAllocOp, Value>>;
LocalAllocTable getLocalAllocTable(scf::ForOp forOp) {
  LocalAllocTable allocTable;
  DenseMap<uint32_t, ttg::LocalAllocOp> storeTable;
  Value selectorValue = nullptr;
  forOp->walk([&](ttg::LocalStoreOp op) {
    if (auto opIdx = op->getAttrOfType<triton::amdgpu::OpIdxAttr>(
            triton::amdgpu::OpIdxAttr::getMnemonic())) {
      auto subviewOp =
          cast<ttg::MemDescSubviewOp>(op->getOperand(1).getDefiningOp());
      selectorValue = subviewOp.getOffsets().front();
      auto localAlloc =
          cast<ttg::LocalAllocOp>(subviewOp.getOperand(0).getDefiningOp());

      storeTable.insert({opIdx.getValue(), localAlloc});
      allocTable.insert({op, {localAlloc, selectorValue}});
    }
  });

  forOp->walk([&](ttg::LocalLoadOp op) {
    Value dst = op.getResult();
    auto dstTensorTy = cast<RankedTensorType>(dst.getType());
    auto dotOperandLayout =
        cast<DotOperandEncodingAttr>(dstTensorTy.getEncoding());
    const size_t opIdx = dotOperandLayout.getOpIdx();
    ttg::LocalAllocOp localAllocOp = storeTable.find(opIdx)->second;

    for (auto [idx, value] : llvm::enumerate(forOp.getYieldedValues())) {
      if (selectorValue == value) {
        selectorValue = forOp.getRegionIterArgs()[idx];
        break;
      }
    }

    allocTable.insert({op, {localAllocOp, selectorValue}});
  });

  return allocTable;
}

SmallVector<Value> createOffset(llvm::ArrayRef<Value> valueOffset,
                                llvm::ArrayRef<int32_t> intOffset,
                                OpBuilder &rewriter, Location loc) {
  SmallVector<Value> values;
  for (auto item : valueOffset) {
    values.push_back(item);
  }

  for (auto item : intOffset) {
    Value value = rewriter.create<arith::ConstantIntOp>(loc, item, 32);
    values.push_back(value);
  }
  return values;
}

struct DotOpMFMAConverter {
  AMDMfmaEncodingAttr mfmaLayout;
  LocalAllocTable &allocTable;
  OpBuilder &rewriter;
  Location loc;
  MLIRContext *ctx{};

  explicit DotOpMFMAConverter(AMDMfmaEncodingAttr mfmaLayout,
                              LocalAllocTable &allocTable, OpBuilder &rewriter,
                              Location loc)
      : mfmaLayout(mfmaLayout), allocTable(allocTable), rewriter(rewriter),
        loc(loc), ctx(mfmaLayout.getContext()) {}

  void convert(DotOp dotOp, DotOpAdaptor adaptor) const {

    InputPrecisionAttr precisionAttr = dotOp.getInputPrecisionAttr();
    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto mDim = mfmaLayout.getMDim();
    auto nDim = mfmaLayout.getNDim();
    // assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
    //        (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));

    SmallVector<int64_t> elementsPerWarp = {mDim * warpsPerCTA[0],
                                            nDim * warpsPerCTA[1]};

    Value a = dotOp.getA();
    Value b = dotOp.getB();
    Value c = dotOp.getC();
    Value d = dotOp.getD();

    auto localLoadA = cast<ttg::LocalLoadOp>(a.getDefiningOp());
    auto localLoadB = cast<ttg::LocalLoadOp>(b.getDefiningOp());
    if (!(allocTable.contains(localLoadA) && allocTable.contains(localLoadB))) {
      return;
    }

    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    auto cTensorTy = cast<RankedTensorType>(c.getType());
    auto dTensorTy = cast<RankedTensorType>(d.getType());

    auto elemTyA = aTensorTy.getElementType();
    auto elemTyB = bTensorTy.getElementType();
    auto elemTyC = cTensorTy.getElementType();
    auto elemTyD = dTensorTy.getElementType();

    auto encodeA = aTensorTy.getEncoding();
    auto encodeB = bTensorTy.getEncoding();
    auto encodeC = cTensorTy.getEncoding();
    auto encodeD = dTensorTy.getEncoding();

    auto shapeA = aTensorTy.getShape();
    auto shapeB = bTensorTy.getShape();
    auto shapeC = cTensorTy.getShape();
    auto shapeD = dTensorTy.getShape();

    int32_t numRepM = shapeA[0] / elementsPerWarp[0];
    int32_t numRepN = shapeB[1] / elementsPerWarp[1];

    constexpr int M = 0;
    constexpr int N = 1;

    SmallVector<int64_t> refinedShapeA = {shapeA[M] / numRepM, shapeA[N]};
    SmallVector<int64_t> refinedShapeB = {shapeB[M], shapeB[N] / numRepN};
    SmallVector<int64_t> refinedShapeCD = {shapeC[M] / numRepM,
                                           shapeC[N] / numRepN};

    auto refinedTensorTypeA =
        RankedTensorType::get(refinedShapeA, elemTyA, encodeA);
    auto refinedTensorTypeB =
        RankedTensorType::get(refinedShapeB, elemTyB, encodeB);
    auto refinedTensorTypeC =
        RankedTensorType::get(refinedShapeCD, elemTyC, encodeC);
    auto refinedTensorTypeD =
        RankedTensorType::get(refinedShapeCD, elemTyD, encodeD);

    auto sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);

    ttg::LocalAllocOp allocOpA = allocTable.find(localLoadA)->second.first;
    Value valueSelectorA = allocTable.find(localLoadA)->second.second;
    ttg::MemDescType allocTypeA =
        cast<ttg::MemDescType>(allocOpA.getResult().getType());

    ttg::LocalAllocOp allocOpB = allocTable.find(localLoadB)->second.first;
    Value valueSelectorB = allocTable.find(localLoadB)->second.second;
    ttg::MemDescType allocTypeB =
        cast<ttg::MemDescType>(allocOpB.getResult().getType());

    constexpr bool mutableMemory = true;
    auto subviewTypeA = ttg::MemDescType::get(
        refinedShapeA, allocTypeA.getElementType(), allocTypeA.getEncoding(),
        sharedMemorySpace, mutableMemory);

    auto subviewTypeB = ttg::MemDescType::get(
        refinedShapeB, allocTypeB.getElementType(), allocTypeB.getEncoding(),
        sharedMemorySpace, mutableMemory);

    rewriter.setInsertionPointAfter(localLoadA);
    SmallVector<ttg::LocalLoadOp> subtilesA;
    for (int32_t i = 0; i < numRepM; ++i) {
      int32_t shift = i * elementsPerWarp[M];
      auto offset = createOffset({valueSelectorA}, {shift, 0}, rewriter, loc);
      auto viewLoadA = rewriter.create<ttg::MemDescSubviewOp>(
          loc, subviewTypeA, allocOpA.getResult(), offset);

      auto refinedLoadA =
          rewriter.create<ttg::LocalLoadOp>(loc, refinedTensorTypeA, viewLoadA);
      subtilesA.push_back(refinedLoadA);
    }

    rewriter.setInsertionPointAfter(localLoadB);
    SmallVector<ttg::LocalLoadOp> subtilesB;
    for (int32_t i = 0; i < numRepN; ++i) {
      int32_t shift = i * elementsPerWarp[N];
      auto offset = createOffset({valueSelectorB}, {0, shift}, rewriter, loc);
      auto viewLoadB = rewriter.create<ttg::MemDescSubviewOp>(
          loc, subviewTypeB, allocOpB.getResult(), offset);

      auto refinedLoadB =
          rewriter.create<ttg::LocalLoadOp>(loc, refinedTensorTypeB, viewLoadB);
      subtilesB.push_back(refinedLoadB);
    }

    rewriter.setInsertionPointAfter(dotOp);
    SmallVector<Value> refinedDotValues;
    for (int32_t m = 0; m < numRepM; ++m) {
      for (int32_t n = 0; n < numRepN; ++n) {
        SmallVector<int64_t> offset = {m * elementsPerWarp[M],
                                       n * elementsPerWarp[N]};
        auto refinedTensorC = rewriter.create<triton::amdgpu::ExtractSliceOp>(
            loc, Type{refinedTensorTypeC}, Value{c}, offset);

        auto refinedTensorA = subtilesA[m];
        auto refinedTensorB = subtilesB[n];

        auto result = rewriter.create<tt::DotOp>(loc, refinedTensorTypeD,
                                                 refinedTensorA, refinedTensorB,
                                                 refinedTensorC, precisionAttr);
        refinedDotValues.push_back(result);
      }
    }

    auto concatDims = DenseI32ArrayAttr::get(ctx, {numRepM, numRepN});
    auto joinedDotsResult = rewriter.create<triton::amdgpu::ConcatOp>(
        loc, dTensorTy, refinedDotValues, concatDims);

    d.replaceAllUsesWith(joinedDotsResult);

    // remove old ops
    // Note: dangling localLoadA or/and localLoadB (if exist)
    // should be removed by the dead code elimination pass
    dotOp.erase();
  }
};

void convertMFMA(OpBuilder &rewriter, triton::DotOp op,
                 LocalAllocTable &localAllocTable) {
  auto rankedTType = [](Value tensor) {
    return cast<RankedTensorType>(tensor.getType());
  };

  assert(isa<DotOperandEncodingAttr>(rankedTType(op.getA()).getEncoding()) &&
         isa<DotOperandEncodingAttr>(rankedTType(op.getB()).getEncoding()) &&
         "Both $a and %b should be DotOperand layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(isa<AMDMfmaEncodingAttr>(cTensorTy.getEncoding()) &&
         "Currently, we only support $c with a mfma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  auto loc = op.getLoc();
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  DotOpMFMAConverter converter(mfmaLayout, localAllocTable, rewriter, loc);
  converter.convert(op, DotOpAdaptor(op));
}

struct TritonAMDGPURefineOps
    : public triton::impl::TritonAMDGPURefineOpsBase<TritonAMDGPURefineOps> {
  explicit TritonAMDGPURefineOps(StringRef targetArch) {
    this->arch = targetArch.str();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::triton::AMD::TargetInfo targetInfo(this->arch.getValue());
    if (targetInfo.getISAFamily() == mlir::triton::AMD::ISAFamily::Unknown) {
      mod.emitError("unsupported target: '") << this->arch.getValue() << "'";
      return signalPassFailure();
    }

    mod->walk([&](triton::FuncOp funcOp) {
      SmallVector<scf::ForOp> forOps = getLeafForOps(funcOp);
      for (auto forOp : forOps) {
        auto allocTable = getLocalAllocTable(forOp);

        // handle loops which operate on local memory
        if (allocTable.empty())
          continue;

        forOp.walk([&](triton::DotOp dotOp) {
          OpBuilder rewriter(dotOp->getContext());

          // TODO: extend to WMMA instructions
          convertMFMA(rewriter, dotOp, allocTable);
        });
      }
    });
  }

private:
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPURefineOpsPass(StringRef targetArch) {
  return std::make_unique<TritonAMDGPURefineOps>(targetArch);
}

} // namespace mlir::triton
