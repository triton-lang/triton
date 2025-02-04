// #include "TritonAMDGPUToLLVM/Passes.h"

// #include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
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

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-stream-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

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

SmallVector<Value> createOffset(llvm::ArrayRef<Value> valueOffset,
                                llvm::ArrayRef<int64_t> intOffset,
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

class CoordinateAux {
public:
  CoordinateAux(llvm::ArrayRef<int64_t> layout) : layout(layout) {
    bounds.resize(layout.size());
    std::exclusive_scan(layout.rbegin(), layout.rend(), bounds.begin(), 1,
                        std::multiplies<>());
  }

  SmallVector<int64_t> map(int64_t index) {
    SmallVector<int64_t> coords(bounds.size(), 0);
    for (size_t i = 1; i < bounds.size(); ++i) {
      size_t d = bounds.size() - i;
      coords[d] = index / bounds[d];
      index = index % bounds[d];
    }
    coords[0] = index;
    std::reverse(coords.begin(), coords.end());
    return coords;
  }

private:
  llvm::ArrayRef<int64_t> layout;
  std::vector<int> bounds;
};

inline bool isRowMajor(::llvm::ArrayRef<unsigned> order) {
  auto rank = order.size();
  return order[rank - 1] == 0;
}

struct DotOpMFMAConverter {
  AMDMfmaEncodingAttr mfmaLayout;
  OpBuilder &rewriter;
  Location loc;
  MLIRContext *ctx{};

  explicit DotOpMFMAConverter(AMDMfmaEncodingAttr mfmaLayout,
                              OpBuilder &rewriter, Location loc)
      : mfmaLayout(mfmaLayout), rewriter(rewriter), loc(loc),
        ctx(mfmaLayout.getContext()) {}

  LogicalResult convert(DotOp dotOp, DotOpAdaptor adaptor) const {

    InputPrecisionAttr precisionAttr = dotOp.getInputPrecisionAttr();
    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto mDim = mfmaLayout.getMDim();
    auto nDim = mfmaLayout.getNDim();

    Value a = dotOp.getA();
    Value b = dotOp.getB();
    Value c = dotOp.getC();
    Value d = dotOp.getD();

    auto localLoadA = cast<ttg::LocalLoadOp>(a.getDefiningOp());
    auto localLoadB = cast<ttg::LocalLoadOp>(b.getDefiningOp());

    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    auto cTensorTy = cast<RankedTensorType>(c.getType());
    auto dTensorTy = cast<RankedTensorType>(d.getType());

    auto elemTyA = aTensorTy.getElementType();
    auto elemTyB = bTensorTy.getElementType();
    auto elemTyC = cTensorTy.getElementType();
    auto elemTyD = dTensorTy.getElementType();

    auto encodeA = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
    auto encodeB = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
    auto encodeC = cast<AMDMfmaEncodingAttr>(cTensorTy.getEncoding());
    auto encodeD = cast<AMDMfmaEncodingAttr>(dTensorTy.getEncoding());

    auto shapeA = aTensorTy.getShape();
    auto shapeB = bTensorTy.getShape();
    auto shapeC = cTensorTy.getShape();
    auto shapeD = dTensorTy.getShape();

    int kWidth = encodeA.getKWidth();
    auto repA = mfmaLayout.getRepForOperand(aTensorTy.getShape(), kWidth, 0);
    auto repB = mfmaLayout.getRepForOperand(bTensorTy.getShape(), kWidth, 1);

    assert(repA[2] == repB[1]);

    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedC = adaptor.getC();

    auto memDescA = localLoadA->getOperand(0);
    auto memDescTypeA = cast<ttg::MemDescType>(memDescA.getType());
    auto memDescEncodingA = cast<triton::gpu::SwizzledSharedEncodingAttr>(
        memDescTypeA.getEncoding());

    auto memDescB = localLoadB->getOperand(0);
    auto memDescTypeB = cast<ttg::MemDescType>(memDescB.getType());
    auto memDescEncodingB = cast<triton::gpu::SwizzledSharedEncodingAttr>(
        memDescTypeB.getEncoding());

    // TODO: adjusting `numRepX` using `isRowMajor` is a work around.
    // it needs to be fixed in the future.
    const auto numRepM = isRowMajor(memDescEncodingA.getOrder()) ? repA[1] : 1;
    const auto numRepN = isRowMajor(memDescEncodingB.getOrder()) ? 1 : repB[2];
    const auto numRepB = repA[0];

    LDBG("numRepM: " << numRepM << "; numRepN: " << numRepN
                     << "; numRepB: " << numRepB);

    constexpr int M = 0;
    constexpr int N = 1;

    SmallVector<int64_t> refinedShapeA = {shapeA[M] / numRepM, shapeA[N]};
    SmallVector<int64_t> refinedShapeB = {shapeB[M], shapeB[N] / numRepN};
    SmallVector<int64_t> refinedShapeCD = {shapeC[M] / numRepM,
                                           shapeC[N] / numRepN};

    SmallVector<int64_t> elementsPerSlice = {refinedShapeCD[0],
                                             refinedShapeCD[1]};

    auto refinedTensorTypeA =
        RankedTensorType::get(refinedShapeA, elemTyA, encodeA);
    auto refinedTensorTypeB =
        RankedTensorType::get(refinedShapeB, elemTyB, encodeB);
    auto refinedTensorTypeC =
        RankedTensorType::get(refinedShapeCD, elemTyC, encodeC);
    auto refinedTensorTypeD =
        RankedTensorType::get(refinedShapeCD, elemTyD, encodeD);

    constexpr bool mutableMemory = true;
    auto sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);

    auto subviewTypeA = ttg::MemDescType::get(
        refinedShapeA, memDescTypeA.getElementType(),
        memDescTypeA.getEncoding(), sharedMemorySpace, mutableMemory);

    auto subviewTypeB = ttg::MemDescType::get(
        refinedShapeB, memDescTypeB.getElementType(),
        memDescTypeB.getEncoding(), sharedMemorySpace, mutableMemory);

    rewriter.setInsertionPointAfter(localLoadA);
    SmallVector<ttg::LocalLoadOp> subtilesA;
    for (int32_t i = 0; i < numRepM; ++i) {
      int32_t shift = i * elementsPerSlice[M];

      auto offset = createOffset({}, {shift, 0}, rewriter, loc);
      auto viewLoadA = rewriter.create<ttg::MemDescSubviewOp>(loc, subviewTypeA,
                                                              memDescA, offset);

      auto refinedLoadA =
          rewriter.create<ttg::LocalLoadOp>(loc, refinedTensorTypeA, viewLoadA);
      subtilesA.push_back(refinedLoadA);
    }

    rewriter.setInsertionPointAfter(localLoadB);
    SmallVector<ttg::LocalLoadOp> subtilesB;
    for (int32_t i = 0; i < numRepN; ++i) {
      int32_t shift = i * elementsPerSlice[N];
      auto offset = createOffset({}, {0, shift}, rewriter, loc);
      auto viewLoadB = rewriter.create<ttg::MemDescSubviewOp>(loc, subviewTypeB,
                                                              memDescB, offset);

      auto refinedLoadB =
          rewriter.create<ttg::LocalLoadOp>(loc, refinedTensorTypeB, viewLoadB);
      subtilesB.push_back(refinedLoadB);
    }

    rewriter.setInsertionPointAfter(dotOp);
    auto dotAttrs = dotOp->getAttrs();
    SmallVector<Value> refinedDotValues;
    for (int32_t m = 0; m < numRepM; ++m) {
      for (int32_t n = 0; n < numRepN; ++n) {
        SmallVector<int64_t> offset = {m * elementsPerSlice[M],
                                       n * elementsPerSlice[N]};
        auto refinedTensorC = rewriter.create<triton::amdgpu::ExtractSliceOp>(
            loc, Type{refinedTensorTypeC}, Value{c}, offset);

        auto refinedTensorA = subtilesA[m];
        auto refinedTensorB = subtilesB[n];

        auto result = rewriter.create<tt::DotOp>(
            loc, refinedTensorTypeD,
            ValueRange{refinedTensorA, refinedTensorB, refinedTensorC},
            dotAttrs);
        refinedDotValues.push_back(result);
      }
    }

    auto concatDims = DenseI64ArrayAttr::get(ctx, {numRepM, numRepN});
    auto joinedDotsResult = rewriter.create<triton::amdgpu::ConcatOp>(
        loc, dTensorTy, refinedDotValues, concatDims);

    d.replaceAllUsesWith(joinedDotsResult);

    // Note: dangling localLoadA or/and localLoadB (if exist)
    // should be removed by the dead code elimination pass
    dotOp.erase();
    return success();
  }
};

inline RankedTensorType rankedTType(Value tensor) {
  return cast<RankedTensorType>(tensor.getType());
};

LogicalResult rewriteMFMA(OpBuilder &rewriter, triton::DotOp op) {
  if (!(isa<DotOperandEncodingAttr>(rankedTType(op.getA()).getEncoding()) &&
        isa<DotOperandEncodingAttr>(rankedTType(op.getB()).getEncoding()))) {
    LDBG("Both $a and %b should be DotOperand layout");
    return failure();
  }

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  if (!isa<AMDMfmaEncodingAttr>(cTensorTy.getEncoding())) {
    LDBG("Currently, we only support $c with a mfma layout");
    return failure();
  }

  if (!(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
        cTensorTy.getShape()[1] == dTensorTy.getShape()[1])) {
    LDBG("DotOp's $c operand should pass the same number of values as $d");
    return failure();
  }

  auto loc = op.getLoc();
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  DotOpMFMAConverter converter(mfmaLayout, rewriter, loc);
  return converter.convert(op, DotOpAdaptor(op));
}

struct RefinedBlock {
  RefinedBlock(ArrayRef<int64_t> shape, Type elemType,
               BlockedEncodingAttr encoding)
      : encoding(encoding), elemType(elemType) {
    auto ctaOrder = encoding.getCTAOrder();
    auto warpsPerCTA = encoding.getWarpsPerCTA();
    auto threadsPerWarp = encoding.getThreadsPerWarp();
    auto sizePerThread = encoding.getSizePerThread();

    numDims = warpsPerCTA.size();
    elementsPerWorkGroup.resize(numDims);
    numPerDims.resize(numDims);
    refinedShape.resize(numDims);
    numSubTiles = 1;
    for (size_t dim = 0; dim < numDims; ++dim) {
      elementsPerWorkGroup[dim] =
          sizePerThread[dim] * threadsPerWarp[dim] * warpsPerCTA[dim];
      numPerDims[dim] = shape[dim] / elementsPerWorkGroup[dim];
      refinedShape[dim] = shape[dim] / numPerDims[dim];
      numSubTiles *= numPerDims[dim];
    }

    tensorType =
        RankedTensorType::get(elementsPerWorkGroup, elemType, encoding);
  }

  BlockedEncodingAttr encoding;
  Type elemType;
  SmallVector<int64_t> elementsPerWorkGroup;
  SmallVector<int64_t> numPerDims;
  SmallVector<int64_t> refinedShape;
  size_t numDims;
  size_t numSubTiles;
  RankedTensorType tensorType;
};

LogicalResult rewriteLoadOp(OpBuilder &rewriter, triton::LoadOp loadOp) {
  auto ctx = loadOp->getContext();
  auto loc = loadOp.getLoc();

  Value origSrc = loadOp->getOperand(0);
  Value origResult = loadOp.getResult();
  Type origResultType = loadOp.getResult().getType();
  auto origPtrs = rankedTType(origSrc);
  auto origShape = origPtrs.getShape();
  auto elemType = origPtrs.getElementType();
  auto encoding = dyn_cast<BlockedEncodingAttr>(origPtrs.getEncoding());
  if (encoding == nullptr)
    return failure();

  RefinedBlock refinedBlock(origShape, elemType, encoding);

  rewriter.setInsertionPointAfter(loadOp);
  SmallVector<Value> refinedTensors;

  Value mask = loadOp.getMask();
  Value other = loadOp.getOther();
  auto boundaryCheck = loadOp.getBoundaryCheck();
  auto padding = loadOp.getPadding();
  auto cache = loadOp.getCache();
  auto evict = loadOp.getEvict();
  auto isVolatile = loadOp.getIsVolatile();

  CoordinateAux aux(refinedBlock.numPerDims);
  for (size_t counter = 0; counter < refinedBlock.numSubTiles; ++counter) {
    auto coords = aux.map(counter);
    SmallVector<int64_t> offset(refinedBlock.numDims, 0);
    for (auto [dim, coord] : llvm::enumerate(coords)) {
      offset[dim] = coord * refinedBlock.elementsPerWorkGroup[dim];
    }

    auto slice = rewriter.create<triton::amdgpu::ExtractSliceOp>(
        loc, Type{refinedBlock.tensorType}, Value{origSrc}, offset);

    auto refinedTensor =
        rewriter.create<triton::LoadOp>(loc, slice, mask, other, boundaryCheck,
                                        padding, cache, evict, isVolatile);
    refinedTensors.push_back(refinedTensor);
  }

  auto concatDims = DenseI64ArrayAttr::get(ctx, refinedBlock.numPerDims);
  auto joinedResult = rewriter.create<triton::amdgpu::ConcatOp>(
      loc, origResultType, refinedTensors, concatDims);

  origResult.replaceAllUsesWith(joinedResult);
  return success();
}

LogicalResult rewriteLocalStoreOp(OpBuilder &rewriter,
                                  triton::gpu::LocalStoreOp loadStoreOp) {
  auto ctx = loadStoreOp->getContext();
  auto loc = loadStoreOp.getLoc();

  Value origSrc = loadStoreOp->getOperand(0);
  auto origMemViewOp =
      cast<ttg::MemDescSubviewOp>(loadStoreOp->getOperand(1).getDefiningOp());
  Value origMemView = origMemViewOp->getOperand(0);
  Value selectValue = origMemViewOp.getOffsets().front();

  auto origSrcType = rankedTType(origSrc);
  auto blockEncoding = dyn_cast<BlockedEncodingAttr>(origSrcType.getEncoding());
  if (blockEncoding == nullptr)
    return failure();

  auto origMemViewType = cast<ttg::MemDescType>(origMemView.getType());
  auto sharedEncoding = cast<triton::gpu::SwizzledSharedEncodingAttr>(
      origMemViewType.getEncoding());
  if (sharedEncoding == nullptr)
    return failure();

  RefinedBlock refinedBlock(origSrcType.getShape(),
                            origSrcType.getElementType(), blockEncoding);

  constexpr bool mutableMemory = true;
  auto sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);

  auto subviewType =
      ttg::MemDescType::get(refinedBlock.refinedShape, refinedBlock.elemType,
                            sharedEncoding, sharedMemorySpace, mutableMemory);

  rewriter.setInsertionPointAfter(loadStoreOp);
  CoordinateAux aux(refinedBlock.numPerDims);
  for (size_t counter = 0; counter < refinedBlock.numSubTiles; ++counter) {
    auto coords = aux.map(counter);
    SmallVector<int64_t> offset(refinedBlock.numDims, 0);
    for (auto [dim, coord] : llvm::enumerate(coords)) {
      offset[dim] = coord * refinedBlock.elementsPerWorkGroup[dim];
    }
    auto offsetValues = createOffset({selectValue}, offset, rewriter, loc);
    auto slicedSharedMemView = rewriter.create<ttg::MemDescSubviewOp>(
        loc, subviewType, origMemView, offsetValues);

    auto slice = rewriter.create<triton::amdgpu::ExtractSliceOp>(
        loc, Type{refinedBlock.tensorType}, Value{origSrc}, offset);

    rewriter.create<ttg::LocalStoreOp>(loc, slice, slicedSharedMemView);
  }

  loadStoreOp.erase();
  return success();
}

struct TritonAMDGPURefineOps
    : public TritonAMDGPURefineOpsBase<TritonAMDGPURefineOps> {
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

    mod->walk([&](amdgpu::InstructionSchedHint hint) {
      if (hint.getVariant() != amdgpu::SchedHint::refine_ops) {
        return WalkResult::advance();
      }

      auto *block = hint->getBlock();
      block->walk([&](triton::DotOp dotOp) {
        OpBuilder rewriter(dotOp->getContext());
        // TODO: extend to WMMA instructions
        if (failed(rewriteMFMA(rewriter, dotOp))) {
          LDBG("failed to refine tt.dotOp: " << *dotOp);
        }
      });

      block->walk([&](triton::LoadOp loadOp) {
        OpBuilder rewriter(loadOp->getContext());
        if (loadOp->getNumOperands() == 1) {
          if (failed(rewriteLoadOp(rewriter, loadOp))) {
            LDBG("failed to refine tt.loadOp: " << *loadOp);
          }
        }
      });

      block->walk([&](triton::gpu::LocalStoreOp storeOp) {
        OpBuilder rewriter(storeOp->getContext());
        if (storeOp->getNumOperands() == 2) {
          if (failed(rewriteLocalStoreOp(rewriter, storeOp))) {
            LDBG("failed to refine ttg.localLoadOp: " << *storeOp);
          }
        }
      });
      return WalkResult::advance();
    });
  }

private:
};

} // namespace

namespace mlir {

std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPURefineOpsPass(StringRef targetArch) {
  return std::make_unique<TritonAMDGPURefineOps>(targetArch);
}

} // namespace mlir
