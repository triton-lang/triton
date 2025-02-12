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
#include "third_party/amd/include/TritonAMDGPUTransforms/DotTiling.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-refine-ops"
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

LogicalResult rewriteLocalLoad(OpBuilder &rewriter,
                               triton::gpu::LocalLoadOp op) {
  auto *ctx = op->getContext();
  auto loc = op->getLoc();

  auto resultType = cast<RankedTensorType>(op.getType());
  auto resultElementType = resultType.getElementType();
  auto resultEncode = cast<DotOperandEncodingAttr>(resultType.getEncoding());
  auto resultShape = resultType.getShape();
  auto opIdx = resultEncode.getOpIdx();

  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(resultEncode.getParent());
  int kWidth = resultEncode.getKWidth();
  auto reps = mfmaLayout.getRepForOperand(resultShape, kWidth, opIdx);

  auto memDesc = op->getOperand(0);
  auto memDescType = cast<ttg::MemDescType>(memDesc.getType());
  auto memDescEncoding =
      cast<triton::gpu::SwizzledSharedEncodingAttr>(memDescType.getEncoding());

  SmallVector<int64_t> refinedShape(resultShape);
  SmallVector<int64_t> numReps = {1, 1};
  bool rowMajor = isRowMajor(memDescEncoding.getOrder());
  auto sliceAxis = opIdx == 0 ? 0 : 1;
  if (opIdx == 0) {
    numReps[sliceAxis] = rowMajor ? reps[1] : 1;
    refinedShape[sliceAxis] /= numReps[sliceAxis];
  } else {
    numReps[sliceAxis] = rowMajor ? 1 : reps[2];
    refinedShape[sliceAxis] /= numReps[sliceAxis];
  }
  auto elementsPerSlice = refinedShape[sliceAxis];
  LDBG("LocalLoad: numRep: " << numReps[sliceAxis] << " : " << *op);

  auto refinedTensorType =
      RankedTensorType::get(refinedShape, resultElementType, resultEncode);

  constexpr bool mutableMemory = true;
  auto sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
  auto subviewType = ttg::MemDescType::get(
      refinedShape, memDescType.getElementType(), memDescType.getEncoding(),
      sharedMemorySpace, mutableMemory);

  rewriter.setInsertionPointAfter(op);
  SmallVector<Value> subtiles;
  for (int32_t i = 0; i < numReps[sliceAxis]; ++i) {
    int32_t shift = i * elementsPerSlice;
    int64_t offset0 = sliceAxis == 0 ? shift : 0;
    int64_t offset1 = sliceAxis == 0 ? 0 : shift;
    auto offset = createOffset({}, {offset0, offset1}, rewriter, loc);
    auto refinedView = rewriter.create<ttg::MemDescSubviewOp>(loc, subviewType,
                                                              memDesc, offset);
    auto refinedLoad =
        rewriter.create<ttg::LocalLoadOp>(loc, refinedTensorType, refinedView);
    subtiles.push_back(refinedLoad);
  }

  auto concatDims = DenseI64ArrayAttr::get(ctx, numReps);
  auto joinedResult = rewriter.create<triton::amdgpu::ConcatOp>(
      loc, resultType, subtiles, concatDims);

  op.replaceAllUsesWith(joinedResult.getResult());
  return success();
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

    const auto numRepM = repA[1];
    const auto numRepN = repB[2];
    const auto numRepK = repA[2];
    const auto numRepB = repA[0];
    SmallVector<int64_t> numRepShape = {numRepM , numRepN , numRepK};
    DBGS() << "Dot-Tiling\n";
    DBGS() << "totalReps: "
      << numRepShape[0] << "x"
      << numRepShape[1] << "x"
      << numRepShape[2] << "\n";
    SmallVector<int64_t> refinedShapeA = {shapeA[0] / numRepM, shapeA[1] / numRepK};
    SmallVector<int64_t> refinedShapeB = {shapeB[0] / numRepK, shapeB[1] / numRepN};
    SmallVector<int64_t> refinedShapeCD = {shapeC[0] / numRepM, shapeC[1] / numRepN};

    // Calculate mfmas per rep.
    SmallVector<int64_t> ctaTile = {shapeC[0], shapeC[1], shapeA[1]};
    SmallVector<int64_t> warpTile = {
      shapeC[0] / warpsPerCTA[0],
      shapeC[1] / warpsPerCTA[1],
      shapeA[1],
      };
    auto mfmaVersion = mfmaLayout.getVersionMajor();
    bool allowXF32 =
        dotOp.getInputPrecision() == InputPrecision::TF32 && mfmaVersion == 3;
    auto maybeMfmaInsn = MfmaInsn::selectMfma(mDim, nDim, elemTyA, elemTyB,
                                              mfmaVersion, allowXF32);
    if (failed(maybeMfmaInsn))
      llvm::report_fatal_error("No match found in MFMA database\n");
    SmallVector<unsigned> mfmaShape = {
      maybeMfmaInsn->getMDim(),
      maybeMfmaInsn->getNDim(),
      maybeMfmaInsn->getKDim()};

    auto mfmasPerRep = calcMfmasPerRep(
        ctaTile,
        warpsPerCTA,
        numRepShape,
        mfmaShape);
    DBGS() << "mfmasPerRep: "
      << mfmasPerRep[0] << "x"
      << mfmasPerRep[1] << "x"
      << mfmasPerRep[2] << "\n";

    // Calculate Dot-Tiling.
    unsigned cyclesPerMfma = calcCyclesPerMfma(mfmaLayout, dotOp);
    // Prefer tile to be skinny along inner loop dimension to minimize registers.
    const bool preferOuterLoopM = (warpTile[0] >= warpTile[1]); // true: row-major when tall warp-tile
    const bool preferTileLargerM = !preferOuterLoopM; // true: tall tiles when wide warp-tile
    // Calculate dot-tile shape (in reps per dot-tile).
    DotTileShape tileShape = calcDotTileShape(mfmasPerRep, preferTileLargerM, cyclesPerMfma);
    DBGS() << "repsPerDotTile: " << tileShape[0] << "x" << tileShape[1]  << "x" << 1 << "\n";
    const int tileShapeM = tileShape[0];
    const int tileShapeN = tileShape[1];
    const DotTileOrder dotTileOrder(numRepM, numRepN, tileShapeM, tileShapeN, preferOuterLoopM);

    // Extract slices for A operands.
    int64_t elementsPerSliceM = refinedShapeCD[0];
    int64_t elementsPerSliceN = refinedShapeCD[1];
    int64_t elementsPerSliceK = refinedShapeA[1];
    auto extractSliceTypeA =
        RankedTensorType::get(refinedShapeA, elemTyA, encodeA);
    rewriter.setInsertionPointAfter(dotOp);
    SmallVector<SmallVector<amdgpu::ExtractSliceOp>> subtilesA;
    unsigned tileIdx = 0;
    for (int32_t k = 0; k < numRepK; ++k) {
      SmallVector<amdgpu::ExtractSliceOp> subtilesK;
      for (int32_t i = 0; i < numRepM; ++i) {
        DBGS() << "local_load_a[" << i << "][" << k << "]\n";
        int32_t shiftM = i * elementsPerSliceM;
        int32_t shiftK = k * elementsPerSliceK;
        auto extract = rewriter.create<amdgpu::ExtractSliceOp>(
          loc, Type{extractSliceTypeA}, Value{a},
          DenseI64ArrayAttr::get(ctx, {shiftM, shiftK})
          );
        // Add dot-tile info to local_load's slice;
        // this specifies which dot-tile this load is needed for.
        int32_t tileM = i / tileShapeM;
        int32_t tileN = -1;
        int32_t tileK = k;
        int32_t tileSerial = dotTileOrder.getOuterTileM() ? tileM * dotTileOrder.getNumTilesN() : tileM;
        tileSerial += k * dotTileOrder.getNumTilesM() * dotTileOrder.getNumTilesN();
        int32_t elementM = i % tileShapeM; // dots are n-major within tile
        int32_t elementN = -1;
        int32_t elementK = 0;
        int32_t elementSerial = elementM * tileShapeN; // dots are n-major within tile
        auto dotTileAttr = triton::amdgpu::DotTileAttr::get(ctx, tileM, tileN, tileK, tileSerial, elementM, elementN, elementK, elementSerial);
        extract->setAttr(triton::amdgpu::DotTileAttr::getMnemonic(), dotTileAttr);
        subtilesK.push_back(extract);
      }
      subtilesA.push_back(subtilesK);
    }

    // Extract slices for B operands.
    auto extractSliceTypeB =
        RankedTensorType::get(refinedShapeB, elemTyB, encodeB);
    SmallVector<SmallVector<amdgpu::ExtractSliceOp>> subtilesB;
    tileIdx = 0;
    for (int32_t k = 0; k < numRepK; ++k) {
      SmallVector<amdgpu::ExtractSliceOp> subtilesK;
      for (int32_t j = 0; j < numRepN; ++j) {
        DBGS() << "local_load_b[" << k << "][" << j << "]\n";
        int32_t shiftN = j * elementsPerSliceN;
        int32_t shiftK = k * elementsPerSliceK;
        auto extract = rewriter.create<amdgpu::ExtractSliceOp>(
          loc, Type{extractSliceTypeB}, Value{b},
          DenseI64ArrayAttr::get(ctx, {shiftK, shiftN}));
        // Add dot-tile info to local_load's slice;
        // this specifies which dot-tile this load is needed for.
        int32_t tileM = -1;
        int32_t tileN = j / tileShapeN;
        int32_t tileK = k;
        int32_t tileSerial = dotTileOrder.getOuterTileM() ? tileN : tileN * dotTileOrder.getNumTilesM();
        tileSerial += k * dotTileOrder.getNumTilesM() * dotTileOrder.getNumTilesN();
        int32_t elementM = -1;
        int32_t elementN = j % tileShapeN; // dots are n-major within tile
        int32_t elementK = 0;
        int32_t elementSerial = elementN; // dots are n-major within tile
        auto dotTileAttr = triton::amdgpu::DotTileAttr::get(ctx, tileM, tileN, tileK, tileSerial, elementM, elementN, elementK, elementSerial);
        extract->setAttr(triton::amdgpu::DotTileAttr::getMnemonic(), dotTileAttr);
        subtilesK.push_back(extract);
      }
      subtilesB.push_back(subtilesK);
    }

    auto refinedTensorTypeC =
        RankedTensorType::get(refinedShapeCD, elemTyC, encodeC);
    auto refinedTensorTypeD =
        RankedTensorType::get(refinedShapeCD, elemTyD, encodeD);
    SmallVector<Value> refinedDotValues;
    // Extract slices for refined C operands for first slice of K.
    // Create these in same order that concat wants them.
    for (int m = 0; m < numRepM; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        SmallVector<int64_t> offset = {m * elementsPerSliceM,
                                      n * elementsPerSliceN};
        auto refinedTensorC = rewriter.create<triton::amdgpu::ExtractSliceOp>(
            loc, Type{refinedTensorTypeC}, Value{c}, offset);
        refinedDotValues.push_back(refinedTensorC);
      }
    }
    auto dotAttrs = dotOp->getAttrs();
    int32_t tileSerial = 0;
    // Iterate over dot-tiles.
    for (int32_t k = 0; k < numRepK; ++k) {
      for (int tileOuterIdx = 0; tileOuterIdx < dotTileOrder.getNumTilesOuter(); ++tileOuterIdx) {
        for (int tileInnerIdx = 0; tileInnerIdx < dotTileOrder.getNumTilesInner(); ++tileInnerIdx) {
          const int tileStartM = dotTileOrder.getTileStartM(tileOuterIdx, tileInnerIdx);
          const int tileStartN = dotTileOrder.getTileStartN(tileOuterIdx, tileInnerIdx);
          int32_t elementSerial = 0;
          DBGS() << "dot-tile[" << tileSerial << "]\n";
          // Iterate over dots within dot-tile.
          for (int m = tileStartM; m < tileStartM + tileShapeM; ++m) {
            for (int n = tileStartN; n < tileStartN + tileShapeN; ++n) {
              DBGS() << "  dot[" << m << "][" << n << "][" << k << "]\n";
              auto refinedTensorA = subtilesA[k][m];
              auto refinedTensorB = subtilesB[k][n];
              auto dotOp = rewriter.create<tt::DotOp>(
                  loc, refinedTensorTypeD,
                  ValueRange{refinedTensorA, refinedTensorB,
                  refinedDotValues[int32_t(m*numRepN+n)]}, dotAttrs);
              // Add dot-tile info to dot.
              int32_t tileM = tileStartM / tileShapeM;
              int32_t tileN = tileStartN / tileShapeN;
              int32_t tileK = k;
              int32_t elementM = m - tileStartM;
              int32_t elementN = n - tileStartN;
              int32_t elementK = 0;
              auto dotTileAttr = triton::amdgpu::DotTileAttr::get(ctx, tileM, tileN, tileK, tileSerial, elementM, elementN, elementK, elementSerial);
              dotOp->setAttr(triton::amdgpu::DotTileAttr::getMnemonic(), dotTileAttr);
              refinedDotValues[int32_t(m*numRepN+n)] = dotOp;
              elementSerial++;
            }
          }
          tileSerial++;
        }
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
      block->walk([&](triton::gpu::LocalLoadOp localLoadOp) {
        OpBuilder rewriter(localLoadOp->getContext());
        if (localLoadOp->getNumOperands() == 1) {
          if (failed(rewriteLocalLoad(rewriter, localLoadOp))) {
            LDBG("failed to refine ttg.localLoad: " << *localLoadOp);
          }
        }
      });

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
