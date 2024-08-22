#include "TypeConverter.h"

#include "cpu/include/Analysis/TensorPtrShapeInfo.h"
#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTMEMORYOPS
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

template <typename OpT>
struct MemoryOpConversion : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OpConversionPattern<OpT>::getContext;
  using OpConversionPattern<OpT>::getTypeConverter;

  MemoryOpConversion(ModuleAxisInfoAnalysis &axisInfoAnalysis,
                     ModuleTensorPtrShapeInfoAnalysis &shapeInfoAnalysis,
                     TypeConverter &typeConverter, bool useScalarLoops,
                     MLIRContext *context)
      : OpConversionPattern<OpT>(typeConverter, context),
        axisAnalysis(axisInfoAnalysis), shapeAnalysis(shapeInfoAnalysis),
        genScalarLoops(useScalarLoops) {}

  Value extractScalarPointer(Location loc, Value ptrs,
                             ArrayRef<int64_t> indices,
                             ConversionPatternRewriter &rewriter) const {
    // If we build a vector of pointers and the extract a pointer from it, then
    // compiler doesn't always optimize it to a simple scalar pointer
    // computation. Here we try to follow a data flow of the tensor to rebuild a
    // scalar pointer for more efficient resulting code.
    if (canComputeScalarValue(ptrs))
      return computeScalarValue(ptrs, indices, rewriter);

    // Fall back to a scalar pointer extraction from the vector.
    Value ptr = rewriter.create<vector::ExtractOp>(
        loc, rewriter.getRemappedValue(ptrs), indices);
    auto ptrTy = dyn_cast<RankedTensorType>(ptrs.getType()).getElementType();
    ptr = rewriter.create<IntToPtrOp>(loc, ptrTy, ptr);
    return ptr;
  }

  bool canComputeScalarValue(Value vals) const {
    auto def = vals.getDefiningOp();
    if (!def)
      return false;

    if (isa<AddPtrOp, BroadcastOp, ExpandDimsOp, TransOp, arith::AddFOp,
            arith::AddIOp, arith::CmpFOp, arith::CmpIOp, arith::DivFOp,
            arith::DivSIOp, arith::MulIOp, arith::MulFOp, arith::RemFOp,
            arith::RemUIOp, arith::RemSIOp>(*def)) {
      for (auto op : def->getOperands()) {
        if (!canComputeScalarValue(op))
          return false;
      }
      return true;
    }

    if (isa<SplatOp, MakeRangeOp>(*def))
      return true;

    if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
      if (auto denseVal = dyn_cast<DenseElementsAttr>(cst.getValue())) {
        return denseVal.isSplat();
      }
      return false;
    }

    return false;
  }

  Value computeScalarValue(Value vals, ArrayRef<int64_t> indices,
                           ConversionPatternRewriter &rewriter) const {
    if (auto def = vals.getDefiningOp<SplatOp>()) {
      return def.getSrc();
    }

    if (auto def = vals.getDefiningOp<MakeRangeOp>()) {
      int32_t start = static_cast<int32_t>(def.getStart());
      assert(indices.size() == 1);
      Type elemTy = cast<RankedTensorType>(def.getType()).getElementType();
      return rewriter.create<arith::ConstantOp>(
          def.getLoc(), elemTy,
          rewriter.getIntegerAttr(elemTy, start + indices[0]));
    }

    if (auto def = vals.getDefiningOp<BroadcastOp>()) {
      // Find broadcasted dimensions and replace indices for those dimensions
      // with 0 (broadcasted dimension always has size 1).
      SmallVector<int64_t> newIndices;
      auto sourceTy = cast<RankedTensorType>(def.getSrc().getType());
      auto targetTy = cast<RankedTensorType>(def.getType());
      assert(sourceTy.getRank() == indices.size() && "Mismatched rank");
      for (int64_t i = 0; i < sourceTy.getRank(); ++i) {
        if (sourceTy.getShape()[i] != targetTy.getShape()[i])
          newIndices.push_back(0);
        else
          newIndices.push_back(indices[i]);
      }
      return computeScalarValue(def.getSrc(), newIndices, rewriter);
    }

    if (auto def = vals.getDefiningOp<ExpandDimsOp>()) {
      // Remove index at expanded dimension.
      SmallVector<int64_t> newIndices(indices);
      newIndices.erase(newIndices.begin() + def.getAxis());
      return computeScalarValue(def.getSrc(), newIndices, rewriter);
    }

    if (auto def = vals.getDefiningOp<arith::ConstantOp>()) {
      auto denseVal = cast<DenseElementsAttr>(def.getValue());
      assert(denseVal.isSplat());
      auto scalarAttr = denseVal.getSplatValue<TypedAttr>();
      Value res = rewriter.create<arith::ConstantOp>(
          def.getLoc(), scalarAttr.getType(), scalarAttr);
      return res;
    }

    if (auto def = vals.getDefiningOp<TransOp>()) {
      // Permute indices.
      SmallVector<int64_t> newIndices;
      auto order = def.getOrder();
      assert(indices.size() == order.size() && "Mismatched rank");
      for (auto idx : order)
        newIndices.push_back(indices[idx]);
      return computeScalarValue(def.getSrc(), newIndices, rewriter);
    }

    // Generic case where we copy defining op with scalar operands.
    auto def = vals.getDefiningOp();
    OperationState newState(def->getLoc(), def->getName());
    for (auto op : def->getOperands()) {
      newState.operands.push_back(computeScalarValue(op, indices, rewriter));
    }
    assert(def->getResults().size() == 1);
    newState.types.push_back(
        cast<ShapedType>(def->getResultTypes()[0]).getElementType());
    newState.attributes = def->getAttrs();
    return rewriter.create(newState)->getResult(0);
  }

  Value computeScalarValue(Value vals, ValueRange indices,
                           ConversionPatternRewriter &rewriter,
                           DenseMap<Value, Value> &valMap) const {
    if (valMap.count(vals))
      return valMap.at(vals);

    if (auto def = vals.getDefiningOp<SplatOp>()) {
      return def.getSrc();
    }

    if (auto def = vals.getDefiningOp<arith::ConstantOp>()) {
      auto denseVal = cast<DenseElementsAttr>(def.getValue());
      assert(denseVal.isSplat());
      auto scalarAttr = denseVal.getSplatValue<TypedAttr>();
      Value res = rewriter.create<arith::ConstantOp>(
          def.getLoc(), scalarAttr.getType(), scalarAttr);
      valMap[vals] = res;
      return res;
    }

    if (auto def = vals.getDefiningOp<MakeRangeOp>()) {
      assert(indices.size() == 1);
      int32_t start = static_cast<int32_t>(def.getStart());
      Type elemTy = cast<RankedTensorType>(def.getType()).getElementType();
      Value startVal = rewriter.create<arith::ConstantOp>(
          def.getLoc(), elemTy, rewriter.getIntegerAttr(elemTy, start));
      Value index = indices[0];
      if (!elemTy.isIndex())
        index =
            rewriter.create<arith::IndexCastUIOp>(def.getLoc(), elemTy, index);
      Value res =
          rewriter.create<arith::AddIOp>(def.getLoc(), elemTy, startVal, index);
      valMap[vals] = res;
      return res;
    }

    if (auto def = vals.getDefiningOp<BroadcastOp>()) {
      // Find broadcasted dimensions and replace indices for those dimensions
      // with 0 (broadcasted dimension has always size 1).
      SmallVector<Value> newIndices;
      auto sourceTy = cast<RankedTensorType>(def.getSrc().getType());
      auto targetTy = cast<RankedTensorType>(def.getType());
      assert(sourceTy.getRank() == indices.size() && "Mismatched rank");
      for (int64_t i = 0; i < sourceTy.getRank(); ++i) {
        if (sourceTy.getShape()[i] != targetTy.getShape()[i])
          newIndices.push_back(
              rewriter.create<arith::ConstantIndexOp>(def.getLoc(), 0));
        else
          newIndices.push_back(indices[i]);
      }
      // The original cache is only used for the original set of indices.
      DenseMap<Value, Value> tmpValMap;
      Value res =
          computeScalarValue(def.getSrc(), newIndices, rewriter, tmpValMap);
      valMap[vals] = res;
      return res;
    }

    if (auto def = vals.getDefiningOp<ExpandDimsOp>()) {
      // Remove index at expanded dimension.
      SmallVector<Value> newIndices = indices;
      newIndices.erase(newIndices.begin() + def.getAxis());
      // The original cache is only used for the original set of indices.
      DenseMap<Value, Value> tmpValMap;
      Value res =
          computeScalarValue(def.getSrc(), newIndices, rewriter, tmpValMap);
      valMap[vals] = res;
      return res;
    }

    if (auto def = vals.getDefiningOp<TransOp>()) {
      // Permute indices.
      SmallVector<Value> newIndices;
      auto order = def.getOrder();
      assert(indices.size() == order.size() && "Mismatched rank");
      for (auto idx : order)
        newIndices.push_back(indices[idx]);
      // The original cache is only used for the original set of indices.
      DenseMap<Value, Value> tmpValMap;
      Value res =
          computeScalarValue(def.getSrc(), newIndices, rewriter, tmpValMap);
      valMap[vals] = res;
      return res;
    }

    // Generic case where we copy defining op with scalar operands.
    auto def = vals.getDefiningOp();
    OperationState newState(def->getLoc(), def->getName());
    for (auto op : def->getOperands()) {
      newState.operands.push_back(
          computeScalarValue(op, indices, rewriter, valMap));
    }
    assert(def->getResults().size() == 1);
    newState.types.push_back(
        cast<ShapedType>(def->getResultTypes()[0]).getElementType());
    newState.attributes = def->getAttrs();
    Value res = rewriter.create(newState)->getResult(0);
    valMap[vals] = res;
    return res;
  }

  Value extractMemRef(Location loc, Value ptr,
                      ConversionPatternRewriter &rewriter) const {
    auto tensorTy = dyn_cast<RankedTensorType>(
        dyn_cast<PointerType>(ptr.getType()).getPointeeType());
    auto elemTy = tensorTy.getElementType();
    auto shapeInfo = shapeAnalysis.getPtrShapeInfo(ptr);
    Type memRefTy;
    if (shapeInfo && shapeInfo->getRank() > 0) {
      auto layout =
          StridedLayoutAttr::get(getContext(), 0, shapeInfo->getStrides());
      memRefTy = MemRefType::get(shapeInfo->getShape(), elemTy, layout);
    } else {
      SmallVector<int64_t> dynVals(tensorTy.getRank(), ShapedType::kDynamic);
      auto layout = StridedLayoutAttr::get(getContext(), 0, dynVals);
      memRefTy = MemRefType::get(dynVals, elemTy, layout);
    }
    return rewriter.create<ExtractMemRefOp>(loc, memRefTy, ptr);
  }

  Value convertOtherVal(triton::LoadOp loadOp,
                        ConversionPatternRewriter &rewriter) const {
    if (loadOp.getOther())
      return rewriter.getRemappedValue(loadOp.getOther());

    auto resTy =
        dyn_cast<VectorType>(getTypeConverter()->convertType(loadOp.getType()));
    return rewriter.create<arith::ConstantOp>(
        loadOp.getLoc(), resTy,
        SplatElementsAttr::get(resTy,
                               rewriter.getZeroAttr(resTy.getElementType())));
  }

  Value createAlloca(Location loc, MemRefType ty, Operation *before,
                     ConversionPatternRewriter &rewriter) const {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(before);
    return rewriter.create<memref::AllocaOp>(
        loc, ty, rewriter.getIntegerAttr(rewriter.getI64Type(), 64));
  }

  // If tensor is not null and its element cannot be recomputed in a scalar
  // loop, then store it to a temporary buffer.
  Value maybeStoreVecToTempBuf(Location loc, Value vals, Value zeroIdx,
                               Operation *allocaPoint,
                               ConversionPatternRewriter &rewriter) const {
    if (!vals || canComputeScalarValue(vals))
      return nullptr;

    auto vec = rewriter.getRemappedValue(vals);
    auto vecTy = cast<VectorType>(vec.getType());
    auto elemTy = vecTy.getElementType();
    // Memref of i1 assumes one element per byte when we load/store element,
    // but vector store (through transfer write) would write 1 bit per element.
    if (elemTy.isInteger(1)) {
      elemTy = rewriter.getI8Type();
      vec = rewriter.create<arith::ExtUIOp>(
          loc, VectorType::get(vecTy.getShape(), elemTy), vec);
    }
    auto memRefTy = MemRefType::get(vecTy.getShape(), elemTy);
    Value memRef = createAlloca(vals.getLoc(), memRefTy, allocaPoint, rewriter);
    SmallVector<Value> indices(vecTy.getRank(), zeroIdx);
    rewriter.create<vector::TransferWriteOp>(vals.getLoc(), vec, memRef,
                                             indices);
    return memRef;
  }

  // Load scalar element from a temporary buffer or recompute it if the
  // buffer doesn't exist.
  Value computeOrLoadScalarValue(Value vals, Value tmpVals, ValueRange indices,
                                 ConversionPatternRewriter &rewriter,
                                 DenseMap<Value, Value> &valMap) const {
    // Allow null value for easier handling of optional arguments.
    if (!vals)
      return nullptr;

    // Load value from a temp buffer if any.
    if (tmpVals) {
      Value val =
          rewriter.create<memref::LoadOp>(vals.getLoc(), tmpVals, indices);
      // If we load a pointer then additional cast is needed because tensor of
      // pointers is transformed into a vector of integers.
      auto elemTy = dyn_cast<RankedTensorType>(vals.getType()).getElementType();
      if (isa<PointerType>(elemTy))
        val = rewriter.create<IntToPtrOp>(vals.getLoc(), elemTy, val);
      // We need to transform loaded i8 back to i1.
      else if (elemTy.isInteger(1))
        val = rewriter.create<arith::TruncIOp>(val.getLoc(),
                                               rewriter.getI1Type(), val);
      return val;
    }

    return computeScalarValue(vals, indices, rewriter, valMap);
  }

  LogicalResult scalarizeWithLoop(triton::LoadOp loadOp,
                                  ConversionPatternRewriter &rewriter) const {
    auto loc = loadOp.getLoc();
    auto vecTy =
        dyn_cast<VectorType>(getTypeConverter()->convertType(loadOp.getType()));

    auto ptrs = loadOp.getPtr();
    auto mask = loadOp.getMask();
    auto other = loadOp.getOther();
    auto cache = loadOp.getCache();
    auto evict = loadOp.getEvict();
    auto isVolatile = loadOp.getIsVolatile();

    // Create some reused constants.
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // There is alloca_scope operation to control alloca scopes. But its usage
    // in combination with nested SCF and multi-dimensional vectors make it
    // impossible to lower scopes to LLVM using existing MLIR passes. For now,
    // simply allocate temp memory in the function's region.
    // TODO: Use alloc for big buffers and revisit alloca scoping.
    Operation *allocaPoint = loadOp;
    while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
      allocaPoint = allocaPoint->getParentOp();

    // Allocate temp buffer for the result. Write the other value there if
    // we cannot write it in a loop.
    auto resMemRefTy =
        MemRefType::get(vecTy.getShape(), vecTy.getElementType());
    Value resMemRef = createAlloca(loc, resMemRefTy, allocaPoint, rewriter);
    bool storeOtherInLoop = static_cast<bool>(mask);
    if (other && !canComputeScalarValue(other)) {
      SmallVector<Value> indices(vecTy.getRank(), zeroIdx);
      rewriter.create<vector::TransferWriteOp>(
          loc, rewriter.getRemappedValue(other), resMemRef, indices);
      storeOtherInLoop = false;
    }

    // Store a tensor of pointers and mask into a temp buf if we can't
    // compute them in a loop.
    Value tmpPtrs =
        maybeStoreVecToTempBuf(loc, ptrs, zeroIdx, allocaPoint, rewriter);
    Value tmpMask =
        maybeStoreVecToTempBuf(loc, mask, zeroIdx, allocaPoint, rewriter);

    // Create for-loops to iterate through all vector dimensions.
    SmallVector<scf::ForOp> forOps;
    SmallVector<Value> ivs;
    for (int64_t i = 0; i < vecTy.getRank(); ++i) {
      Value upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, vecTy.getShape()[i]);
      auto forOp =
          rewriter.create<scf::ForOp>(loc, zeroIdx, upperBound, oneIdx);
      forOps.push_back(forOp);
      ivs.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    }

    // Compute or load a scalar arguments.
    DenseMap<Value, Value> valMap;
    Value scalarPtr =
        computeOrLoadScalarValue(ptrs, tmpPtrs, ivs, rewriter, valMap);
    Value scalarMask =
        computeOrLoadScalarValue(mask, tmpMask, ivs, rewriter, valMap);
    Value scalarOther;
    if (storeOtherInLoop) {
      if (other) {
        scalarOther = computeScalarValue(other, ivs, rewriter, valMap);
      } else {
        scalarOther = rewriter.create<arith::ConstantOp>(
            loc, vecTy.getElementType(),
            rewriter.getZeroAttr(vecTy.getElementType()));
      }
    }

    if (!mask) {
      // Regular load case.
      Value val = rewriter.create<triton::LoadOp>(loc, scalarPtr, cache, evict,
                                                  isVolatile);
      rewriter.create<memref::StoreOp>(loc, val, resMemRef, ivs);
    } else {
      // Conditional load case
      rewriter.create<scf::IfOp>(
          loc, scalarMask,
          [&](OpBuilder &builder, Location loc) {
            Value val = builder.create<triton::LoadOp>(loc, scalarPtr, cache,
                                                       evict, isVolatile);
            builder.create<memref::StoreOp>(loc, val, resMemRef, ivs);
            builder.create<scf::YieldOp>(loc);
          },
          [&](OpBuilder &builder, Location loc) {
            if (storeOtherInLoop)
              builder.create<memref::StoreOp>(loc, scalarOther, resMemRef, ivs);
            builder.create<scf::YieldOp>(loc);
          });
    }

    // Load vector from the temp storage and return it from alloca scope.
    rewriter.setInsertionPointAfter(forOps.front());
    SmallVector<Value> indices(vecTy.getRank(), zeroIdx);
    Value res =
        rewriter.create<vector::TransferReadOp>(loc, vecTy, resMemRef, indices);

    rewriter.replaceOp(loadOp, res);
    return success();
  }

  LogicalResult scalarizeWithLoop(triton::StoreOp storeOp,
                                  ConversionPatternRewriter &rewriter) const {
    auto loc = storeOp.getLoc();
    auto vecTy = dyn_cast<VectorType>(
        getTypeConverter()->convertType(storeOp.getValue().getType()));

    auto ptrs = storeOp.getPtr();
    auto mask = storeOp.getMask();
    auto vals = storeOp.getValue();
    auto cache = storeOp.getCache();
    auto evict = storeOp.getEvict();

    // Create some reused constants.
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Alloca is inserted similar to the load case.
    Operation *allocaPoint = storeOp;
    while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
      allocaPoint = allocaPoint->getParentOp();

    // Store a tensor of pointers, mask, and values into a temp buf if we can't
    // compute them in a loop.
    Value tmpPtrs =
        maybeStoreVecToTempBuf(loc, ptrs, zeroIdx, allocaPoint, rewriter);
    Value tmpMask =
        maybeStoreVecToTempBuf(loc, mask, zeroIdx, allocaPoint, rewriter);
    Value tmpVals =
        maybeStoreVecToTempBuf(loc, vals, zeroIdx, allocaPoint, rewriter);

    // Create for-loops to iterate through all vector dimensions.
    SmallVector<scf::ForOp> forOps;
    SmallVector<Value> ivs;
    for (int64_t i = 0; i < vecTy.getRank(); ++i) {
      Value upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, vecTy.getShape()[i]);
      auto forOp =
          rewriter.create<scf::ForOp>(loc, zeroIdx, upperBound, oneIdx);
      forOps.push_back(forOp);
      ivs.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    }

    // Compute or load scalar args.
    DenseMap<Value, Value> valMap;
    Value scalarPtr =
        computeOrLoadScalarValue(ptrs, tmpPtrs, ivs, rewriter, valMap);
    Value scalarMask =
        computeOrLoadScalarValue(mask, tmpMask, ivs, rewriter, valMap);
    Value scalarVal =
        computeOrLoadScalarValue(vals, tmpVals, ivs, rewriter, valMap);

    if (!mask) {
      // Regular store case.
      rewriter.create<triton::StoreOp>(loc, scalarPtr, scalarVal, cache, evict);
    } else {
      // Conditional store case
      rewriter.create<scf::IfOp>(loc, scalarMask,
                                 [&](OpBuilder &builder, Location loc) {
                                   builder.create<triton::StoreOp>(
                                       loc, scalarPtr, scalarVal, cache, evict);
                                   builder.create<scf::YieldOp>(loc);
                                 });
    }

    rewriter.eraseOp(storeOp);
    return success();
  }

protected:
  ModuleAxisInfoAnalysis &axisAnalysis;
  ModuleTensorPtrShapeInfoAnalysis &shapeAnalysis;
  bool genScalarLoops;
};

struct LoadOpConversion : public MemoryOpConversion<triton::LoadOp> {
  using MemoryOpConversion::MemoryOpConversion;

  static Value
  getPaddingValue(Location loc, Type type,
                  const std::optional<triton::PaddingOption> &padding,
                  ConversionPatternRewriter &rewriter) {
    auto padding_option = padding.value_or(PaddingOption::PAD_ZERO);

    TypedAttr attr;
    switch (padding_option) {
    case PaddingOption::PAD_ZERO:
      attr = rewriter.getZeroAttr(type);
      break;
    case PaddingOption::PAD_NAN:
      assert(!type.isIntOrIndex());
      auto apNaN =
          llvm::APFloat::getNaN(cast<FloatType>(type).getFloatSemantics());
      attr = FloatAttr::get(type, apNaN);
      break;
    }

    return rewriter.create<arith::ConstantOp>(loc, attr);
  }

  LogicalResult
  matchAndRewrite(triton::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();
    auto mask = loadOp.getMask();
    auto ptr = loadOp.getPtr();
    auto boundaryChecks = loadOp.getBoundaryCheck();

    if (!triton::isTensorPointerType(ptr.getType())) {
      auto axisInfo = axisAnalysis.getAxisInfo(ptr);
      if (axisInfo) {
        return lowerUsingAxisInfo(axisInfo, loadOp, rewriter);
      }
      return lowerToScalarLoads(loadOp, rewriter);
    }

    // TODO: support masks.
    if (mask) {
      llvm_unreachable("unsupported load op");
    }

    auto memRef = extractMemRef(loc, ptr, rewriter);
    auto rank = dyn_cast<MemRefType>(memRef.getType()).getRank();
    auto resTy = dyn_cast<VectorType>(
        getTypeConverter()->convertType(loadOp.getResult().getType()));
    auto indices = rewriter.create<ExtractIndicesOp>(loc, ptr).getResults();
    SmallVector<bool, 4> inBounds(rank, true);
    for (auto dim : boundaryChecks) {
      inBounds[dim] = false;
    }
    Value padding = getPaddingValue(loc, resTy.getElementType(),
                                    loadOp.getPadding(), rewriter);
    auto vecRead = rewriter.create<vector::TransferReadOp>(
        loc, resTy, memRef, indices, padding, inBounds);
    rewriter.replaceOp(loadOp, vecRead);
    return success();
  }

  LogicalResult lowerUsingAxisInfo(AxisInfo *axisInfo, triton::LoadOp loadOp,
                                   ConversionPatternRewriter &rewriter) const {
    // This is an experimental code that covers only a simple case of axis info
    // usage to demostrate load by tensor of pointers transformation into vector
    // loads.
    // TODO: Support more cases.
    // TODO: Make separate pass to produce block pointer stores?
    auto loc = loadOp.getLoc();
    auto vecTy =
        dyn_cast<VectorType>(getTypeConverter()->convertType(loadOp.getType()));
    auto shape = vecTy.getShape();
    auto contiguity = axisInfo->getContiguity();
    if (shape.back() > 1 && shape.back() == contiguity.back()) {
      auto strides = computeStrides(shape);
      int64_t numElems = vecTy.getNumElements();
      Type subVecTy = VectorType::get(shape.back(), vecTy.getElementType());
      Type memRefTy = MemRefType::get(shape.back(), vecTy.getElementType());
      Value mask = loadOp.getMask()
                       ? rewriter.getRemappedValue(loadOp.getMask())
                       : nullptr;
      Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value defaultVal = convertOtherVal(loadOp, rewriter);
      Value res = defaultVal;
      for (int64_t idx = 0; idx < numElems; idx += shape.back()) {
        auto indices = delinearize(idx, strides);
        SmallVector<int64_t> subIndices(indices.begin(),
                                        indices.begin() + indices.size() - 1);
        auto ptr =
            extractScalarPointer(loc, loadOp.getPtr(), indices, rewriter);
        Value memRef =
            rewriter.create<triton::cpu::PtrToMemRefOp>(loc, memRefTy, ptr);
        Value vec;
        if (mask) {
          Value subMask = mask;
          Value passThru = defaultVal;
          if (shape.size() > 1) {
            subMask = rewriter.create<vector::ExtractOp>(loc, mask, subIndices);
            passThru =
                rewriter.create<vector::ExtractOp>(loc, defaultVal, subIndices);
          }
          vec = rewriter.create<vector::MaskedLoadOp>(
              loc, subVecTy, memRef, zeroIdx, subMask, passThru);
        } else {
          vec = rewriter.create<vector::LoadOp>(loc, subVecTy, memRef, zeroIdx);
        }

        if (shape.size() > 1) {
          res = rewriter.create<vector::InsertOp>(loc, vec, res, subIndices);
        } else {
          res = vec;
        }
      }

      rewriter.replaceOp(loadOp, res);
      return success();
    }

    return lowerToScalarLoads(loadOp, rewriter);
  }

  LogicalResult lowerToScalarLoads(triton::LoadOp loadOp,
                                   ConversionPatternRewriter &rewriter) const {
    // Scalar loads and boundary checks are not expected.
    assert(loadOp.getBoundaryCheck().empty());
    assert(isa<RankedTensorType>(loadOp.getType()));

    auto loc = loadOp.getLoc();
    auto vecTy =
        dyn_cast<VectorType>(getTypeConverter()->convertType(loadOp.getType()));

    // We want to avoid a code explosion when scalarize loads of big vectors,
    // so try to build a scalar loop.
    if (genScalarLoops && vecTy.getNumElements() >= 16 &&
        succeeded(scalarizeWithLoop(loadOp, rewriter)))
      return success();

    auto ptrs = rewriter.getRemappedValue(loadOp.getPtr());
    auto mask = loadOp.getMask() ? rewriter.getRemappedValue(loadOp.getMask())
                                 : nullptr;
    auto ptrTy =
        dyn_cast<RankedTensorType>(loadOp.getPtr().getType()).getElementType();
    auto cache = loadOp.getCache();
    auto evict = loadOp.getEvict();
    auto isVolatile = loadOp.getIsVolatile();

    auto loadOne = [=, &rewriter](ArrayRef<int64_t> indices, Value dst) {
      Value ptr = rewriter.create<vector::ExtractOp>(loc, ptrs, indices);
      ptr = rewriter.create<IntToPtrOp>(loc, ptrTy, ptr);
      Value val =
          rewriter.create<triton::LoadOp>(loc, ptr, cache, evict, isVolatile);
      return rewriter.create<vector::InsertOp>(loc, val, dst, indices);
    };

    Value dst = convertOtherVal(loadOp, rewriter);
    int64_t numElems = vecTy.getNumElements();
    auto strides = computeStrides(vecTy.getShape());
    for (auto idx = 0; idx < numElems; ++idx) {
      auto indices = delinearize(idx, strides);
      if (!mask) {
        dst = loadOne(indices, dst);
        continue;
      }
      // Create a conditional block for load if there is a mask.
      auto predicate = rewriter.create<vector::ExtractOp>(loc, mask, indices);
      auto ifOp = rewriter.create<scf::IfOp>(
          loc, predicate,
          [&](OpBuilder &builder, Location loc) {
            auto result = loadOne(indices, dst).getResult();
            rewriter.create<scf::YieldOp>(loc, result);
          },
          [&](OpBuilder &builder, Location loc) {
            rewriter.create<scf::YieldOp>(loc, dst);
          });
      dst = ifOp.getResult(0);
    }

    rewriter.replaceOp(loadOp, dst);

    return success();
  }
};

struct StoreOpConversion : public MemoryOpConversion<triton::StoreOp> {
  using MemoryOpConversion::MemoryOpConversion;

  LogicalResult
  matchAndRewrite(triton::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = storeOp.getLoc();
    auto mask = storeOp.getMask();
    auto ptr = storeOp.getPtr();
    auto boundaryChecks = storeOp.getBoundaryCheck();

    if (!triton::isTensorPointerType(ptr.getType())) {
      auto axisInfo = axisAnalysis.getAxisInfo(ptr);
      if (axisInfo) {
        return lowerUsingAxisInfo(axisInfo, storeOp, rewriter);
      }
      return lowerToScalarStores(storeOp, rewriter);
    }

    // TODO: support masks.
    if (mask) {
      llvm_unreachable("unsupported store op");
    }

    auto value = rewriter.getRemappedValue(storeOp.getValue());
    auto memRef = extractMemRef(loc, ptr, rewriter);
    auto rank = dyn_cast<MemRefType>(memRef.getType()).getRank();
    auto indices = rewriter.create<ExtractIndicesOp>(loc, ptr).getResults();
    SmallVector<bool, 4> inBounds(rank, true);
    for (auto dim : boundaryChecks) {
      inBounds[dim] = false;
    }
    auto vecWrite = rewriter.create<vector::TransferWriteOp>(loc, value, memRef,
                                                             indices, inBounds);
    rewriter.replaceOp(storeOp, vecWrite);
    return success();
  }

  LogicalResult lowerUsingAxisInfo(AxisInfo *axisInfo, triton::StoreOp storeOp,
                                   ConversionPatternRewriter &rewriter) const {
    // This is an experimental code that covers only a simple case of axis info
    // usage to demostrate load by tensor of pointers transformation into vector
    // loads.
    // TODO: Support more cases.
    // TODO: Make separate pass to produce block pointer stores instead?
    auto loc = storeOp.getLoc();
    auto vals = rewriter.getRemappedValue(storeOp.getValue());
    auto vecTy = dyn_cast<VectorType>(vals.getType());
    auto shape = vecTy.getShape();
    auto contiguity = axisInfo->getContiguity();
    if (shape.back() > 1 && shape.back() == contiguity.back()) {
      auto strides = computeStrides(shape);
      int64_t numElems = vecTy.getNumElements();
      Type memRefTy = MemRefType::get(shape.back(), vecTy.getElementType());
      Value mask = storeOp.getMask()
                       ? rewriter.getRemappedValue(storeOp.getMask())
                       : nullptr;
      Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto vals = rewriter.getRemappedValue(storeOp.getValue());
      for (int64_t idx = 0; idx < numElems; idx += shape.back()) {
        auto indices = delinearize(idx, strides);
        auto ptr =
            extractScalarPointer(loc, storeOp.getPtr(), indices, rewriter);
        Value memRef =
            rewriter.create<triton::cpu::PtrToMemRefOp>(loc, memRefTy, ptr);
        indices.pop_back();
        auto val = rewriter.create<vector::ExtractOp>(loc, vals, indices);

        if (mask) {
          Value subMask = mask;
          if (shape.size() > 1) {
            SmallVector<int64_t> subIndices = indices;
            subIndices.pop_back();
            subMask = rewriter.create<vector::ExtractOp>(loc, mask, indices);
          }
          rewriter.create<vector::MaskedStoreOp>(loc, memRef, zeroIdx, subMask,
                                                 val);
        } else {
          rewriter.create<vector::StoreOp>(loc, val, memRef, zeroIdx);
        }
      }

      rewriter.eraseOp(storeOp);
      return success();
    }

    return lowerToScalarStores(storeOp, rewriter);
  }

  LogicalResult lowerToScalarStores(triton::StoreOp storeOp,
                                    ConversionPatternRewriter &rewriter) const {
    // Scalar stores and boundary checks are not expected.
    assert(storeOp.getBoundaryCheck().empty());
    assert(isa<RankedTensorType>(storeOp.getValue().getType()));

    auto loc = storeOp.getLoc();
    auto tensorTy = dyn_cast<RankedTensorType>(storeOp.getPtr().getType());

    // We want to avoid a code explosion when scalarize stores of big vectors,
    // so try to build a scalar loop.
    if (genScalarLoops && tensorTy.getNumElements() >= 16 &&
        succeeded(scalarizeWithLoop(storeOp, rewriter)))
      return success();

    auto ptrs = rewriter.getRemappedValue(storeOp.getPtr());
    auto mask = storeOp.getMask() ? rewriter.getRemappedValue(storeOp.getMask())
                                  : nullptr;
    auto vals = rewriter.getRemappedValue(storeOp.getValue());
    auto ptrTy = tensorTy.getElementType();
    auto cache = storeOp.getCache();
    auto evict = storeOp.getEvict();

    auto storeOne = [=, &rewriter](ArrayRef<int64_t> indices) {
      Value ptr = rewriter.create<vector::ExtractOp>(loc, ptrs, indices);
      ptr = rewriter.create<IntToPtrOp>(loc, ptrTy, ptr);
      Value val = rewriter.create<vector::ExtractOp>(loc, vals, indices);
      rewriter.create<triton::StoreOp>(loc, ptr, val, cache, evict);
    };

    int64_t numElems = tensorTy.getNumElements();
    auto strides = computeStrides(tensorTy.getShape());
    for (auto idx = 0; idx < numElems; ++idx) {
      auto indices = delinearize(idx, strides);
      if (!mask) {
        storeOne(indices);
        continue;
      }
      // Create a conditional block for store if there is a mask.
      auto predicate = rewriter.create<vector::ExtractOp>(loc, mask, indices);
      rewriter.create<scf::IfOp>(loc, predicate,
                                 [&](OpBuilder &builder, Location loc) {
                                   storeOne(indices);
                                   rewriter.create<scf::YieldOp>(loc);
                                 });
    }

    rewriter.eraseOp(storeOp);

    return success();
  }
};

class MemoryOpConversionTarget : public ConversionTarget {
public:
  explicit MemoryOpConversionTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<scf::SCFDialect>();
    addLegalDialect<memref::MemRefDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    // Allow only scalar loads and stores.
    addDynamicallyLegalOp<triton::LoadOp>([](triton::LoadOp loadOp) {
      return loadOp.getType().isIntOrIndexOrFloat();
    });
    addDynamicallyLegalOp<triton::StoreOp>([](triton::StoreOp storeOp) {
      return storeOp.getValue().getType().isIntOrIndexOrFloat();
    });
  }
};

struct ConvertMemoryOps
    : public triton::cpu::impl::ConvertMemoryOpsBase<ConvertMemoryOps> {
  ConvertMemoryOps() = default;

  ConvertMemoryOps(bool useScalarLoops) {
    this->useScalarLoops = useScalarLoops;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    ModuleTensorPtrShapeInfoAnalysis shapeInfoAnalysis(mod);
    MemoryOpConversionTarget convTarget(*context);
    TritonToTritonCPUTypeConverter pointerConverter;
    RewritePatternSet patterns(context);
    patterns.add<LoadOpConversion>(axisInfoAnalysis, shapeInfoAnalysis,
                                   pointerConverter, useScalarLoops, context);
    patterns.add<StoreOpConversion>(axisInfoAnalysis, shapeInfoAnalysis,
                                    pointerConverter, useScalarLoops, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertMemoryOps() {
  return std::make_unique<ConvertMemoryOps>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertMemoryOps(bool useScalarLoops) {
  return std::make_unique<ConvertMemoryOps>(useScalarLoops);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
