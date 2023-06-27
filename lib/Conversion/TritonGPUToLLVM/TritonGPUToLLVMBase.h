#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_BASE_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_BASE_H

// TODO: refactor so that it doesn't fail if Allocation.h
// is included after utility.h (due to conflict in `store` macro
// and <atomic>
#include "triton/Analysis/Allocation.h"

#include "TypeConverter.h"
//
#include "Utility.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Analysis/AxisInfo.h"
#include <set>
using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::SharedMemoryObject;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

namespace mlir {
namespace LLVM {

// Helper function for using printf in LLVM conversion.
void vprintf(StringRef msg, ValueRange args,
             ConversionPatternRewriter &rewriter);

void vprintf_array(Value thread, ArrayRef<Value> arr, std::string info,
                   std::string elem_repr, ConversionPatternRewriter &builder);

} // namespace LLVM
} // namespace mlir

// FuncOpConversion/FuncOpConversionBase is borrowed from
// https://github.com/llvm/llvm-project/blob/fae656b2dd80246c3c6f01e9c77c49560368752c/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp#L276
// since it is not exposed on header files in mlir v14
// TODO(Superjomn): remove the code when MLIR v15.0 is included.
// All the rights are reserved by the LLVM community.

struct FuncOpConversionBase : public ConvertOpToLLVMPattern<triton::FuncOp> {
protected:
  /// Only retain those attributes that are not constructed by
  /// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
  /// attributes.
  static void filterFuncAttributes(triton::FuncOp op, bool filterArgAttrs,
                                   SmallVectorImpl<NamedAttribute> &result) {

    for (const auto &attr : op->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == op.getFunctionTypeAttrName() ||
          attr.getName() == "std.varargs" ||
          (filterArgAttrs && attr.getName() == op.getArgAttrsAttrName()))
        continue;
      result.push_back(attr);
    }
  }

  /// Helper function for wrapping all attributes into a single DictionaryAttr
  static auto wrapAsStructAttrs(OpBuilder &b, ArrayAttr attrs) {
    return DictionaryAttr::get(b.getContext(),
                               b.getNamedAttr("llvm.struct_attrs", attrs));
  }

protected:
  using ConvertOpToLLVMPattern<triton::FuncOp>::ConvertOpToLLVMPattern;

  // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter provided
  // to this legalization pattern.
  LLVM::LLVMFuncOp
  convertFuncOpToLLVMFuncOp(triton::FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) const {
    // Convert the original function arguments. They are converted using the
    // LLVMTypeConverter provided to this legalization pattern.
    auto varargsAttr = funcOp->getAttrOfType<BoolAttr>("func.varargs");
    TypeConverter::SignatureConversion result(funcOp.getNumArguments());
    auto llvmType = getTypeConverter()->convertFunctionSignature(
        funcOp.getFunctionType(), varargsAttr && varargsAttr.getValue(), false,
        result);
    if (!llvmType)
      return nullptr;

    // Propagate argument/result attributes to all converted arguments/result
    // obtained after converting a given original argument/result.
    SmallVector<NamedAttribute, 4> attributes;
    filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, attributes);
    if (ArrayAttr resAttrDicts = funcOp.getAllResultAttrs()) {
      assert(!resAttrDicts.empty() && "expected array to be non-empty");
      auto newResAttrDicts =
          (funcOp.getNumResults() == 1)
              ? resAttrDicts
              : rewriter.getArrayAttr(
                    {wrapAsStructAttrs(rewriter, resAttrDicts)});
      attributes.push_back(
          rewriter.getNamedAttr(funcOp.getResAttrsAttrName(), newResAttrDicts));
    }
    if (ArrayAttr argAttrDicts = funcOp.getAllArgAttrs()) {
      SmallVector<Attribute, 4> newArgAttrs(
          llvmType.cast<LLVM::LLVMFunctionType>().getNumParams());
      for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
        auto mapping = result.getInputMapping(i);
        assert(mapping && "unexpected deletion of function argument");
        for (size_t j = 0; j < mapping->size; ++j)
          newArgAttrs[mapping->inputNo + j] = argAttrDicts[i];
      }
      attributes.push_back(rewriter.getNamedAttr(
          funcOp.getArgAttrsAttrName(), rewriter.getArrayAttr(newArgAttrs)));
    }
    for (const auto &pair : llvm::enumerate(attributes)) {
      if (pair.value().getName() == "llvm.linkage") {
        attributes.erase(attributes.begin() + pair.index());
        break;
      }
    }

    // Create an LLVM function, use external linkage by default until MLIR
    // functions have linkage.
    LLVM::Linkage linkage = LLVM::Linkage::External;
    if (auto linkageAttr = funcOp->getDiscardableAttr("llvm.linkage")) {
      auto attr = linkageAttr.dyn_cast<mlir::LLVM::LinkageAttr>();
      if (!attr) {
        funcOp->emitError()
            << "Contains llvm.linkage attribute not of type LLVM::LinkageAttr";
        return nullptr;
      }
      linkage = attr.getLinkage();
    }
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getName(), llvmType, linkage,
        /*dsoLocal*/ false, LLVM::CConv::C, /*comdat=*/nullptr, attributes);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &result)))
      return nullptr;

    return newFuncOp;
  }
};

using IndexCacheKeyT = std::pair<Attribute, RankedTensorType>;

struct CacheKeyDenseMapInfo {
  static IndexCacheKeyT getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return std::make_pair(
        mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer)),
        RankedTensorType{});
  }
  static IndexCacheKeyT getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    auto tombstone = llvm::DenseMapInfo<RankedTensorType>::getTombstoneKey();
    return std::make_pair(
        mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer)),
        tombstone);
  }
  static unsigned getHashValue(IndexCacheKeyT key) {
    auto shape = key.second.getShape();
    return llvm::hash_combine(mlir::hash_value(key.first),
                              mlir::hash_value(key.second));
  }
  static bool isEqual(IndexCacheKeyT LHS, IndexCacheKeyT RHS) {
    return LHS == RHS;
  }
};

class ConvertTritonGPUOpToLLVMPatternBase {
public:
  // Two levels of value cache in emitting indices calculation:
  // Key: pair<layout, shape>
  struct IndexCacheInfo {
    DenseMap<IndexCacheKeyT, SmallVector<Value>, CacheKeyDenseMapInfo>
        *baseIndexCache;
    DenseMap<IndexCacheKeyT, SmallVector<SmallVector<Value>>,
             CacheKeyDenseMapInfo> *indexCache;
    OpBuilder::InsertPoint *indexInsertPoint;
  };

  explicit ConvertTritonGPUOpToLLVMPatternBase(
      TritonGPUToLLVMTypeConverter &typeConverter)
      : converter(&typeConverter) {}

  explicit ConvertTritonGPUOpToLLVMPatternBase(
      TritonGPUToLLVMTypeConverter &typeConverter,
      IndexCacheInfo indexCacheInfo)
      : converter(&typeConverter), indexCacheInfo(indexCacheInfo) {}

  explicit ConvertTritonGPUOpToLLVMPatternBase(
      TritonGPUToLLVMTypeConverter &typeConverter, ModuleAllocation &allocation)
      : converter(&typeConverter), allocation(&allocation) {}

  explicit ConvertTritonGPUOpToLLVMPatternBase(
      TritonGPUToLLVMTypeConverter &typeConverter, ModuleAllocation &allocation,
      IndexCacheInfo indexCacheInfo)
      : converter(&typeConverter), allocation(&allocation),
        indexCacheInfo(indexCacheInfo) {}

  TritonGPUToLLVMTypeConverter *getTypeConverter() const { return converter; }

  static Value
  getStructFromSharedMemoryObject(Location loc,
                                  const SharedMemoryObject &smemObj,
                                  ConversionPatternRewriter &rewriter) {
    auto elems = smemObj.getElems();
    auto types = smemObj.getTypes();
    auto structTy =
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);
    // pack into struct
    Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structTy);
    for (const auto &v : llvm::enumerate(elems)) {
      assert(v.value() && "can not insert null values");
      llvmStruct = insert_val(structTy, llvmStruct, v.value(), v.index());
    }
    return llvmStruct;
  }

  Value getThreadId(ConversionPatternRewriter &rewriter, Location loc) const {
    auto tid = rewriter.create<::mlir::gpu::ThreadIdOp>(
        loc, ::mlir::gpu::Dimension::x);
    return rewriter.create<arith::IndexCastOp>(loc, i32_ty, tid);
  }

  // -----------------------------------------------------------------------
  // Shared memory utilities
  // -----------------------------------------------------------------------
  template <typename T>
  Value getSharedMemoryBase(Location loc, ConversionPatternRewriter &rewriter,
                            T value) const {
    auto ptrTy = LLVM::LLVMPointerType::get(
        this->getTypeConverter()->convertType(rewriter.getI8Type()), 3);
    FunctionOpInterface funcOp;
    if constexpr (std::is_pointer_v<T>)
      funcOp = value->template getParentOfType<FunctionOpInterface>();
    else
      funcOp = value.getParentRegion()
                   ->template getParentOfType<FunctionOpInterface>();
    auto *funcAllocation = allocation->getFuncData(funcOp);
    auto smem = allocation->getFunctionSharedMemoryBase(funcOp);
    auto bufferId = funcAllocation->getBufferId(value);
    assert(bufferId != Allocation::InvalidBufferId && "BufferId not found");
    size_t offset = funcAllocation->getOffset(bufferId);
    Value offVal = i32_val(offset);
    Value base = gep(ptrTy, smem, offVal);
    return base;
  }

  DenseMap<unsigned, Value>
  getSwizzledSharedPtrs(Location loc, unsigned inVec, RankedTensorType srcTy,
                        triton::gpu::SharedEncodingAttr resSharedLayout,
                        Type resElemTy, SharedMemoryObject smemObj,
                        ConversionPatternRewriter &rewriter,
                        SmallVectorImpl<Value> &offsetVals,
                        SmallVectorImpl<Value> &srcStrides) const {
    // This utililty computes the pointers for accessing the provided swizzled
    // shared memory layout `resSharedLayout`. More specifically, it computes,
    // for all indices (row, col) of `srcEncoding` such that idx % inVec = 0,
    // the pointer: ptr[(row, col)] = base + (rowOff * strides[ord[1]] +
    // colOff) where :
    //   compute phase = (row // perPhase) % maxPhase
    //   rowOff = row
    //   colOff = colOffSwizzled + colOffOrdered
    //     colOffSwizzled = ((col // outVec) ^ phase) * outVec
    //     colOffOrdered = (col % outVec) // minVec * minVec
    //
    // Note 1:
    // -------
    // Because swizzling happens at a granularity of outVec, we need to
    // decompose the offset into a swizzled factor and a non-swizzled
    // (ordered) factor
    //
    // Note 2:
    // -------
    // If we have x, y, z of the form:
    // x = 0b00000xxxx
    // y = 0byyyyy0000
    // z = 0b00000zzzz
    // then (x + y) XOR z = 0byyyyxxxx XOR 0b00000zzzz = (x XOR z) + y
    // This means that we can use some immediate offsets for shared memory
    // operations.
    auto dstPtrTy = ptr_ty(resElemTy, 3);
    auto dstOffset = dot(rewriter, loc, offsetVals, smemObj.strides);
    Value dstPtrBase = gep(dstPtrTy, smemObj.base, dstOffset);

    auto srcEncoding = srcTy.getEncoding();
    auto srcShape = srcTy.getShape();
    unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
    // swizzling params as described in TritonGPUAttrDefs.td
    unsigned outVec = resSharedLayout.getVec();
    unsigned perPhase = resSharedLayout.getPerPhase();
    unsigned maxPhase = resSharedLayout.getMaxPhase();
    // order
    auto inOrder = triton::gpu::getOrder(srcEncoding);
    auto outOrder = triton::gpu::getOrder(resSharedLayout);
    // tensor indices held by the current thread, as LLVM values
    auto srcIndices = emitIndices(loc, rewriter, srcEncoding, srcTy);
    // return values
    DenseMap<unsigned, Value> ret;
    // cache for non-immediate offsets
    DenseMap<unsigned, Value> cacheCol, cacheRow;
    unsigned minVec = std::min(outVec, inVec);
    for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
      // extract multi dimensional index for current element
      auto idx = srcIndices[elemIdx];
      Value idxCol = idx[outOrder[0]]; // contiguous dimension
      Value idxRow = idx[outOrder[1]]; // discontiguous dimension
      Value strideCol = srcStrides[outOrder[0]];
      Value strideRow = srcStrides[outOrder[1]];
      // extract dynamic/static offset for immediate offsetting
      unsigned immedateOffCol = 0;
      if (auto add = dyn_cast_or_null<LLVM::AddOp>(idxCol.getDefiningOp()))
        if (auto _cst = dyn_cast_or_null<LLVM::ConstantOp>(
                add.getRhs().getDefiningOp())) {
          unsigned cst =
              _cst.getValue().cast<IntegerAttr>().getValue().getSExtValue();
          unsigned key = cst % (outVec * maxPhase);
          cacheCol.insert({key, idxCol});
          idxCol = cacheCol[key];
          immedateOffCol = cst / (outVec * maxPhase) * (outVec * maxPhase);
        }
      // extract dynamic/static offset for immediate offsetting
      unsigned immedateOffRow = 0;
      if (auto add = dyn_cast_or_null<LLVM::AddOp>(idxRow.getDefiningOp()))
        if (auto _cst = dyn_cast_or_null<LLVM::ConstantOp>(
                add.getRhs().getDefiningOp())) {
          unsigned cst =
              _cst.getValue().cast<IntegerAttr>().getValue().getSExtValue();
          unsigned key = cst % (perPhase * maxPhase);
          cacheRow.insert({key, idxRow});
          idxRow = cacheRow[key];
          immedateOffRow = cst / (perPhase * maxPhase) * (perPhase * maxPhase);
        }
      // compute phase = (row // perPhase) % maxPhase
      Value phase = urem(udiv(idxRow, i32_val(perPhase)), i32_val(maxPhase));
      // row offset is simply row index
      Value rowOff = mul(idxRow, strideRow);
      // because swizzling happens at a granularity of outVec, we need to
      // decompose the offset into a swizzled factor and a non-swizzled
      // (ordered) factor: colOffSwizzled = ((col // outVec) ^ phase) * outVec
      // colOffOrdered = (col % outVec) // minVec * minVec
      Value colOffSwizzled = xor_(udiv(idxCol, i32_val(outVec)), phase);
      colOffSwizzled = mul(colOffSwizzled, i32_val(outVec));
      Value colOffOrdered = urem(idxCol, i32_val(outVec));
      colOffOrdered = udiv(colOffOrdered, i32_val(minVec));
      colOffOrdered = mul(colOffOrdered, i32_val(minVec));
      Value colOff = add(colOffSwizzled, colOffOrdered);
      // compute non-immediate offset
      Value offset = add(rowOff, mul(colOff, strideCol));
      Value currPtr = gep(dstPtrTy, dstPtrBase, offset);
      // compute immediate offset
      Value immedateOff =
          add(mul(i32_val(immedateOffRow), srcStrides[outOrder[1]]),
              i32_val(immedateOffCol));
      ret[elemIdx] = gep(dstPtrTy, currPtr, immedateOff);
    }
    return ret;
  }

  SmallVector<Value>
  loadSharedToDistributed(Value dst, ArrayRef<SmallVector<Value>> dstIndices,
                          Value src, SharedMemoryObject smemObj, Type elemTy,
                          Location loc,
                          ConversionPatternRewriter &rewriter) const {
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto dstShape = dstTy.getShape();
    assert(dstShape.size() == 2 &&
           "Unexpected rank of loadSharedToDistributed");
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto dstDistributedLayout = dstTy.getEncoding();
    if (auto mmaLayout = dstDistributedLayout.dyn_cast<MmaEncodingAttr>()) {
      assert((!mmaLayout.isVolta()) &&
             "ConvertLayout Shared->MMAv1 is not supported yet");
    }
    auto srcSharedLayout =
        srcTy.getEncoding().cast<triton::gpu::SharedEncodingAttr>();
    auto srcElemTy = srcTy.getElementType();
    auto dstElemTy = dstTy.getElementType();
    auto inOrd = triton::gpu::getOrder(srcSharedLayout);
    auto outOrd = triton::gpu::getOrder(dstDistributedLayout);
    unsigned outVec =
        inOrd == outOrd
            ? triton::gpu::getContigPerThread(dstDistributedLayout)[outOrd[0]]
            : 1;
    unsigned inVec = srcSharedLayout.getVec();
    unsigned minVec = std::min(outVec, inVec);
    unsigned outElems = triton::gpu::getTotalElemsPerThread(dstTy);
    assert(outElems == dstIndices.size());

    DenseMap<unsigned, Value> sharedPtrs = getSwizzledSharedPtrs(
        loc, outVec, dstTy, srcSharedLayout, srcElemTy, smemObj, rewriter,
        smemObj.offsets, smemObj.strides);
    assert(outElems % minVec == 0 && "Unexpected number of elements");
    unsigned numVecs = outElems / minVec;
    auto wordTy = vec_ty(elemTy, minVec);
    SmallVector<Value> outVals(outElems);
    for (unsigned i = 0; i < numVecs; ++i) {
      Value smemAddr = sharedPtrs[i * minVec];
      smemAddr = bitcast(smemAddr, ptr_ty(wordTy, 3));
      Value valVec = load(smemAddr);
      for (unsigned v = 0; v < minVec; ++v) {
        Value currVal = extract_element(dstElemTy, valVec, i32_val(v));
        outVals[i * minVec + v] = currVal;
      }
    }
    return outVals;
  }

  void storeDistributedToShared(Value src, Value llSrc,
                                ArrayRef<Value> dstStrides,
                                ArrayRef<SmallVector<Value>> srcIndices,
                                Value dst, Value smemBase, Type elemTy,
                                Location loc,
                                ConversionPatternRewriter &rewriter) const {
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto srcShape = srcTy.getShape();
    assert(srcShape.size() == 2 &&
           "Unexpected rank of storeDistributedToShared");
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto srcDistributedLayout = srcTy.getEncoding();
    if (auto mmaLayout = srcDistributedLayout.dyn_cast<MmaEncodingAttr>()) {
      assert((!mmaLayout.isVolta()) &&
             "ConvertLayout MMAv1->Shared is not supported yet");
    }
    auto dstSharedLayout =
        dstTy.getEncoding().cast<triton::gpu::SharedEncodingAttr>();
    auto dstElemTy = dstTy.getElementType();
    auto inOrd = triton::gpu::getOrder(srcDistributedLayout);
    auto outOrd = dstSharedLayout.getOrder();
    unsigned inVec =
        inOrd == outOrd
            ? triton::gpu::getContigPerThread(srcDistributedLayout)[inOrd[0]]
            : 1;
    unsigned outVec = dstSharedLayout.getVec();
    unsigned minVec = std::min(outVec, inVec);
    unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
    assert(numElems == srcIndices.size());
    auto inVals =
        getTypeConverter()->unpackLLElements(loc, llSrc, rewriter, srcTy);
    auto wordTy = vec_ty(elemTy, minVec);
    Value word;

    SmallVector<Value> srcStrides = {dstStrides[0], dstStrides[1]};
    SmallVector<Value> offsetVals = {i32_val(0), i32_val(0)};
    SharedMemoryObject smemObj(smemBase, srcStrides, offsetVals);

    DenseMap<unsigned, Value> sharedPtrs =
        getSwizzledSharedPtrs(loc, inVec, srcTy, dstSharedLayout, dstElemTy,
                              smemObj, rewriter, offsetVals, srcStrides);

    for (unsigned i = 0; i < numElems; ++i) {
      if (i % minVec == 0)
        word = undef(wordTy);
      word = insert_element(wordTy, word, inVals[i], i32_val(i % minVec));
      if (i % minVec == minVec - 1) {
        Value smemAddr = sharedPtrs[i / minVec * minVec];
        smemAddr = bitcast(smemAddr, ptr_ty(wordTy, 3));
        store(word, smemAddr);
      }
    }
  }

  // -----------------------------------------------------------------------
  // Utilities
  // -----------------------------------------------------------------------
  Value getMask(Type valueTy, ConversionPatternRewriter &rewriter,
                Location loc) const {
    auto tensorTy = valueTy.dyn_cast<RankedTensorType>();
    Value mask = int_val(1, 1);
    auto tid = tid_val();
    if (tensorTy) {
      auto layout = tensorTy.getEncoding();
      auto shape = tensorTy.getShape();
      unsigned rank = shape.size();
      auto sizePerThread = triton::gpu::getSizePerThread(layout);
      auto threadsPerWarp = triton::gpu::getThreadsPerWarp(layout);
      auto warpsPerCTA = triton::gpu::getWarpsPerCTA(layout);
      auto order = triton::gpu::getOrder(layout);
      auto shapePerCTA = triton::gpu::getShapePerCTA(layout, shape);
      Value warpSize = i32_val(32);
      Value laneId = urem(tid, warpSize);
      Value warpId = udiv(tid, warpSize);
      SmallVector<Value> multiDimWarpId =
          delinearize(rewriter, loc, warpId, warpsPerCTA, order);
      SmallVector<Value> multiDimThreadId =
          delinearize(rewriter, loc, laneId, threadsPerWarp, order);
      for (unsigned dim = 0; dim < rank; ++dim) {
        // if there is no data replication across threads on this dimension
        if (shape[dim] >= shapePerCTA[dim])
          continue;
        // Otherwise, we need to mask threads that will replicate data on this
        // dimension. Calculate the thread index on this dimension for the CTA
        Value threadDim =
            add(mul(multiDimWarpId[dim], i32_val(threadsPerWarp[dim])),
                multiDimThreadId[dim]);
        mask = and_(mask, icmp_slt(mul(threadDim, i32_val(sizePerThread[dim])),
                                   i32_val(shape[dim])));
      }
    } else {
      // If the tensor is not ranked, then it is a scalar and only thread 0 can
      // write
      mask = and_(mask, icmp_eq(tid, i32_val(0)));
    }
    return mask;
  }

  Value dot(ConversionPatternRewriter &rewriter, Location loc,
            ArrayRef<Value> offsets, ArrayRef<Value> strides) const {
    assert(offsets.size() == strides.size());
    Value ret = i32_val(0);
    for (auto [offset, stride] : llvm::zip(offsets, strides)) {
      ret = add(ret, mul(offset, stride));
    }
    return ret;
  }

  struct SmallVectorKeyInfo {
    static unsigned getHashValue(const SmallVector<unsigned> &key) {
      return llvm::hash_combine_range(key.begin(), key.end());
    }
    static bool isEqual(const SmallVector<unsigned> &lhs,
                        const SmallVector<unsigned> &rhs) {
      return lhs == rhs;
    }
    static SmallVector<unsigned> getEmptyKey() {
      return SmallVector<unsigned>();
    }
    static SmallVector<unsigned> getTombstoneKey() {
      return {std::numeric_limits<unsigned>::max()};
    }
  };

  // -----------------------------------------------------------------------
  // Get offsets / indices for any layout
  // -----------------------------------------------------------------------

  SmallVector<Value> emitBaseIndexForLayout(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            Attribute layout,
                                            RankedTensorType type) const {
    IndexCacheKeyT key = std::make_pair(layout, type);
    auto cache = indexCacheInfo.baseIndexCache;
    auto insertPt = indexCacheInfo.indexInsertPoint;
    if (cache && cache->count(key) > 0) {
      return cache->lookup(key);
    } else {
      ConversionPatternRewriter::InsertionGuard guard(rewriter);
      if (cache)
        restoreInsertionPointIfSet(insertPt, rewriter);
      SmallVector<Value> result;
      if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
        result =
            emitBaseIndexForBlockedLayout(loc, rewriter, blockedLayout, type);
      } else if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
        if (mmaLayout.isVolta())
          result = emitBaseIndexForMmaLayoutV1(loc, rewriter, mmaLayout, type);
        if (mmaLayout.isAmpere())
          result = emitBaseIndexForMmaLayoutV2(loc, rewriter, mmaLayout, type);
      } else if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
        auto parentLayout = sliceLayout.getParent();
        auto parentShape = sliceLayout.paddedShape(type.getShape());
        RankedTensorType parentTy = RankedTensorType::get(
            parentShape, type.getElementType(), parentLayout);
        result = emitBaseIndexForLayout(loc, rewriter, parentLayout, parentTy);
        result.erase(result.begin() + sliceLayout.getDim());
      } else {
        llvm_unreachable("unsupported emitBaseIndexForLayout");
      }
      if (cache) {
        cache->insert(std::make_pair(key, result));
        *insertPt = rewriter.saveInsertionPoint();
      }
      return result;
    }
  }

  SmallVector<SmallVector<unsigned>>
  emitOffsetForLayout(Attribute layout, RankedTensorType type) const {
    if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>())
      return emitOffsetForBlockedLayout(blockedLayout, type);
    if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
      if (mmaLayout.isVolta())
        return emitOffsetForMmaLayoutV1(mmaLayout, type);
      if (mmaLayout.isAmpere())
        return emitOffsetForMmaLayoutV2(mmaLayout, type);
    }
    if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>())
      return emitOffsetForSliceLayout(sliceLayout, type);
    llvm_unreachable("unsupported emitOffsetForLayout");
  }

  // -----------------------------------------------------------------------
  // Emit indices
  // -----------------------------------------------------------------------
  SmallVector<SmallVector<Value>> emitIndices(Location loc,
                                              ConversionPatternRewriter &b,
                                              Attribute layout,
                                              RankedTensorType type) const {
    IndexCacheKeyT key(layout, type);
    auto cache = indexCacheInfo.indexCache;
    auto insertPt = indexCacheInfo.indexInsertPoint;
    if (cache && cache->count(key) > 0) {
      return cache->lookup(key);
    } else {
      ConversionPatternRewriter::InsertionGuard guard(b);
      if (cache)
        restoreInsertionPointIfSet(insertPt, b);
      SmallVector<SmallVector<Value>> result;
      if (auto blocked = layout.dyn_cast<BlockedEncodingAttr>()) {
        result = emitIndicesForDistributedLayout(loc, b, blocked, type);
      } else if (auto mma = layout.dyn_cast<MmaEncodingAttr>()) {
        result = emitIndicesForDistributedLayout(loc, b, mma, type);
      } else if (auto slice = layout.dyn_cast<SliceEncodingAttr>()) {
        result = emitIndicesForDistributedLayout(loc, b, slice, type);
      } else {
        llvm_unreachable(
            "emitIndices for layouts other than blocked & slice not "
            "implemented yet");
      }
      if (cache) {
        cache->insert(std::make_pair(key, result));
        *insertPt = b.saveInsertionPoint();
      }
      return result;
    }
  }

private:
  void restoreInsertionPointIfSet(OpBuilder::InsertPoint *insertPt,
                                  ConversionPatternRewriter &rewriter) const {
    if (insertPt->isSet()) {
      rewriter.restoreInsertionPoint(*insertPt);
    } else {
      auto func =
          rewriter.getInsertionPoint()->getParentOfType<LLVM::LLVMFuncOp>();
      rewriter.setInsertionPointToStart(&func.getBody().front());
    }
  }

  // -----------------------------------------------------------------------
  // Blocked layout indices
  // -----------------------------------------------------------------------

  // Get an index-base for each dimension for a \param blocked_layout.
  SmallVector<Value> emitBaseIndexForBlockedLayout(
      Location loc, ConversionPatternRewriter &rewriter,
      const BlockedEncodingAttr &blocked_layout, RankedTensorType type) const {
    auto shape = type.getShape();
    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(32);
    Value laneId = urem(threadId, warpSize);
    Value warpId = udiv(threadId, warpSize);
    auto sizePerThread = blocked_layout.getSizePerThread();
    auto threadsPerWarp = blocked_layout.getThreadsPerWarp();
    auto warpsPerCTA = blocked_layout.getWarpsPerCTA();
    auto order = blocked_layout.getOrder();
    unsigned rank = shape.size();

    // delinearize threadId to get the base index
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, order);
    SmallVector<Value> multiDimThreadId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);

    SmallVector<Value> multiDimBase(rank);
    for (unsigned k = 0; k < rank; ++k) {
      // Wrap around multiDimWarpId/multiDimThreadId in case
      // shape[k] > shapePerCTA[k]
      auto maxWarps =
          ceil<unsigned>(shape[k], sizePerThread[k] * threadsPerWarp[k]);
      auto maxThreads = ceil<unsigned>(shape[k], sizePerThread[k]);
      multiDimWarpId[k] = urem(multiDimWarpId[k], i32_val(maxWarps));
      multiDimThreadId[k] = urem(multiDimThreadId[k], i32_val(maxThreads));
      // multiDimBase[k] = (multiDimThreadId[k] +
      //                    multiDimWarpId[k] * threadsPerWarp[k]) *
      //                   sizePerThread[k];
      Value threadsPerWarpK = i32_val(threadsPerWarp[k]);
      Value sizePerThreadK = i32_val(sizePerThread[k]);
      multiDimBase[k] =
          mul(sizePerThreadK, add(multiDimThreadId[k],
                                  mul(multiDimWarpId[k], threadsPerWarpK)));
    }
    return multiDimBase;
  }

  SmallVector<SmallVector<unsigned>>
  emitOffsetForBlockedLayout(const BlockedEncodingAttr &blockedLayout,
                             RankedTensorType type) const {
    auto shape = type.getShape();
    auto sizePerThread = blockedLayout.getSizePerThread();
    auto threadsPerWarp = blockedLayout.getThreadsPerWarp();
    auto warpsPerCTA = blockedLayout.getWarpsPerCTA();
    auto order = blockedLayout.getOrder();

    unsigned rank = shape.size();
    SmallVector<unsigned> shapePerCTA = getShapePerCTA(blockedLayout);
    SmallVector<unsigned> tilesPerDim(rank);
    for (unsigned k = 0; k < rank; ++k)
      tilesPerDim[k] = ceil<unsigned>(shape[k], shapePerCTA[k]);

    SmallVector<SmallVector<unsigned>> offset(rank);
    for (unsigned k = 0; k < rank; ++k) {
      // 1 block in minimum if shape[k] is less than shapePerCTA[k]
      for (unsigned blockOffset = 0; blockOffset < tilesPerDim[k];
           ++blockOffset)
        for (unsigned warpOffset = 0; warpOffset < warpsPerCTA[k]; ++warpOffset)
          for (unsigned threadOffset = 0; threadOffset < threadsPerWarp[k];
               ++threadOffset)
            for (unsigned elemOffset = 0; elemOffset < sizePerThread[k];
                 ++elemOffset)
              offset[k].push_back(blockOffset * sizePerThread[k] *
                                      threadsPerWarp[k] * warpsPerCTA[k] +
                                  warpOffset * sizePerThread[k] *
                                      threadsPerWarp[k] +
                                  threadOffset * sizePerThread[k] + elemOffset);
    }

    unsigned elemsPerThread = triton::gpu::getTotalElemsPerThread(type);
    unsigned totalSizePerThread = product<unsigned>(sizePerThread);
    SmallVector<SmallVector<unsigned>> reorderedOffset(elemsPerThread);
    for (unsigned n = 0; n < elemsPerThread; ++n) {
      unsigned linearNanoTileId = n / totalSizePerThread;
      unsigned linearNanoTileElemId = n % totalSizePerThread;
      SmallVector<unsigned> multiDimNanoTileId =
          getMultiDimIndex<unsigned>(linearNanoTileId, tilesPerDim, order);
      SmallVector<unsigned> multiDimNanoTileElemId = getMultiDimIndex<unsigned>(
          linearNanoTileElemId, sizePerThread, order);
      for (unsigned k = 0; k < rank; ++k) {
        unsigned reorderedMultiDimId =
            multiDimNanoTileId[k] *
                (sizePerThread[k] * threadsPerWarp[k] * warpsPerCTA[k]) +
            multiDimNanoTileElemId[k];
        reorderedOffset[n].push_back(offset[k][reorderedMultiDimId]);
      }
    }
    return reorderedOffset;
  }

  // -----------------------------------------------------------------------
  // Mma layout indices
  // -----------------------------------------------------------------------

  SmallVector<Value>
  emitBaseIndexForMmaLayoutV1(Location loc, ConversionPatternRewriter &rewriter,
                              const MmaEncodingAttr &mmaLayout,
                              RankedTensorType type) const {
    auto shape = type.getShape();

    auto wpt = mmaLayout.getWarpsPerCTA();
    static constexpr std::array<int, 3> fpw{{2, 2, 1}};
    auto [isARow, isBRow, isAVec4, isBVec4, _] =
        mmaLayout.decodeVoltaLayoutStates();

    Value thread = getThreadId(rewriter, loc);
    auto *ctx = thread.getContext();
    Value _1 = i32_val(1);
    Value _2 = i32_val(2);
    Value _4 = i32_val(4);
    Value _16 = i32_val(16);
    Value _32 = i32_val(32);
    Value _fpw0 = i32_val(fpw[0]);
    Value _fpw1 = i32_val(fpw[1]);

    // A info
    auto aEncoding = DotOperandEncodingAttr::get(ctx, 0, mmaLayout, 0);
    auto aRep = aEncoding.getMMAv1Rep();
    auto aSpw = aEncoding.getMMAv1ShapePerWarp();
    // B info
    auto bEncoding = DotOperandEncodingAttr::get(ctx, 1, mmaLayout, 0);
    auto bSpw = bEncoding.getMMAv1ShapePerWarp();
    auto bRep = bEncoding.getMMAv1Rep();

    SmallVector<int, 2> rep({aRep[0], bRep[1]});
    SmallVector<int, 2> spw({aSpw[0], bSpw[1]});
    SmallVector<unsigned, 2> shapePerCTA({spw[0] * wpt[0], spw[1] * wpt[1]});

    Value lane = urem(thread, _32);
    Value warp = udiv(thread, _32);

    Value warp0 = urem(warp, i32_val(wpt[0]));
    Value warp12 = udiv(warp, i32_val(wpt[0]));
    Value warp1 = urem(warp12, i32_val(wpt[1]));

    // warp offset
    Value offWarpM = mul(warp0, i32_val(spw[0]));
    Value offWarpN = mul(warp1, i32_val(spw[1]));
    // quad offset
    Value offQuadM = mul(udiv(and_(lane, _16), _4), _fpw0);
    Value offQuadN = mul(udiv(and_(lane, _16), _4), _fpw1);
    // pair offset
    Value offPairM = udiv(urem(lane, _16), _4);
    offPairM = urem(offPairM, _fpw0);
    offPairM = mul(offPairM, _4);
    Value offPairN = udiv(urem(lane, _16), _4);
    offPairN = udiv(offPairN, _fpw0);
    offPairN = urem(offPairN, _fpw1);
    offPairN = mul(offPairN, _4);
    offPairM = mul(offPairM, i32_val(rep[0] / 2));
    offQuadM = mul(offQuadM, i32_val(rep[0] / 2));
    offPairN = mul(offPairN, i32_val(rep[1] / 2));
    offQuadN = mul(offQuadN, i32_val(rep[1] / 2));
    // quad pair offset
    Value offLaneM = add(offPairM, offQuadM);
    Value offLaneN = add(offPairN, offQuadN);
    // a, b offset
    Value offsetAM = add(offWarpM, offLaneM);
    Value offsetBN = add(offWarpN, offLaneN);
    // m indices
    Value offsetCM = add(and_(lane, _1), offsetAM);
    // n indices
    Value offsetCN = add((and_(lane, _2)), (add(offWarpN, offPairN)));
    return {offsetCM, offsetCN};
  }

  SmallVector<SmallVector<unsigned>>
  emitOffsetForMmaLayoutV1(const MmaEncodingAttr &mmaLayout,
                           RankedTensorType type) const {
    auto shape = type.getShape();

    auto [isARow, isBRow, isAVec4, isBVec4, _] =
        mmaLayout.decodeVoltaLayoutStates();

    // TODO: seems like the apttern below to get `rep`/`spw` appears quite often
    // A info
    auto aEncoding =
        DotOperandEncodingAttr::get(type.getContext(), 0, mmaLayout, 0);
    auto aRep = aEncoding.getMMAv1Rep();
    auto aSpw = aEncoding.getMMAv1ShapePerWarp();
    // B info
    auto bEncoding =
        DotOperandEncodingAttr::get(type.getContext(), 1, mmaLayout, 0);
    auto bSpw = bEncoding.getMMAv1ShapePerWarp();
    auto bRep = bEncoding.getMMAv1Rep();

    auto wpt = mmaLayout.getWarpsPerCTA();
    static constexpr std::array<int, 3> fpw{{2, 2, 1}};
    SmallVector<int, 2> rep({aRep[0], bRep[1]});
    SmallVector<int, 2> spw({aSpw[0], bSpw[1]});
    SmallVector<unsigned, 2> shapePerCTA({spw[0] * wpt[0], spw[1] * wpt[1]});

    SmallVector<unsigned> idxM;
    for (unsigned m = 0; m < shape[0]; m += shapePerCTA[0])
      for (unsigned mm = 0; mm < rep[0]; ++mm)
        idxM.push_back(m + mm * 2);

    SmallVector<unsigned> idxN;
    for (int n = 0; n < shape[1]; n += shapePerCTA[1]) {
      for (int nn = 0; nn < rep[1]; ++nn) {
        idxN.push_back(n + nn / 2 * 4 + (nn % 2) * 2 * fpw[1] * rep[1]);
        idxN.push_back(n + nn / 2 * 4 + (nn % 2) * 2 * fpw[1] * rep[1] + 1);
      }
    }

    SmallVector<SmallVector<unsigned>> ret;
    for (unsigned x1 : idxN) {   // N
      for (unsigned x0 : idxM) { // M
        SmallVector<unsigned> idx(2);
        idx[0] = x0; // M
        idx[1] = x1; // N
        ret.push_back(std::move(idx));
      }
    }
    return ret;
  }

  SmallVector<Value>
  emitBaseIndexForMmaLayoutV2(Location loc, ConversionPatternRewriter &rewriter,
                              const MmaEncodingAttr &mmaLayout,
                              RankedTensorType type) const {
    auto shape = type.getShape();
    auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
    assert(warpsPerCTA.size() == 2);
    auto order = triton::gpu::getOrder(mmaLayout);
    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(32);
    Value laneId = urem(threadId, warpSize);
    Value warpId = udiv(threadId, warpSize);

    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, order);
    unsigned lastAxis = order[order.size() - 1];
    multiDimWarpId[lastAxis] =
        urem(multiDimWarpId[lastAxis], i32_val(warpsPerCTA[lastAxis]));
    multiDimWarpId[0] = urem(multiDimWarpId[0], i32_val(shape[0] / 16));
    multiDimWarpId[1] = urem(multiDimWarpId[1], i32_val(shape[1] / 8));
    Value offWarp0 = mul(multiDimWarpId[0], i32_val(16));
    Value offWarp1 = mul(multiDimWarpId[1], i32_val(8));

    SmallVector<Value> multiDimBase(2);
    multiDimBase[0] = add(udiv(laneId, i32_val(4)), offWarp0);
    multiDimBase[1] = add(mul(i32_val(2), urem(laneId, i32_val(4))), offWarp1);
    return multiDimBase;
  }

  SmallVector<SmallVector<unsigned>>
  emitOffsetForMmaLayoutV2(const MmaEncodingAttr &mmaLayout,
                           RankedTensorType type) const {
    auto shape = type.getShape();
    SmallVector<SmallVector<unsigned>> ret;

    for (unsigned i = 0; i < shape[0]; i += getShapePerCTA(mmaLayout)[0]) {
      for (unsigned j = 0; j < shape[1]; j += getShapePerCTA(mmaLayout)[1]) {
        ret.push_back({i, j});
        ret.push_back({i, j + 1});
        ret.push_back({i + 8, j});
        ret.push_back({i + 8, j + 1});
      }
    }
    return ret;
  }

  // Emit indices calculation within each ConversionPattern, and returns a
  // [elemsPerThread X rank] index matrix.
  SmallVector<SmallVector<Value>> emitIndicesForDistributedLayout(
      Location loc, ConversionPatternRewriter &rewriter, Attribute layout,
      RankedTensorType type) const {
    // step 1, delinearize threadId to get the base index
    auto multiDimBase = emitBaseIndexForLayout(loc, rewriter, layout, type);
    // step 2, get offset of each element
    auto offset = emitOffsetForLayout(layout, type);
    // step 3, add offset to base, and reorder the sequence of indices to
    // guarantee that elems in the same sizePerThread are adjacent in order
    auto shape = type.getShape();
    unsigned rank = shape.size();
    unsigned elemsPerThread = offset.size();
    SmallVector<SmallVector<Value>> multiDimIdx(elemsPerThread,
                                                SmallVector<Value>(rank));
    for (unsigned n = 0; n < elemsPerThread; ++n)
      for (unsigned k = 0; k < rank; ++k)
        multiDimIdx[n][k] = add(multiDimBase[k], i32_val(offset[n][k]));
    return multiDimIdx;
  }

  SmallVector<SmallVector<unsigned>>
  emitOffsetForSliceLayout(const SliceEncodingAttr &sliceLayout,
                           RankedTensorType type) const {
    auto parentEncoding = sliceLayout.getParent();
    unsigned dim = sliceLayout.getDim();
    auto parentShape = sliceLayout.paddedShape(type.getShape());
    RankedTensorType parentTy = RankedTensorType::get(
        parentShape, type.getElementType(), parentEncoding);
    auto parentOffsets = emitOffsetForLayout(parentEncoding, parentTy);

    unsigned numOffsets = parentOffsets.size();
    SmallVector<SmallVector<unsigned>> resultOffsets;
    std::set<SmallVector<unsigned>> uniqueOffsets;

    for (unsigned i = 0; i < numOffsets; ++i) {
      SmallVector<unsigned> offsets = parentOffsets[i];
      offsets.erase(offsets.begin() + dim);
      if (uniqueOffsets.find(offsets) == uniqueOffsets.end()) {
        resultOffsets.push_back(offsets);
        uniqueOffsets.insert(offsets);
      }
    }
    return resultOffsets;
  }

protected:
  TritonGPUToLLVMTypeConverter *converter;
  ModuleAllocation *allocation;
  IndexCacheInfo indexCacheInfo;
};

template <typename SourceOp>
class ConvertTritonGPUOpToLLVMPattern
    : public ConvertOpToLLVMPattern<SourceOp>,
      public ConvertTritonGPUOpToLLVMPatternBase {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ConvertTritonGPUOpToLLVMPattern(
      TritonGPUToLLVMTypeConverter &typeConverter, PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        ConvertTritonGPUOpToLLVMPatternBase(typeConverter) {}

  explicit ConvertTritonGPUOpToLLVMPattern(
      TritonGPUToLLVMTypeConverter &typeConverter,
      IndexCacheInfo indexCacheInfo, PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        ConvertTritonGPUOpToLLVMPatternBase(typeConverter, indexCacheInfo) {}

  explicit ConvertTritonGPUOpToLLVMPattern(
      TritonGPUToLLVMTypeConverter &typeConverter, ModuleAllocation &allocation,
      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        ConvertTritonGPUOpToLLVMPatternBase(typeConverter, allocation) {}

  explicit ConvertTritonGPUOpToLLVMPattern(
      TritonGPUToLLVMTypeConverter &typeConverter, ModuleAllocation &allocation,
      IndexCacheInfo indexCacheInfo, PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        ConvertTritonGPUOpToLLVMPatternBase(typeConverter, allocation,
                                            indexCacheInfo) {}

protected:
  TritonGPUToLLVMTypeConverter *getTypeConverter() const {
    LLVMTypeConverter *ret =
        ((ConvertTritonGPUOpToLLVMPatternBase *)this)->getTypeConverter();
    return (TritonGPUToLLVMTypeConverter *)ret;
  }
};

#endif
