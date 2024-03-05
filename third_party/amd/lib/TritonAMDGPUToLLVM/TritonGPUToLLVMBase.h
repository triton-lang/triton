#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_BASE_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_BASE_H

// TODO: refactor so that it doesn't fail if Allocation.h
// is included after utility.h (due to conflict in `store` macro
// and <atomic>
#include "triton/Analysis/Allocation.h"

#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
//
#include "Utility.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <set>
#include <type_traits>

#define DEBUG_TYPE "ttgpu_to_llvm"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::CTALayoutAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;
namespace ttng = ::mlir::triton::nvidia_gpu;

typedef DenseMap<Operation *, triton::MakeTensorPtrOp> TensorPtrMapT;

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
        /*dsoLocal*/ false, LLVM::CConv::C, /*comdat=*/SymbolRefAttr{},
        attributes);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &result)))
      return nullptr;

    return newFuncOp;
  }
};

struct IndexCacheKeyT {
  Attribute layout;
  RankedTensorType type;
  bool withCTAOffset;
};

struct CacheKeyDenseMapInfo {
  static IndexCacheKeyT getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return {mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer)),
            RankedTensorType{}, true};
  }
  static IndexCacheKeyT getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    auto tombstone = llvm::DenseMapInfo<RankedTensorType>::getTombstoneKey();
    return {mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer)),
            tombstone, true};
  }
  static unsigned getHashValue(IndexCacheKeyT key) {
    return llvm::hash_combine(mlir::hash_value(key.layout),
                              mlir::hash_value(key.type),
                              llvm::hash_value(key.withCTAOffset));
  }
  static bool isEqual(IndexCacheKeyT LHS, IndexCacheKeyT RHS) {
    return LHS.layout == RHS.layout && LHS.type == RHS.type &&
           LHS.withCTAOffset == RHS.withCTAOffset;
  }
};

namespace mlir::triton {
class ReduceOp;
class ScanOp;
} // namespace mlir::triton

namespace AMD {
class ConvertTritonGPUOpToLLVMPatternBase {
public:
  // Two levels of value cache in emitting indices calculation:
  // Key: {layout, shape, withCTAOffset}
  struct IndexCacheInfo {
    DenseMap<IndexCacheKeyT, SmallVector<Value>, CacheKeyDenseMapInfo>
        *baseIndexCache = nullptr;
    DenseMap<IndexCacheKeyT, SmallVector<SmallVector<Value>>,
             CacheKeyDenseMapInfo> *indexCache = nullptr;
    OpBuilder::InsertPoint *indexInsertPoint = nullptr;
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

  // Returns CTA level thread idx
  Value getThreadIdInCTA(ConversionPatternRewriter &rewriter,
                         Location loc) const {
    Value tid = rewriter.create<::mlir::gpu::ThreadIdOp>(
        loc, ::mlir::gpu::Dimension::x);
    return rewriter.create<arith::IndexCastOp>(loc, i32_ty, tid);
  }

  // Returns CTA level thread idx.
  Value getThreadId(ConversionPatternRewriter &rewriter, Location loc) const {
    Value tid = getThreadIdInCTA(rewriter, loc);
    auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    return tid;
  }

  Value getClusterCTAId(ConversionPatternRewriter &rewriter,
                        Location loc) const {
    return rewriter.create<triton::nvgpu::ClusterCTAIdOp>(
        loc, rewriter.getI32Type());
  }

  // -----------------------------------------------------------------------
  // Shared memory utilities
  // -----------------------------------------------------------------------
  template <typename T>
  Value getSharedMemoryBase(Location loc, ConversionPatternRewriter &rewriter,
                            T value) const {
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
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
    Value base =
        gep(ptrTy, this->getTypeConverter()->convertType(rewriter.getI8Type()),
            smem, offVal);
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
    //   phase = (row // perPhase) % maxPhase
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
    auto dstPtrTy = ptr_ty(rewriter.getContext(), 3);
    auto dstOffset = dot(rewriter, loc, offsetVals, smemObj.strides);
    Value dstPtrBase = gep(dstPtrTy, getTypeConverter()->convertType(resElemTy),
                           smemObj.base, dstOffset);

    auto srcEncoding = srcTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto srcShapePerCTA = triton::gpu::getShapePerCTA(srcTy);
    unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
    // swizzling params as described in TritonGPUAttrDefs.td
    unsigned outVec = resSharedLayout.getVec();
    unsigned perPhase = resSharedLayout.getPerPhase();
    unsigned maxPhase = resSharedLayout.getMaxPhase();
    // Order
    auto inOrder = triton::gpu::getOrder(srcEncoding);
    auto outOrder = triton::gpu::getOrder(resSharedLayout);
    assert(maxPhase == 1 ||
           outVec * maxPhase <= srcShape[outOrder[0]] &&
               "Swizzling would generate out of bounds memory accesses");
    // Tensor indices held by the current thread, as LLVM values
    auto srcIndices = emitIndices(loc, rewriter, srcEncoding, srcTy, false);
    // Swizzling with leading offsets (e.g. Hopper GMMA)
    unsigned swizzlingByteWidth = 0;
    if (resSharedLayout.getHasLeadingOffset()) {
      if (perPhase == 4 && maxPhase == 2)
        swizzlingByteWidth = 32;
      else if (perPhase == 2 && maxPhase == 4)
        swizzlingByteWidth = 64;
      else if (perPhase == 1 && maxPhase == 8)
        swizzlingByteWidth = 128;
      else
        llvm::report_fatal_error("Unsupported shared layout.");
    }
    unsigned numElemsPerSwizzlingRow =
        swizzlingByteWidth * 8 / resElemTy.getIntOrFloatBitWidth();
    Value numElemsPerSwizzlingRowVal = i32_val(numElemsPerSwizzlingRow);
    unsigned leadingDimOffset;
    if (outOrder.size() == 2) {
      leadingDimOffset = numElemsPerSwizzlingRow * srcShapePerCTA[outOrder[1]];
    } else {
      leadingDimOffset = numElemsPerSwizzlingRow;
    }

    Value leadingDimOffsetVal = i32_val(leadingDimOffset);
    // Return values
    DenseMap<unsigned, Value> ret;
    // cache for non-immediate offsets
    DenseMap<unsigned, Value> cacheCol, cacheRow;
    unsigned minVec = std::min(outVec, inVec);
    for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
      Value offset = i32_val(0);
      // Extract multi dimensional index for current element
      auto idx = srcIndices[elemIdx];
      Value idxCol = idx[outOrder[0]]; // contiguous dimension
      Value idxRow, strideRow;
      if (outOrder.size() == 2) {
        idxRow = idx[outOrder[1]]; // discontiguous dimension
        strideRow = srcStrides[outOrder[1]];
      } else {
        idxRow = i32_val(0);
        strideRow = i32_val(0);
      }
      Value strideCol = srcStrides[outOrder[0]];
      // compute phase = (row // perPhase) % maxPhase
      Value phase = urem(udiv(idxRow, i32_val(perPhase)), i32_val(maxPhase));
      // extract dynamic/static offset for immediate offsetting
      unsigned immedateOffCol = 0;
      unsigned immedateOffRow = 0;
      if (leadingDimOffset) {
        // hopper
        offset =
            mul(udiv(idxCol, numElemsPerSwizzlingRowVal), leadingDimOffsetVal);
        // Shrink by swizzling blocks
        idxCol = urem(idxCol, numElemsPerSwizzlingRowVal);
        strideRow = numElemsPerSwizzlingRowVal;
      } else {
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
        if (auto add = dyn_cast_or_null<LLVM::AddOp>(idxRow.getDefiningOp()))
          if (auto _cst = dyn_cast_or_null<LLVM::ConstantOp>(
                  add.getRhs().getDefiningOp())) {
            unsigned cst =
                _cst.getValue().cast<IntegerAttr>().getValue().getSExtValue();
            unsigned key = cst % (perPhase * maxPhase);
            cacheRow.insert({key, idxRow});
            idxRow = cacheRow[key];
            immedateOffRow =
                cst / (perPhase * maxPhase) * (perPhase * maxPhase);
          }
      }
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
      offset = add(offset, add(rowOff, mul(colOff, strideCol)));
      Value currPtr = gep(dstPtrTy, getTypeConverter()->convertType(resElemTy),
                          dstPtrBase, offset);
      // compute immediate offset
      Value immediateOff;
      if (outOrder.size() == 2) {
        immediateOff =
            add(mul(i32_val(immedateOffRow), srcStrides[outOrder[1]]),
                i32_val(immedateOffCol));
      } else {
        immediateOff = i32_val(immedateOffCol);
      }

      ret[elemIdx] = gep(dstPtrTy, getTypeConverter()->convertType(resElemTy),
                         currPtr, immediateOff);
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
    assert(dstShape.size() <= 2 &&
           "Unexpected rank of loadSharedToDistributed");
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto dstDistributedLayout = dstTy.getEncoding();
    if (auto mmaLayout =
            dstDistributedLayout.dyn_cast<NvidiaMmaEncodingAttr>()) {
      assert((!mmaLayout.isVolta()) &&
             "ConvertLayout Shared->MMAv1 is not supported yet");
    }
    auto srcSharedLayout =
        srcTy.getEncoding().cast<triton::gpu::SharedEncodingAttr>();
    auto srcElemTy = srcTy.getElementType();
    auto dstElemTy = dstTy.getElementType();
    auto inOrd = triton::gpu::getOrder(srcSharedLayout);
    auto outOrd = triton::gpu::getOrder(dstDistributedLayout);
    unsigned outVec = inOrd == outOrd
                          ? triton::gpu::getUniqueContigPerThread(
                                dstDistributedLayout, dstShape)[outOrd[0]]
                          : 1;
    unsigned inVec = srcSharedLayout.getVec();
    unsigned minVec = std::min(outVec, inVec);
    unsigned outElems = triton::gpu::getTotalElemsPerThread(dstTy);
    SmallVector<Value> offsetVals(smemObj.strides.size(), i32_val(0));
    assert(outElems == dstIndices.size());

    DenseMap<unsigned, Value> sharedPtrs =
        getSwizzledSharedPtrs(loc, outVec, dstTy, srcSharedLayout, srcElemTy,
                              smemObj, rewriter, offsetVals, smemObj.strides);
    assert(outElems % minVec == 0 && "Unexpected number of elements");
    unsigned numVecs = outElems / minVec;
    auto wordTy = vec_ty(elemTy, minVec);
    SmallVector<Value> outVals(outElems);
    for (unsigned i = 0; i < numVecs; ++i) {
      Value smemAddr = sharedPtrs[i * minVec];
      smemAddr = bitcast(smemAddr, ptr_ty(rewriter.getContext(), 3));
      Value valVec = load(wordTy, smemAddr);
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
    if (auto mmaLayout =
            srcDistributedLayout.dyn_cast<NvidiaMmaEncodingAttr>()) {
      assert((!mmaLayout.isVolta()) &&
             "ConvertLayout MMAv1->Shared is not supported yet");
    }
    auto dstSharedLayout =
        dstTy.getEncoding().cast<triton::gpu::SharedEncodingAttr>();
    auto dstElemTy = dstTy.getElementType();
    auto inOrd = triton::gpu::getOrder(srcDistributedLayout);
    auto outOrd = dstSharedLayout.getOrder();
    unsigned inVec = inOrd == outOrd
                         ? triton::gpu::getUniqueContigPerThread(
                               srcDistributedLayout, srcShape)[inOrd[0]]
                         : 1;
    unsigned outVec = dstSharedLayout.getVec();
    unsigned minVec = std::min(outVec, inVec);
    unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
    assert(numElems == srcIndices.size());
    auto inVals = unpackLLElements(loc, llSrc, rewriter);
    auto wordTy = vec_ty(elemTy, minVec);
    Value word;

    SmallVector<Value> srcStrides = {dstStrides[0], dstStrides[1]};
    SmallVector<Value> offsetVals = {i32_val(0), i32_val(0)};
    SharedMemoryObject smemObj(smemBase, elemTy, srcStrides, offsetVals);

    DenseMap<unsigned, Value> sharedPtrs =
        getSwizzledSharedPtrs(loc, inVec, srcTy, dstSharedLayout, dstElemTy,
                              smemObj, rewriter, offsetVals, srcStrides);

    for (unsigned i = 0; i < numElems; ++i) {
      if (i % minVec == 0)
        word = undef(wordTy);
      word = insert_element(wordTy, word, inVals[i], i32_val(i % minVec));
      if (i % minVec == minVec - 1) {
        Value smemAddr = sharedPtrs[i / minVec * minVec];
        smemAddr = bitcast(smemAddr, ptr_ty(rewriter.getContext(), 3));
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
    auto clusterCTAId = getClusterCTAId(rewriter, loc);
    if (tensorTy) {
      auto layout = tensorTy.getEncoding();
      auto shape = tensorTy.getShape();
      unsigned rank = shape.size();
      auto sizePerThread = triton::gpu::getSizePerThread(layout);
      auto threadsPerWarp = triton::gpu::getThreadsPerWarp(layout);
      auto warpsPerCTA = triton::gpu::getWarpsPerCTA(layout);
      auto order = triton::gpu::getOrder(layout);
      auto shapePerCTATile = triton::gpu::getShapePerCTATile(layout, shape);
      Value warpSize = i32_val(triton::gpu::getWarpSize(layout));
      Value laneId = urem(tid, warpSize);
      Value warpId = udiv(tid, warpSize);
      SmallVector<Value> multiDimWarpId =
          delinearize(rewriter, loc, warpId, warpsPerCTA, order);
      SmallVector<Value> multiDimThreadId =
          delinearize(rewriter, loc, laneId, threadsPerWarp, order);
      for (unsigned dim = 0; dim < rank; ++dim) {
        // if there is no data replication across threads on this dimension
        if (shape[dim] >= shapePerCTATile[dim])
          continue;
        // Otherwise, we need to mask threads that will replicate data on this
        // dimension. Calculate the thread index on this dimension for the CTA
        Value threadDim =
            add(mul(multiDimWarpId[dim], i32_val(threadsPerWarp[dim])),
                multiDimThreadId[dim]);
        mask = and_(mask, icmp_slt(mul(threadDim, i32_val(sizePerThread[dim])),
                                   i32_val(shape[dim])));
      }
      // Do not write duplicated data when multicast is enabled
      if (triton::gpu::getNumCTAs(layout) > 1) {
        auto _0 = i32_val(0);
        auto CTAsPerCGA = triton::gpu::getCTAsPerCGA(layout);
        auto CTASplitNum = triton::gpu::getCTASplitNum(layout);
        auto CTAOrder = triton::gpu::getCTAOrder(layout);

        auto multiDimClusterCTAId =
            delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

        for (unsigned dim = 0; dim < rank; ++dim) {
          // Skip when multicast is not enabled in this dimension
          if (CTAsPerCGA[dim] == CTASplitNum[dim])
            continue;
          // This wrapping rule must be consistent with emitCTAOffsetForLayout
          unsigned splitNum = std::min<unsigned>(shape[dim], CTASplitNum[dim]);
          Value repId = udiv(multiDimClusterCTAId[dim], i32_val(splitNum));
          // Consider the example where CTAsPerCGA = [4] and CTASplitNum = [2]:
          //     CTA0 and CTA2 holds data of block0,
          //     CTA1 and CTA3 holds data of block1.
          // Only CTA0 and CTA1 are expected to write while CTA2 and CTA3 should
          // be masked. We add the following mask:
          //     multiDimClusterCTAId[dim] / splitNum == 0
          // Actually in all existing cases of multicast, splitNum is always 1.
          // The mask is equivalent to:
          //     multiDimClusterCTAId[dim] == 0
          mask = and_(mask, icmp_eq(repId, _0));
        }
      }
    } else {
      // If the tensor is not ranked, then it is a scalar and only thread 0 of
      // CTA0 can write
      mask = and_(mask, icmp_eq(clusterCTAId, i32_val(0)));
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
      TritonGPUToLLVMTypeConverter &typeConverter,
      PatternBenefit benefit = patternBenefitDefault)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        ConvertTritonGPUOpToLLVMPatternBase(typeConverter) {}

  explicit ConvertTritonGPUOpToLLVMPattern(
      TritonGPUToLLVMTypeConverter &typeConverter, ModuleAllocation &allocation,
      PatternBenefit benefit = patternBenefitDefault)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        ConvertTritonGPUOpToLLVMPatternBase(typeConverter, allocation) {}

  explicit ConvertTritonGPUOpToLLVMPattern(
      TritonGPUToLLVMTypeConverter &typeConverter,
      IndexCacheInfo indexCacheInfo,
      PatternBenefit benefit = patternBenefitDefault)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        ConvertTritonGPUOpToLLVMPatternBase(typeConverter, indexCacheInfo) {}

  explicit ConvertTritonGPUOpToLLVMPattern(
      TritonGPUToLLVMTypeConverter &typeConverter, ModuleAllocation &allocation,
      IndexCacheInfo indexCacheInfo,
      PatternBenefit benefit = patternBenefitDefault)
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

template <typename SourceOp>
class ConvertTritonGPUReduceScanToLLVMPattern
    : public ConvertTritonGPUOpToLLVMPattern<SourceOp> {
public:
  // Make sure the class is only instantiated with Reduce and Scan
  static_assert(std::is_same_v<SourceOp, ReduceOp> ||
                std::is_same_v<SourceOp, ScanOp>);

  using ConvertTritonGPUOpToLLVMPatternBase::getSharedMemoryBase;
  using ConvertTritonGPUOpToLLVMPatternBase::getTypeConverter;
  using ConvertTritonGPUOpToLLVMPattern<
      SourceOp>::ConvertTritonGPUOpToLLVMPattern;

  // Return the pointee type of the shared memory pointer for operand i.
  Type getElementType(SourceOp op, int i) const {
    auto ty = op.getInputTypes()[i].getElementType();
    return getTypeConverter()->convertType(ty);
  }

  // Helper to compute the smem bases in both reductions and scans
  SmallVector<Value> getSmemBases(SourceOp op, unsigned elems,
                                  ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    // indices will store the index of the op operands in descending order
    // of their bitwidths
    std::vector<unsigned> indices(op.getNumOperands());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](unsigned i, unsigned j) {
      return op.getElementTypes()[i].getIntOrFloatBitWidth() >
             op.getElementTypes()[j].getIntOrFloatBitWidth();
    });
    // Assign base index to each operand in their order in indices
    std::map<unsigned, Value> indexToBase;
    indexToBase[indices[0]] =
        getSharedMemoryBase(loc, rewriter, op.getOperation());
    for (unsigned i = 1; i < op.getNumOperands(); ++i) {
      indexToBase[indices[i]] = gep(
          ptr_ty(rewriter.getContext(), 3), getElementType(op, indices[i - 1]),
          indexToBase[indices[i - 1]], i32_val(elems));
    }
    // smemBases[k] is the base pointer for the k-th operand
    SmallVector<Value> smemBases(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      smemBases[i] = indexToBase[i];
    }
    return smemBases;
  }
};
} // namespace AMD
#endif
