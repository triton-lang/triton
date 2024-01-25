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
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <set>
#include <type_traits>

#define DEBUG_TYPE "ttgpu_to_llvm"

constexpr ::llvm::StringLiteral kAttrNumTMALoadDescsName =
    "triton_gpu.num-tma-load";
constexpr ::llvm::StringLiteral kAttrNumTMAStoreDescsName =
    "triton_gpu.num-tma-store";
using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::SharedMemoryObject;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::CTALayoutAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
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

class ConvertTritonGPUOpToLLVMPatternBase {
public:
  explicit ConvertTritonGPUOpToLLVMPatternBase(
      TritonGPUToLLVMTypeConverter &typeConverter)
      : converter(&typeConverter) {}

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

  // Returns CTA level thread idx for not ws mode.
  // Returns agent level thread idx for ws mode.
  Value getThreadId(ConversionPatternRewriter &rewriter, Location loc) const {
    Value tid = getThreadIdInCTA(rewriter, loc);
    auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    if (ttng::TritonNvidiaGPUDialect::getWSSupportedAttr(mod)) {
      Value _128 = rewriter.create<arith::ConstantIntOp>(loc, 128, 32);
      tid = rewriter.create<arith::RemSIOp>(loc, tid, _128);
    }
    return tid;
  }

  Value GetCanonicalWarpId(ConversionPatternRewriter &rewriter,
                           Location loc) const {
    return rewriter.create<triton::nvgpu::CanonicalWarpIdOp>(
        loc, rewriter.getI32Type());
  }

  Value getClusterCTAId(ConversionPatternRewriter &rewriter,
                        Location loc) const {
    return rewriter.create<triton::nvgpu::ClusterCTAIdOp>(
        loc, rewriter.getI32Type());
  }

  // -----------------------------------------------------------------------
  // Shared memory utilities
  // -----------------------------------------------------------------------

  DenseMap<unsigned, Value>
  getSwizzledSharedPtrs(Location loc, unsigned inVec, RankedTensorType srcTy,
                        triton::gpu::SharedEncodingAttr resSharedLayout,
                        Type resElemTy, SharedMemoryObject smemObj,
                        ConversionPatternRewriter &rewriter,
                        SmallVectorImpl<Value> &offsetVals,
                        SmallVectorImpl<Value> &srcStrides) const {
    // This utility computes the pointers for accessing the provided swizzled
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

  /*-------*/
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
    SmallVector<Value> offsetVals = {i32_val(0), i32_val(0)};
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
    auto inVals = getTypeConverter()->unpackLLElements(loc, llSrc, rewriter);
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
      Value warpSize = i32_val(32);
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

  // -----------------------------------------------------------------------
  // Get offsets / indices for any layout
  // -----------------------------------------------------------------------

  SmallVector<Value> emitCTAOffsetForLayout(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            Attribute layout,
                                            ArrayRef<int64_t> shape) const {
    unsigned rank = shape.size();
    SmallVector<unsigned> CTAsPerCGA = triton::gpu::getCTAsPerCGA(layout);
    SmallVector<unsigned> CTASplitNum = triton::gpu::getCTASplitNum(layout);
    SmallVector<unsigned> CTAOrder = triton::gpu::getCTAOrder(layout);
    SmallVector<int64_t> shapePerCTA =
        triton::gpu::getShapePerCTA(CTASplitNum, shape);

    // Delinearize clusterCTAId
    Value clusterCTAId = getClusterCTAId(rewriter, loc);
    SmallVector<Value> multiDimClusterCTAId =
        delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

    // CTA Wrapping
    for (unsigned i = 0; i < rank; ++i) {
      // This wrapping rule must be consistent with getShapePerCTA
      unsigned splitNum = std::min<unsigned>(shape[i], CTASplitNum[i]);
      multiDimClusterCTAId[i] =
          urem(multiDimClusterCTAId[i], i32_val(splitNum));
    }

    SmallVector<Value> CTAOffset(rank);
    for (unsigned i = 0; i < rank; ++i)
      CTAOffset[i] = mul(multiDimClusterCTAId[i], i32_val(shapePerCTA[i]));

    return CTAOffset;
  }

  SmallVector<Value> emitBaseIndexForLayout(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            Attribute layout,
                                            RankedTensorType type,
                                            bool withCTAOffset) const {
    auto shape = type.getShape();

    SmallVector<Value> baseIndex;
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    SmallVector<Value> result;
    if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
      result = emitBaseIndexWithinCTAForBlockedLayout(loc, rewriter,
                                                      blockedLayout, type);
    } else if (auto mmaLayout = layout.dyn_cast<NvidiaMmaEncodingAttr>()) {
      if (mmaLayout.isVolta())
        result = emitBaseIndexWithinCTAForMmaLayoutV1(loc, rewriter, mmaLayout,
                                                      type);
      if (mmaLayout.isAmpere() || mmaLayout.isHopper())
        result = emitBaseIndexWithinCTAForMmaLayoutV2V3(loc, rewriter,
                                                        mmaLayout, type);
    } else if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
      auto parentLayout = sliceLayout.getParent();
      auto parentShape = sliceLayout.paddedShape(type.getShape());
      RankedTensorType parentTy = RankedTensorType::get(
          parentShape, type.getElementType(), parentLayout);
      result = emitBaseIndexForLayout(loc, rewriter, parentLayout, parentTy,
                                      withCTAOffset);
      result.erase(result.begin() + sliceLayout.getDim());
      // CTAOffset has been added in emitBaseIndexForLayout of parentLayout
      return result;
    } else {
      llvm_unreachable("unsupported emitBaseIndexForLayout");
    }
    if (withCTAOffset) {
      auto CTAOffset = emitCTAOffsetForLayout(loc, rewriter, layout, shape);
      assert(CTAOffset.size() == result.size() && "Rank mismatch");
      for (unsigned k = 0; k < result.size(); ++k)
        result[k] = add(result[k], CTAOffset[k]);
    }
    return result;
  }

  SmallVector<SmallVector<unsigned>>
  emitOffsetForLayout(Attribute layout, RankedTensorType type) const {
    if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>())
      return emitOffsetForBlockedLayout(blockedLayout, type);
    if (auto mmaLayout = layout.dyn_cast<NvidiaMmaEncodingAttr>()) {
      if (mmaLayout.isVolta())
        return emitOffsetForMmaLayoutV1(mmaLayout, type);
      if (mmaLayout.isAmpere())
        return emitOffsetForMmaLayoutV2(mmaLayout, type);
      if (mmaLayout.isHopper())
        return emitOffsetForMmaLayoutV3(mmaLayout, type);
    }
    if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>())
      return emitOffsetForSliceLayout(sliceLayout, type);
    llvm_unreachable("unsupported emitOffsetForLayout");
  }

  // Emit indices calculation within each ConversionPattern, and returns a
  // [elemsPerThread X rank] index matrix.
  SmallVector<SmallVector<Value>>
  emitIndices(Location loc, ConversionPatternRewriter &rewriter,
              Attribute layout, RankedTensorType type,
              bool withCTAOffset) const {
    // step 1, delinearize threadId to get the base index
    auto multiDimBase =
        emitBaseIndexForLayout(loc, rewriter, layout, type, withCTAOffset);
    // step 2, get offset of each element
    auto offset = emitOffsetForLayout(layout, type);
    // step 3, add offset to base, and reorder the sequence
    // of indices to guarantee that elems in the same
    // sizePerThread are adjacent in order
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

  // Get an index-base for each dimension for a \param blockedLayout.
  SmallVector<Value> emitBaseIndexWithinCTAForBlockedLayout(
      Location loc, ConversionPatternRewriter &rewriter,
      const BlockedEncodingAttr &blockedLayout, RankedTensorType type) const {
    auto shape = type.getShape();
    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(32);
    Value laneId = urem(threadId, warpSize);
    Value warpId = udiv(threadId, warpSize);
    auto sizePerThread = blockedLayout.getSizePerThread();
    auto threadsPerWarp = blockedLayout.getThreadsPerWarp();
    auto warpsPerCTA = blockedLayout.getWarpsPerCTA();
    auto order = blockedLayout.getOrder();
    auto shapePerCTA = triton::gpu::getShapePerCTA(blockedLayout, shape);
    unsigned rank = shape.size();

    // delinearize threadId to get the base index
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, order);
    SmallVector<Value> multiDimThreadId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);

    SmallVector<Value> multiDimBase(rank);
    for (unsigned k = 0; k < rank; ++k) {
      // Wrap around multiDimWarpId/multiDimThreadId in case
      // shapePerCTATile[k] > shapePerCTA[k]
      auto maxWarps =
          ceil<unsigned>(shapePerCTA[k], sizePerThread[k] * threadsPerWarp[k]);
      auto maxThreads = ceil<unsigned>(shapePerCTA[k], sizePerThread[k]);
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
    auto shapePerCTATile = getShapePerCTATile(blockedLayout);
    auto shapePerCTA = triton::gpu::getShapePerCTA(blockedLayout, shape);

    unsigned rank = shape.size();
    SmallVector<unsigned> tilesPerDim(rank);
    for (unsigned k = 0; k < rank; ++k)
      tilesPerDim[k] = ceil<unsigned>(shapePerCTA[k], shapePerCTATile[k]);

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
        reorderedOffset[n].push_back(reorderedMultiDimId);
      }
    }
    return reorderedOffset;
  }

  // -----------------------------------------------------------------------
  // Mma layout indices
  // -----------------------------------------------------------------------

  SmallVector<Value> emitBaseIndexWithinCTAForMmaLayoutV1(
      Location loc, ConversionPatternRewriter &rewriter,
      const NvidiaMmaEncodingAttr &mmaLayout, RankedTensorType type) const {
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
    auto aRep = mmaLayout.getMMAv1Rep(0);
    auto aSpw = mmaLayout.getMMAv1ShapePerWarp(0);
    // B info
    auto bSpw = mmaLayout.getMMAv1ShapePerWarp(1);
    auto bRep = mmaLayout.getMMAv1Rep(1);

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
  emitOffsetForMmaLayoutV1(const NvidiaMmaEncodingAttr &mmaLayout,
                           RankedTensorType type) const {
    auto shape = type.getShape();

    auto [isARow, isBRow, isAVec4, isBVec4, _] =
        mmaLayout.decodeVoltaLayoutStates();

    // TODO: seems like the apttern below to get `rep`/`spw` appears quite often
    // A info
    auto aRep = mmaLayout.getMMAv1Rep(0);
    auto aSpw = mmaLayout.getMMAv1ShapePerWarp(0);
    // B info
    auto bSpw = mmaLayout.getMMAv1ShapePerWarp(1);
    auto bRep = mmaLayout.getMMAv1Rep(1);

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

  SmallVector<SmallVector<unsigned>>
  emitOffsetForMmaLayoutV2(const NvidiaMmaEncodingAttr &mmaLayout,
                           RankedTensorType type) const {
    auto shape = type.getShape();
    auto shapePerCTA = getShapePerCTA(mmaLayout, shape);
    SmallVector<SmallVector<unsigned>> ret;

    for (unsigned i = 0; i < shapePerCTA[0];
         i += getShapePerCTATile(mmaLayout)[0]) {
      for (unsigned j = 0; j < shapePerCTA[1];
           j += getShapePerCTATile(mmaLayout)[1]) {
        ret.push_back({i, j});
        ret.push_back({i, j + 1});
        ret.push_back({i + 8, j});
        ret.push_back({i + 8, j + 1});
      }
    }
    return ret;
  }

  SmallVector<Value> emitBaseIndexWithinCTAForMmaLayoutV2V3(
      Location loc, ConversionPatternRewriter &rewriter,
      const NvidiaMmaEncodingAttr &mmaLayout, RankedTensorType type) const {
    auto shape = type.getShape();
    auto _warpsPerCTA = mmaLayout.getWarpsPerCTA();
    assert(_warpsPerCTA.size() == 2);
    auto order = triton::gpu::getOrder(mmaLayout);
    ArrayRef<unsigned int> instrShape = mmaLayout.getInstrShape();
    SmallVector<Value> warpsPerCTA = {i32_val(_warpsPerCTA[0]),
                                      i32_val(_warpsPerCTA[1])};
    auto shapePerCTA = getShapePerCTA(mmaLayout, shape);

    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(32);
    Value laneId = urem(threadId, warpSize);
    Value warpId = udiv(threadId, warpSize);

    uint32_t repM = (_warpsPerCTA[0] * instrShape[0]) / shapePerCTA[0];
    uint32_t repN = (_warpsPerCTA[1] * instrShape[1]) / shapePerCTA[1];

    uint32_t warpsM;
    if (repM > 1)
      warpsM = _warpsPerCTA[0] / repM;
    else
      warpsM = shape[0] / instrShape[0];

    uint32_t warpsN;
    if (repN > 1)
      warpsN = _warpsPerCTA[1] / repN;
    else
      warpsN = shape[1] / instrShape[1];

    SmallVector<Value> multiDimWarpId(2);
    if (mmaLayout.isHopper()) {
      // TODO[goostavz]: the tiling order from CTA->warp level is different for
      // MMAv2/3. This is a workaround since we don't explicitly have warpGrp
      // level in the layout definition, and the tiling order of warpGrp->warp
      // must be fixed to meet the HW's needs. We may need to consider to
      // explicitly define warpGrpPerCTA for MMAv3 layout.
      multiDimWarpId[0] = urem(warpId, warpsPerCTA[0]);
      multiDimWarpId[1] = urem(udiv(warpId, warpsPerCTA[0]), warpsPerCTA[1]);
    } else {
      multiDimWarpId = delinearize(rewriter, loc, warpId, _warpsPerCTA, order);
    }
    Value warpId0 = urem(multiDimWarpId[0], i32_val(warpsM));
    Value warpId1 = urem(multiDimWarpId[1], i32_val(warpsN));

    Value offWarp0 = mul(warpId0, i32_val(instrShape[0]));
    Value offWarp1 = mul(warpId1, i32_val(instrShape[1]));

    SmallVector<Value> multiDimBase(2);
    multiDimBase[0] = add(udiv(laneId, i32_val(4)), offWarp0);
    multiDimBase[1] = add(mul(i32_val(2), urem(laneId, i32_val(4))), offWarp1);
    return multiDimBase;
  }

  SmallVector<SmallVector<unsigned>>
  emitOffsetForMmaLayoutV3(const NvidiaMmaEncodingAttr &mmaLayout,
                           RankedTensorType type) const {
    auto shape = type.getShape();
    auto shapePerCTA = getShapePerCTA(mmaLayout, shape);
    SmallVector<SmallVector<unsigned>> ret;
    ArrayRef<unsigned int> instrShape = mmaLayout.getInstrShape();

    for (unsigned i = 0; i < shapePerCTA[0];
         i += getShapePerCTATile(mmaLayout)[0]) {
      for (unsigned j = 0; j < shapePerCTA[1];
           j += getShapePerCTATile(mmaLayout)[1]) {
        for (unsigned k = 0; k < instrShape[1]; k += 8) {
          ret.push_back({i, j + k});
          ret.push_back({i, j + k + 1});
          ret.push_back({i + 8, j + k});
          ret.push_back({i + 8, j + k + 1});
        }
      }
    }
    return ret;
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

protected:
  TritonGPUToLLVMTypeConverter *getTypeConverter() const {
    LLVMTypeConverter *ret =
        ((ConvertTritonGPUOpToLLVMPatternBase *)this)->getTypeConverter();
    return (TritonGPUToLLVMTypeConverter *)ret;
  }
};

namespace mlir::triton {
class ReduceOp;
class ScanOp;
} // namespace mlir::triton

template <typename SourceOp>
class ConvertTritonGPUReduceScanToLLVMPattern
    : public ConvertTritonGPUOpToLLVMPattern<SourceOp> {
public:
  // Make sure the class is only instantiated with Reduce and Scan
  static_assert(std::is_same_v<SourceOp, ReduceOp> ||
                std::is_same_v<SourceOp, ScanOp>);

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
        LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation());
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

#endif
