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

using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::CTALayoutAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

// typedef DenseMap<Operation *, triton::MakeTensorPtrOp> TensorPtrMapT;

// FuncOpConversion/FuncOpConversionBase is borrowed from
// https://github.com/llvm/llvm-project/blob/fae656b2dd80246c3c6f01e9c77c49560368752c/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp#L276
// since it is not exposed on header files in mlir v14
// TODO(Superjomn): remove the code when MLIR v15.0 is included.
// All the rights are reserved by the LLVM community.
// struct FuncOpConversionBase : public ConvertOpToLLVMPattern<triton::FuncOp> {
// protected:
//   /// Only retain those attributes that are not constructed by
//   /// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out
//   argument
//   /// attributes.
//   static void filterFuncAttributes(triton::FuncOp op, bool filterArgAttrs,
//                                    SmallVectorImpl<NamedAttribute> &result) {

//     for (const auto &attr : op->getAttrs()) {
//       if (attr.getName() == SymbolTable::getSymbolAttrName() ||
//           attr.getName() == op.getFunctionTypeAttrName() ||
//           attr.getName() == "std.varargs" ||
//           (filterArgAttrs && attr.getName() == op.getArgAttrsAttrName()))
//         continue;
//       result.push_back(attr);
//     }
//   }

//   /// Helper function for wrapping all attributes into a single
//   DictionaryAttr static auto wrapAsStructAttrs(OpBuilder &b, ArrayAttr attrs)
//   {
//     return DictionaryAttr::get(b.getContext(),
//                                b.getNamedAttr("llvm.struct_attrs", attrs));
//   }

// protected:
//   using ConvertOpToLLVMPattern<triton::FuncOp>::ConvertOpToLLVMPattern;

//   // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter
//   provided
//   // to this legalization pattern.
//   LLVM::LLVMFuncOp
//   convertFuncOpToLLVMFuncOp(triton::FuncOp funcOp,
//                             ConversionPatternRewriter &rewriter) const {
//     // Convert the original function arguments. They are converted using the
//     // LLVMTypeConverter provided to this legalization pattern.
//     auto varargsAttr = funcOp->getAttrOfType<BoolAttr>("func.varargs");
//     TypeConverter::SignatureConversion result(funcOp.getNumArguments());
//     auto llvmType = getTypeConverter()->convertFunctionSignature(
//         funcOp.getFunctionType(), varargsAttr && varargsAttr.getValue(),
//         false, result);
//     if (!llvmType)
//       return nullptr;

//     // Propagate argument/result attributes to all converted arguments/result
//     // obtained after converting a given original argument/result.
//     SmallVector<NamedAttribute, 4> attributes;
//     filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, attributes);
//     if (ArrayAttr resAttrDicts = funcOp.getAllResultAttrs()) {
//       assert(!resAttrDicts.empty() && "expected array to be non-empty");
//       auto newResAttrDicts =
//           (funcOp.getNumResults() == 1)
//               ? resAttrDicts
//               : rewriter.getArrayAttr(
//                     {wrapAsStructAttrs(rewriter, resAttrDicts)});
//       attributes.push_back(
//           rewriter.getNamedAttr(funcOp.getResAttrsAttrName(),
//           newResAttrDicts));
//     }
//     if (ArrayAttr argAttrDicts = funcOp.getAllArgAttrs()) {
//       SmallVector<Attribute, 4> newArgAttrs(
//           llvmType.cast<LLVM::LLVMFunctionType>().getNumParams());
//       for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
//         auto mapping = result.getInputMapping(i);
//         assert(mapping && "unexpected deletion of function argument");
//         for (size_t j = 0; j < mapping->size; ++j)
//           newArgAttrs[mapping->inputNo + j] = argAttrDicts[i];
//       }
//       attributes.push_back(rewriter.getNamedAttr(
//           funcOp.getArgAttrsAttrName(), rewriter.getArrayAttr(newArgAttrs)));
//     }
//     for (const auto &pair : llvm::enumerate(attributes)) {
//       if (pair.value().getName() == "llvm.linkage") {
//         attributes.erase(attributes.begin() + pair.index());
//         break;
//       }
//     }

//     // Create an LLVM function, use external linkage by default until MLIR
//     // functions have linkage.
//     LLVM::Linkage linkage = LLVM::Linkage::External;
//     if (auto linkageAttr = funcOp->getDiscardableAttr("llvm.linkage")) {
//       auto attr = linkageAttr.dyn_cast<mlir::LLVM::LinkageAttr>();
//       if (!attr) {
//         funcOp->emitError()
//             << "Contains llvm.linkage attribute not of type
//             LLVM::LinkageAttr";
//         return nullptr;
//       }
//       linkage = attr.getLinkage();
//     }
//     auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
//         funcOp.getLoc(), funcOp.getName(), llvmType, linkage,
//         /*dsoLocal*/ false, LLVM::CConv::C, /*comdat=*/SymbolRefAttr{},
//         attributes);
//     rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
//                                 newFuncOp.end());
//     if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(),
//     *typeConverter,
//                                            &result)))
//       return nullptr;

//     return newFuncOp;
//   }
// };

// namespace AMD {
// class ConvertTritonGPUOpToLLVMPatternBase {
// public:
//   // Two levels of value cache in emitting indices calculation:
//   // Key: {layout, shape, withCTAOffset}
//   struct IndexCacheInfo {
//     DenseMap<IndexCacheKeyT, SmallVector<Value>, CacheKeyDenseMapInfo>
//         *baseIndexCache = nullptr;
//     DenseMap<IndexCacheKeyT, SmallVector<SmallVector<Value>>,
//              CacheKeyDenseMapInfo> *indexCache = nullptr;
//     OpBuilder::InsertPoint *indexInsertPoint = nullptr;
//   };

//   explicit ConvertTritonGPUOpToLLVMPatternBase(
//       TritonGPUToLLVMTypeConverter &typeConverter)
//       : converter(&typeConverter) {}

//   explicit ConvertTritonGPUOpToLLVMPatternBase(
//       TritonGPUToLLVMTypeConverter &typeConverter,
//       IndexCacheInfo indexCacheInfo)
//       : converter(&typeConverter), indexCacheInfo(indexCacheInfo) {}

//   explicit ConvertTritonGPUOpToLLVMPatternBase(
//       TritonGPUToLLVMTypeConverter &typeConverter, ModuleAllocation
//       &allocation) : converter(&typeConverter), allocation(&allocation) {}

//   explicit ConvertTritonGPUOpToLLVMPatternBase(
//       TritonGPUToLLVMTypeConverter &typeConverter, ModuleAllocation
//       &allocation, IndexCacheInfo indexCacheInfo) :
//       converter(&typeConverter), allocation(&allocation),
//         indexCacheInfo(indexCacheInfo) {}

//   TritonGPUToLLVMTypeConverter *getTypeConverter() const { return converter;
//   }

// -----------------------------------------------------------------------
// Shared memory utilities
// -----------------------------------------------------------------------
// template <typename T>
// Value getSharedMemoryBase(Location loc, ConversionPatternRewriter
// &rewriter,
//                           T value) const {
//   auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
//   FunctionOpInterface funcOp;
//   if constexpr (std::is_pointer_v<T>)
//     funcOp = value->template getParentOfType<FunctionOpInterface>();
//   else
//     funcOp = value.getParentRegion()
//                  ->template getParentOfType<FunctionOpInterface>();
//   auto *funcAllocation = allocation->getFuncData(funcOp);
//   auto smem = allocation->getFunctionSharedMemoryBase(funcOp);
//   auto bufferId = funcAllocation->getBufferId(value);
//   assert(bufferId != Allocation::InvalidBufferId && "BufferId not found");
//   size_t offset = funcAllocation->getOffset(bufferId);
//   Value offVal = i32_val(offset);
//   Value base =
//       gep(ptrTy,
//       this->getTypeConverter()->convertType(rewriter.getI8Type()),
//           smem, offVal);
//   return base;
// }

// protected:
//   TritonGPUToLLVMTypeConverter *converter;
//   ModuleAllocation *allocation;
//   IndexCacheInfo indexCacheInfo;
// };

// template <typename SourceOp>
// class ConvertTritonGPUOpToLLVMPattern
//     : public ConvertOpToLLVMPattern<SourceOp>,
//       public ConvertTritonGPUOpToLLVMPatternBase {
// public:
//   using OpAdaptor = typename SourceOp::Adaptor;

//   explicit ConvertTritonGPUOpToLLVMPattern(
//       TritonGPUToLLVMTypeConverter &typeConverter,
//       PatternBenefit benefit = patternBenefitDefault)
//       : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
//         ConvertTritonGPUOpToLLVMPatternBase(typeConverter) {}

//   explicit ConvertTritonGPUOpToLLVMPattern(
//       TritonGPUToLLVMTypeConverter &typeConverter, ModuleAllocation
//       &allocation, PatternBenefit benefit = patternBenefitDefault) :
//       ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
//         ConvertTritonGPUOpToLLVMPatternBase(typeConverter, allocation) {}

//   explicit ConvertTritonGPUOpToLLVMPattern(
//       TritonGPUToLLVMTypeConverter &typeConverter,
//       IndexCacheInfo indexCacheInfo,
//       PatternBenefit benefit = patternBenefitDefault)
//       : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
//         ConvertTritonGPUOpToLLVMPatternBase(typeConverter, indexCacheInfo) {}

//   explicit ConvertTritonGPUOpToLLVMPattern(
//       TritonGPUToLLVMTypeConverter &typeConverter, ModuleAllocation
//       &allocation, IndexCacheInfo indexCacheInfo, PatternBenefit benefit =
//       patternBenefitDefault) :
//       ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
//         ConvertTritonGPUOpToLLVMPatternBase(typeConverter, allocation,
//                                             indexCacheInfo) {}

// protected:
//   TritonGPUToLLVMTypeConverter *getTypeConverter() const {
//     LLVMTypeConverter *ret =
//         ((ConvertTritonGPUOpToLLVMPatternBase *)this)->getTypeConverter();
//     return (TritonGPUToLLVMTypeConverter *)ret;
//   }
// };

// } // namespace AMD
#endif
