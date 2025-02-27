#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/Passes.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/PatternProtonOpToLLVM.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/Utility.h"
#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "third_party/proton/dialect/include/Dialect/ProtonGPU/IR/Dialect.h"

#include "llvm/Support/FormatVariadic.h"



using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton::proton {
#define GEN_PASS_DEF_ADDPROTONKERNELARG
#include "proton/dialect/include/Conversion/ProtonGPUToLLVM/Passes.h.inc"
} // namespace triton::proton
} // namespace mlir

namespace {
// The following convertFuncOpToLLVMFuncOp is copied and modified from
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp
static constexpr StringRef varargsAttrName = "func.varargs";
static constexpr StringRef linkageAttrName = "llvm.linkage";
static constexpr StringRef barePtrAttrName = "llvm.bareptr";

/// Return `true` if the `op` should use bare pointer calling convention.
static bool shouldUseBarePtrCallConv(Operation *op,
                                     const LLVMTypeConverter *typeConverter) {
  return (op && op->hasAttr(barePtrAttrName)) ||
         typeConverter->getOptions().useBarePtrCallConv;
}

/// Only retain those attributes that are not constructed by
/// `LLVMFuncOp::build`.
static void filterFuncAttributes(FunctionOpInterface func,
                                 SmallVectorImpl<NamedAttribute> &result) {
  for (const NamedAttribute &attr : func->getDiscardableAttrs()) {
    if (attr.getName() == linkageAttrName ||
        attr.getName() == varargsAttrName ||
        attr.getName() == LLVM::LLVMDialect::getReadnoneAttrName())
      continue;
    result.push_back(attr);
  }
}

/// Propagate argument/results attributes.
static void propagateArgResAttrs(OpBuilder &builder, bool resultStructType,
                                 FunctionOpInterface funcOp,
                                 LLVM::LLVMFuncOp wrapperFuncOp) {
  auto argAttrs = funcOp.getAllArgAttrs();
  if (!resultStructType) {
    if (auto resAttrs = funcOp.getAllResultAttrs())
      wrapperFuncOp.setAllResultAttrs(resAttrs);
    if (argAttrs)
      wrapperFuncOp.setAllArgAttrs(argAttrs);
  } else {
    SmallVector<Attribute> argAttributes;
    // Only modify the argument and result attributes when the result is now
    // an argument.
    if (argAttrs) {
      argAttributes.push_back(builder.getDictionaryAttr({}));
      argAttributes.append(argAttrs.begin(), argAttrs.end());
      wrapperFuncOp.setAllArgAttrs(argAttributes);
    }
  }
  cast<FunctionOpInterface>(wrapperFuncOp.getOperation())
      .setVisibility(funcOp.getVisibility());
}

/// Creates an auxiliary function with pointer-to-memref-descriptor-struct
/// arguments instead of unpacked arguments. This function can be called from C
/// by passing a pointer to a C struct corresponding to a memref descriptor.
/// Similarly, returned memrefs are passed via pointers to a C struct that is
/// passed as additional argument.
/// Internally, the auxiliary function unpacks the descriptor into individual
/// components and forwards them to `newFuncOp` and forwards the results to
/// the extra arguments.
static void wrapForExternalCallers(OpBuilder &rewriter, Location loc,
                                   const LLVMTypeConverter &typeConverter,
                                   FunctionOpInterface funcOp,
                                   LLVM::LLVMFuncOp newFuncOp) {
  auto type = cast<FunctionType>(funcOp.getFunctionType());
  auto [wrapperFuncType, resultStructType] =
      typeConverter.convertFunctionTypeCWrapper(type);

  SmallVector<NamedAttribute> attributes;
  filterFuncAttributes(funcOp, attributes);

  auto wrapperFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
      loc, llvm::formatv("_mlir_ciface_{0}", funcOp.getName()).str(),
      wrapperFuncType, LLVM::Linkage::External, /*dsoLocal=*/false,
      /*cconv=*/LLVM::CConv::C, /*comdat=*/nullptr, attributes);
  propagateArgResAttrs(rewriter, !!resultStructType, funcOp, wrapperFuncOp);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(wrapperFuncOp.addEntryBlock(rewriter));

  SmallVector<Value, 8> args;
  size_t argOffset = resultStructType ? 1 : 0;
  for (auto [index, argType] : llvm::enumerate(type.getInputs())) {
    Value arg = wrapperFuncOp.getArgument(index + argOffset);
    if (auto memrefType = dyn_cast<MemRefType>(argType)) {
      Value loaded = rewriter.create<LLVM::LoadOp>(
          loc, typeConverter.convertType(memrefType), arg);
      MemRefDescriptor::unpack(rewriter, loc, loaded, memrefType, args);
      continue;
    }
    if (isa<UnrankedMemRefType>(argType)) {
      Value loaded = rewriter.create<LLVM::LoadOp>(
          loc, typeConverter.convertType(argType), arg);
      UnrankedMemRefDescriptor::unpack(rewriter, loc, loaded, args);
      continue;
    }

    args.push_back(arg);
  }

  auto call = rewriter.create<LLVM::CallOp>(loc, newFuncOp, args);

  if (resultStructType) {
    rewriter.create<LLVM::StoreOp>(loc, call.getResult(),
                                   wrapperFuncOp.getArgument(0));
    rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});
  } else {
    rewriter.create<LLVM::ReturnOp>(loc, call.getResults());
  }
}

/// Creates an auxiliary function with pointer-to-memref-descriptor-struct
/// arguments instead of unpacked arguments. Creates a body for the (external)
/// `newFuncOp` that allocates a memref descriptor on stack, packs the
/// individual arguments into this descriptor and passes a pointer to it into
/// the auxiliary function. If the result of the function cannot be directly
/// returned, we write it to a special first argument that provides a pointer
/// to a corresponding struct. This auxiliary external function is now
/// compatible with functions defined in C using pointers to C structs
/// corresponding to a memref descriptor.
static void wrapExternalFunction(OpBuilder &builder, Location loc,
                                 const LLVMTypeConverter &typeConverter,
                                 FunctionOpInterface funcOp,
                                 LLVM::LLVMFuncOp newFuncOp) {
  OpBuilder::InsertionGuard guard(builder);

  auto [wrapperType, resultStructType] =
      typeConverter.convertFunctionTypeCWrapper(
          cast<FunctionType>(funcOp.getFunctionType()));
  // This conversion can only fail if it could not convert one of the argument
  // types. But since it has been applied to a non-wrapper function before, it
  // should have failed earlier and not reach this point at all.
  assert(wrapperType && "unexpected type conversion failure");

  SmallVector<NamedAttribute, 4> attributes;
  filterFuncAttributes(funcOp, attributes);

  // Create the auxiliary function.
  auto wrapperFunc = builder.create<LLVM::LLVMFuncOp>(
      loc, llvm::formatv("_mlir_ciface_{0}", funcOp.getName()).str(),
      wrapperType, LLVM::Linkage::External, /*dsoLocal=*/false,
      /*cconv=*/LLVM::CConv::C, /*comdat=*/nullptr, attributes);
  propagateArgResAttrs(builder, !!resultStructType, funcOp, wrapperFunc);

  // The wrapper that we synthetize here should only be visible in this module.
  newFuncOp.setLinkage(LLVM::Linkage::Private);
  builder.setInsertionPointToStart(newFuncOp.addEntryBlock(builder));

  // Get a ValueRange containing arguments.
  FunctionType type = cast<FunctionType>(funcOp.getFunctionType());
  SmallVector<Value, 8> args;
  args.reserve(type.getNumInputs());
  ValueRange wrapperArgsRange(newFuncOp.getArguments());

  if (resultStructType) {
    // Allocate the struct on the stack and pass the pointer.
    Type resultType = cast<LLVM::LLVMFunctionType>(wrapperType).getParamType(0);
    Value one = builder.create<LLVM::ConstantOp>(
        loc, typeConverter.convertType(builder.getIndexType()),
        builder.getIntegerAttr(builder.getIndexType(), 1));
    Value result =
        builder.create<LLVM::AllocaOp>(loc, resultType, resultStructType, one);
    args.push_back(result);
  }

  // Iterate over the inputs of the original function and pack values into
  // memref descriptors if the original type is a memref.
  for (Type input : type.getInputs()) {
    Value arg;
    int numToDrop = 1;
    auto memRefType = dyn_cast<MemRefType>(input);
    auto unrankedMemRefType = dyn_cast<UnrankedMemRefType>(input);
    if (memRefType || unrankedMemRefType) {
      numToDrop = memRefType
                      ? MemRefDescriptor::getNumUnpackedValues(memRefType)
                      : UnrankedMemRefDescriptor::getNumUnpackedValues();
      Value packed =
          memRefType
              ? MemRefDescriptor::pack(builder, loc, typeConverter, memRefType,
                                       wrapperArgsRange.take_front(numToDrop))
              : UnrankedMemRefDescriptor::pack(
                    builder, loc, typeConverter, unrankedMemRefType,
                    wrapperArgsRange.take_front(numToDrop));

      auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
      Value one = builder.create<LLVM::ConstantOp>(
          loc, typeConverter.convertType(builder.getIndexType()),
          builder.getIntegerAttr(builder.getIndexType(), 1));
      Value allocated = builder.create<LLVM::AllocaOp>(
          loc, ptrTy, packed.getType(), one, /*alignment=*/0);
      builder.create<LLVM::StoreOp>(loc, packed, allocated);
      arg = allocated;
    } else {
      arg = wrapperArgsRange[0];
    }

    args.push_back(arg);
    wrapperArgsRange = wrapperArgsRange.drop_front(numToDrop);
  }
  assert(wrapperArgsRange.empty() && "did not map some of the arguments");

  auto call = builder.create<LLVM::CallOp>(loc, wrapperFunc, args);

  if (resultStructType) {
    Value result =
        builder.create<LLVM::LoadOp>(loc, resultStructType, args.front());
    builder.create<LLVM::ReturnOp>(loc, result);
  } else {
    builder.create<LLVM::ReturnOp>(loc, call.getResults());
  }
}

/// Inserts `llvm.load` ops in the function body to restore the expected pointee
/// value from `llvm.byval`/`llvm.byref` function arguments that were converted
/// to LLVM pointer types.
static void restoreByValRefArgumentType(
    ConversionPatternRewriter &rewriter, const LLVMTypeConverter &typeConverter,
    ArrayRef<std::optional<NamedAttribute>> byValRefNonPtrAttrs,
    ArrayRef<BlockArgument> oldBlockArgs, LLVM::LLVMFuncOp funcOp) {
  // Nothing to do for function declarations.
  if (funcOp.isExternal())
    return;

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());

  for (const auto &[arg, oldArg, byValRefAttr] :
       llvm::zip(funcOp.getArguments(), oldBlockArgs, byValRefNonPtrAttrs)) {
    // Skip argument if no `llvm.byval` or `llvm.byref` attribute.
    if (!byValRefAttr)
      continue;

    // Insert load to retrieve the actual argument passed by value/reference.
    assert(isa<LLVM::LLVMPointerType>(arg.getType()) &&
           "Expected LLVM pointer type for argument with "
           "`llvm.byval`/`llvm.byref` attribute");
    Type resTy = typeConverter.convertType(
        cast<TypeAttr>(byValRefAttr->getValue()).getValue());

    auto valueArg = rewriter.create<LLVM::LoadOp>(arg.getLoc(), resTy, arg);
    rewriter.replaceUsesOfBlockArgument(oldArg, valueArg);
  }
}	

FailureOr<LLVM::LLVMFuncOp>
convertFuncOpToLLVMFuncOp(FunctionOpInterface funcOp,
                                ConversionPatternRewriter &rewriter,
                                const LLVMTypeConverter &converter) {
  // Check the funcOp has `FunctionType`.
  auto funcTy = dyn_cast<FunctionType>(funcOp.getFunctionType());
  if (!funcTy)
    return rewriter.notifyMatchFailure(
        funcOp, "Only support FunctionOpInterface with FunctionType");

  // Keep track of the entry block arguments. They will be needed later.
  SmallVector<BlockArgument> oldBlockArgs =
      llvm::to_vector(funcOp.getArguments());

  // Convert the original function arguments. They are converted using the
  // LLVMTypeConverter provided to this legalization pattern.
  auto varargsAttr = funcOp->getAttrOfType<BoolAttr>(varargsAttrName);
  // Gather `llvm.byval` and `llvm.byref` arguments whose type convertion was
  // overriden with an LLVM pointer type for later processing.
  SmallVector<std::optional<NamedAttribute>> byValRefNonPtrAttrs;
  TypeConverter::SignatureConversion result(funcOp.getNumArguments());
  auto llvmType = converter.convertFunctionSignature(
      funcOp, varargsAttr && varargsAttr.getValue(),
      shouldUseBarePtrCallConv(funcOp, &converter), result,
      byValRefNonPtrAttrs);
  if (!llvmType)
    return rewriter.notifyMatchFailure(funcOp, "signature conversion failed");

  // Create an LLVM function, use external linkage by default until MLIR
  // functions have linkage.
  LLVM::Linkage linkage = LLVM::Linkage::External;
  if (funcOp->hasAttr(linkageAttrName)) {
    auto attr =
        dyn_cast<mlir::LLVM::LinkageAttr>(funcOp->getAttr(linkageAttrName));
    if (!attr) {
      funcOp->emitError() << "Contains " << linkageAttrName
                          << " attribute not of type LLVM::LinkageAttr";
      return rewriter.notifyMatchFailure(
          funcOp, "Contains linkage attribute not of type LLVM::LinkageAttr");
    }
    linkage = attr.getLinkage();
  }

  SmallVector<NamedAttribute, 4> attributes;
  filterFuncAttributes(funcOp, attributes);
  auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
      funcOp.getLoc(), funcOp.getName(), llvmType, linkage,
      /*dsoLocal=*/false, /*cconv=*/LLVM::CConv::C, /*comdat=*/nullptr,
      attributes);
  cast<FunctionOpInterface>(newFuncOp.getOperation())
      .setVisibility(funcOp.getVisibility());

  // Create a memory effect attribute corresponding to readnone.
  StringRef readnoneAttrName = LLVM::LLVMDialect::getReadnoneAttrName();
  if (funcOp->hasAttr(readnoneAttrName)) {
    auto attr = funcOp->getAttrOfType<UnitAttr>(readnoneAttrName);
    if (!attr) {
      funcOp->emitError() << "Contains " << readnoneAttrName
                          << " attribute not of type UnitAttr";
      return rewriter.notifyMatchFailure(
          funcOp, "Contains readnone attribute not of type UnitAttr");
    }
    auto memoryAttr = LLVM::MemoryEffectsAttr::get(
        rewriter.getContext(),
        {LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::NoModRef,
         LLVM::ModRefInfo::NoModRef});
    newFuncOp.setMemoryEffectsAttr(memoryAttr);
  }

  // Propagate argument/result attributes to all converted arguments/result
  // obtained after converting a given original argument/result.
  if (ArrayAttr resAttrDicts = funcOp.getAllResultAttrs()) {
    assert(!resAttrDicts.empty() && "expected array to be non-empty");
    if (funcOp.getNumResults() == 1)
      newFuncOp.setAllResultAttrs(resAttrDicts);
  }
  if (ArrayAttr argAttrDicts = funcOp.getAllArgAttrs()) {
    SmallVector<Attribute> newArgAttrs(
        cast<LLVM::LLVMFunctionType>(llvmType).getNumParams());
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
      // Some LLVM IR attribute have a type attached to them. During FuncOp ->
      // LLVMFuncOp conversion these types may have changed. Account for that
      // change by converting attributes' types as well.
      SmallVector<NamedAttribute, 4> convertedAttrs;
      auto attrsDict = cast<DictionaryAttr>(argAttrDicts[i]);
      convertedAttrs.reserve(attrsDict.size());
      for (const NamedAttribute &attr : attrsDict) {
        const auto convert = [&](const NamedAttribute &attr) {
          return TypeAttr::get(converter.convertType(
              cast<TypeAttr>(attr.getValue()).getValue()));
        };
        if (attr.getName().getValue() ==
            LLVM::LLVMDialect::getByValAttrName()) {
          convertedAttrs.push_back(rewriter.getNamedAttr(
              LLVM::LLVMDialect::getByValAttrName(), convert(attr)));
        } else if (attr.getName().getValue() ==
                   LLVM::LLVMDialect::getByRefAttrName()) {
          convertedAttrs.push_back(rewriter.getNamedAttr(
              LLVM::LLVMDialect::getByRefAttrName(), convert(attr)));
        } else if (attr.getName().getValue() ==
                   LLVM::LLVMDialect::getStructRetAttrName()) {
          convertedAttrs.push_back(rewriter.getNamedAttr(
              LLVM::LLVMDialect::getStructRetAttrName(), convert(attr)));
        } else if (attr.getName().getValue() ==
                   LLVM::LLVMDialect::getInAllocaAttrName()) {
          convertedAttrs.push_back(rewriter.getNamedAttr(
              LLVM::LLVMDialect::getInAllocaAttrName(), convert(attr)));
        } else {
          convertedAttrs.push_back(attr);
        }
      }
      auto mapping = result.getInputMapping(i);
      assert(mapping && "unexpected deletion of function argument");
      // Only attach the new argument attributes if there is a one-to-one
      // mapping from old to new types. Otherwise, attributes might be
      // attached to types that they do not support.
      if (mapping->size == 1) {
        newArgAttrs[mapping->inputNo] =
            DictionaryAttr::get(rewriter.getContext(), convertedAttrs);
        continue;
      }
      // TODO: Implement custom handling for types that expand to multiple
      // function arguments.
      for (size_t j = 0; j < mapping->size; ++j)
        newArgAttrs[mapping->inputNo + j] =
            DictionaryAttr::get(rewriter.getContext(), {});
    }
    if (!newArgAttrs.empty())
      newFuncOp.setAllArgAttrs(rewriter.getArrayAttr(newArgAttrs));
  }

  rewriter.inlineRegionBefore(funcOp.getFunctionBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  // Convert just the entry block. The remaining unstructured control flow is
  // converted by ControlFlowToLLVM.
  if (!newFuncOp.getBody().empty())
    rewriter.applySignatureConversion(&newFuncOp.getBody().front(), result,
                                      &converter);

  // Fix the type mismatch between the materialized `llvm.ptr` and the expected
  // pointee type in the function body when converting `llvm.byval`/`llvm.byref`
  // function arguments.
  restoreByValRefArgumentType(rewriter, converter, byValRefNonPtrAttrs,
                              oldBlockArgs, newFuncOp);

  if (!shouldUseBarePtrCallConv(funcOp, &converter)) {
    if (funcOp->getAttrOfType<UnitAttr>(
            LLVM::LLVMDialect::getEmitCWrapperAttrName())) {
      if (newFuncOp.isVarArg())
        return funcOp.emitError("C interface for variadic functions is not "
                                "supported yet.");

      if (newFuncOp.isExternal())
        wrapExternalFunction(rewriter, funcOp->getLoc(), converter, funcOp,
                             newFuncOp);
      else
        wrapForExternalCallers(rewriter, funcOp->getLoc(), converter, funcOp,
                               newFuncOp);
    }
  }

  return newFuncOp;
}

struct AddProtonKernelArg
    : public mlir::triton::proton::impl::AddProtonKernelArgBase<
          AddProtonKernelArg> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    bool hasProtonGlobalScratchAllocOp = false;
    moduleOp.walk([&](LLVM::LLVMFuncOp funcOp) {
      funcOp.walk([&](proton::gpu::GlobalScratchAllocOp op) {
        hasProtonGlobalScratchAllocOp = true;
        auto funcTy = funcOp.getFunctionType();		    
	llvm::errs() << funcTy << "\n";
      });
    });    

    if (!hasProtonGlobalScratchAllocOp)
      return;
    IRRewriter rewriter(ctx);

    assert(llvm::range_size(moduleOp.getOps<LLVM::LLVMFuncOp>()) == 1);
    LLVM::LLVMFuncOp funcOp = *moduleOp.getOps<LLVM::LLVMFuncOp>().begin();

  }
};

} // namespace

namespace mlir {

namespace triton::proton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>>
createAddProtonKernelArgPass() {
  return std::make_unique<AddProtonKernelArg>();
}

} // namespace gpu

} // namespace triton::proton

} // namespace mlir    
