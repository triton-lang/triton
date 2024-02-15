#include "triton/Conversion/TritonGPUToLLVM/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#ifndef USE_ROCM
#else
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#endif
#include "triton/Tools/Sys/GetPlatform.hpp"

#include "BarrierOpToLLVM.h"
// #include "ClusterOpsToLLVM.h"
#include "ConvertLayoutOpToLLVM.h"
#include "DotOpToLLVM.h"
#include "ElementwiseOpToLLVM.h"
#include "LoadStoreOpToLLVM.h"
#include "ReduceOpToLLVM.h"
// #include "RegReallocOpToLLVM.h"
#include "ScanOpToLLVM.h"
// #include "TensorPtrOpsToLLVM.h"
#include "TritonGPUToLLVM.h"
#include "TritonGPUToLLVMBase.h"
#include "TypeConverter.h"
#include "ViewOpToLLVM.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONGPUTOLLVM
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
namespace ttng = mlir::triton::nvidia_gpu;
using ::AMD::TritonGPUToLLVMTypeConverter;
using ::AMD::ConvertTritonGPUOpToLLVMPatternBase;
using ::AMD::ConvertTritonGPUOpToLLVMPattern;

namespace {

// pass ws related named attrs.
static void addWSNamedAttrs(Operation *op,
                            ArrayRef<mlir::NamedAttribute> attrs) {
  for (const NamedAttribute attr : attrs)
    if (attr.getName() == "async_agent" || attr.getName() == "agent.mutex_role")
      op->setAttr(attr.getName(), attr.getValue());
}

#ifdef USE_ROCM
constexpr int LDSSize = 65536;
constexpr int kPtrBitWidth = 64;
#endif
class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx, Target target)
      : ConversionTarget(ctx) {
    addLegalDialect<index::IndexDialect>();
    addLegalDialect<LLVM::LLVMDialect>();
    switch (target) {
    case Target::NVVM:
      addLegalDialect<NVVM::NVVMDialect>();
      break;
    case Target::ROCDL:
      addLegalDialect<ROCDL::ROCDLDialect>();
      addLegalDialect<mlir::scf::SCFDialect>();
      break;
    }
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ReturnOpConversion : public ConvertOpToLLVMPattern<triton::ReturnOp> {
  using ConvertOpToLLVMPattern<triton::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (funcOp->hasAttr("nvvm.kernel")) {
      // A GPU kernel
      if (op.getNumOperands() > 0) {
        return rewriter.notifyMatchFailure(
            op, "Kernel functions do not support return with operands");
      }
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), ValueRange(),
                                                  op->getAttrs());
    } else {
      // A device function
      LLVM::ReturnOp newOp;
      if (adaptor.getOperands().size() < 2) {
        // Single or no return value.
        newOp =
            rewriter.create<LLVM::ReturnOp>(op.getLoc(), adaptor.getOperands());
      } else {
        // Pack the results into a struct.
        auto packedResultsTy = this->getTypeConverter()->packFunctionResults(
            funcOp.getResultTypes());
        Value packedResults =
            rewriter.create<LLVM::UndefOp>(op.getLoc(), packedResultsTy);
        auto loc = op.getLoc();
        for (auto it : llvm::enumerate(adaptor.getOperands())) {
          packedResults = insert_val(packedResultsTy, packedResults, it.value(),
                                     it.index());
        }
        newOp = rewriter.create<LLVM::ReturnOp>(op.getLoc(), packedResults);
      }
      newOp->setAttrs(op->getAttrs());
      rewriter.replaceOp(op, newOp->getResults());
    }
    return success();
  }
};

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncOpConversion : public FuncOpConversionBase {
  FuncOpConversion(LLVMTypeConverter &converter, int numWarps,
                   ModuleAllocation &allocation, PatternBenefit benefit)
      : FuncOpConversionBase(converter, benefit), numWarps(numWarps),
        allocation(allocation) {}

  triton::FuncOp amendFuncOp(triton::FuncOp funcOp,
                             ConversionPatternRewriter &rewriter) const {
    // Push back a variable that indicates the current stack pointer of shared
    // memory to the function arguments.
    auto loc = funcOp.getLoc();
    auto ctx = funcOp->getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
    // 1. Modify the function type to add the new argument.
    auto funcTy = funcOp.getFunctionType();
    auto amendedInputTy = llvm::to_vector<4>(funcTy.getInputs());
    amendedInputTy.push_back(ptrTy);
    auto amendedFuncTy = FunctionType::get(funcTy.getContext(), amendedInputTy,
                                           funcTy.getResults());
    // 2. Modify the argument attributes to add the new argument.
    SmallVector<NamedAttribute> amendedAttrs;
    filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, amendedAttrs);
    auto amendedArgAttrs = llvm::to_vector<4>(funcOp.getAllArgAttrs());
    amendedArgAttrs.emplace_back(DictionaryAttr::get(ctx));
    amendedAttrs.push_back(rewriter.getNamedAttr(
        funcOp.getArgAttrsAttrName(), rewriter.getArrayAttr(amendedArgAttrs)));
    // 3. Add a new argument to the region
    auto amendedFuncOp = rewriter.create<triton::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), amendedFuncTy, amendedAttrs);
    auto &region = funcOp.getBody();
    region.addArgument(ptrTy, loc);
    rewriter.inlineRegionBefore(region, amendedFuncOp.getBody(),
                                amendedFuncOp.end());
    return amendedFuncOp;
  }

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Prevent LLVM's inliner to inline this function
    auto amendedFuncOp = funcOp;
    if (!allocation.isRoot(funcOp))
      amendedFuncOp = amendFuncOp(funcOp, rewriter);

    auto newFuncOp = convertFuncOpToLLVMFuncOp(amendedFuncOp, rewriter);
    if (!newFuncOp) {
      return failure();
    }

    auto ctx = funcOp->getContext();

    if (allocation.isRoot(funcOp)) {
      // Set an attribute to indicate this function is a kernel entry.
      newFuncOp->setAttr("nvvm.kernel",
                         rewriter.getIntegerAttr(type::u1Ty(ctx), 1));
    } else {
      // The noinline attribute will be used by the LLVM codegen to prevent
      // inlining.
      // https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/LLVMIR/IR/LLVMInlining.cpp#L267
      newFuncOp.setPassthroughAttr(
          ArrayAttr::get(ctx, rewriter.getStringAttr("noinline")));
      rewriter.eraseOp(amendedFuncOp);
    }
#ifndef USE_ROCM
    // Set an attribute for maxntidx, it could be used in latter LLVM codegen
    // for `nvvm.annotation` metadata.
    newFuncOp->setAttr("nvvm.maxntid", rewriter.getI32ArrayAttr(32 * numWarps));
#endif
    // The call graph is updated by mapping the old function to the new one.
    allocation.mapFuncOp(funcOp, newFuncOp);

    auto funcTy = newFuncOp.getFunctionType().cast<LLVM::LLVMFunctionType>();
    SmallVector<Type> newInputsTy(funcTy.getParams().begin(),
                                  funcTy.getParams().end());
    newFuncOp.setType(
        LLVM::LLVMFunctionType::get(funcTy.getReturnType(), newInputsTy));
    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  int numWarps{0};
  ModuleAllocation &allocation;
};

// CallOpInterfaceLowering is adapted from
// https://github.com/llvm/llvm-project/blob/fae656b2dd80246c3c6f01e9c77c49560368752c/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp#L485
struct CallOpConversion : public ConvertOpToLLVMPattern<triton::CallOp> {
  CallOpConversion(LLVMTypeConverter &converter, int numWarps,
                   ModuleAllocation &allocation, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::CallOp>(converter, benefit),
        numWarps(numWarps), allocation(allocation) {}

  LogicalResult
  matchAndRewrite(triton::CallOp callOp,
                  typename triton::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto promotedOperands = promoteOperands(callOp, adaptor, rewriter);
    auto newCallOp =
        convertCallOpToLLVMCallOp(callOp, promotedOperands, rewriter);
    if (!newCallOp)
      return failure();
    allocation.mapCallOp(callOp, newCallOp);
    auto results = getCallOpResults(callOp, newCallOp, rewriter);
    rewriter.replaceOp(callOp, results);
    return success();
  }

private:
  SmallVector<Value, 4>
  promoteOperands(triton::CallOp callOp,
                  typename triton::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    // Get the last argument of the caller, which is the current stack pointer
    // of shared memory and append it to the operands of the callOp.
    auto loc = callOp.getLoc();
    auto caller = callOp->getParentOfType<FunctionOpInterface>();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                            NVVM::kSharedMemorySpace);
    auto promotedOperands = this->getTypeConverter()->promoteOperands(
        callOp.getLoc(), /*opOperands=*/callOp->getOperands(),
        adaptor.getOperands(), rewriter);
    auto base = allocation.getFunctionSharedMemoryBase(caller);
    auto *funcAllocation = allocation.getFuncData(caller);
    auto bufferId = funcAllocation->getBufferId(callOp);
    // function doesn't have a shared mem buffer
    if (bufferId == (size_t)-1) {
      promotedOperands.push_back(base);
      return promotedOperands;
    }
    // function has a shared mem buffer
    auto offset = funcAllocation->getOffset(bufferId);
    auto offsetValue =
        gep(ptrTy, this->getTypeConverter()->convertType(rewriter.getI8Type()),
            base, i32_val(offset));
    promotedOperands.push_back(offsetValue);
    return promotedOperands;
  }

  LLVM::CallOp
  convertCallOpToLLVMCallOp(triton::CallOp callOp,
                            ArrayRef<Value> promotedOperands,
                            ConversionPatternRewriter &rewriter) const {
    // Pack the result types into a struct.
    Type packedResult = nullptr;
    unsigned numResults = callOp.getNumResults();
    auto resultTypes = llvm::to_vector<4>(callOp.getResultTypes());

    if (numResults != 0) {
      if (!(packedResult =
                this->getTypeConverter()->packFunctionResults(resultTypes)))
        return nullptr;
    }
    auto newCallOp = rewriter.create<LLVM::CallOp>(
        callOp.getLoc(), packedResult ? TypeRange(packedResult) : TypeRange(),
        promotedOperands, callOp->getAttrs());
    return newCallOp;
  }

  SmallVector<Value>
  getCallOpResults(triton::CallOp callOp, LLVM::CallOp newCallOp,
                   ConversionPatternRewriter &rewriter) const {
    auto numResults = callOp.getNumResults();
    SmallVector<Value> results;
    if (numResults < 2) {
      // If < 2 results, packing did not do anything and we can just return.
      results.append(newCallOp.result_begin(), newCallOp.result_end());
    } else {
      // Otherwise, it had been converted to an operation producing a structure.
      // Extract individual results from the structure and return them as list.
      results.reserve(numResults);
      for (unsigned i = 0; i < numResults; ++i) {
        results.push_back(rewriter.create<LLVM::ExtractValueOp>(
            callOp.getLoc(), newCallOp->getResult(0), i));
      }
    }
    return results;
  }

  int numWarps{0};
  ModuleAllocation &allocation;
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx, Target target)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    switch (target) {
    case Target::NVVM:
      addLegalDialect<NVVM::NVVMDialect>();
      break;
    case Target::ROCDL:
      addLegalDialect<ROCDL::ROCDLDialect>();
      addLegalDialect<mlir::scf::SCFDialect>();
      break;
    }
    addLegalDialect<mlir::triton::nvgpu::NVGPUDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertTritonGPUToLLVM
    : public triton::impl::ConvertTritonGPUToLLVMBase<ConvertTritonGPUToLLVM> {
  using ConvertTritonGPUToLLVMBase<
      ConvertTritonGPUToLLVM>::ConvertTritonGPUToLLVMBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::nvgpu::NVGPUDialect, LLVM::LLVMDialect,
                    NVVM::NVVMDialect>();
  }

  ConvertTritonGPUToLLVM(int32_t computeCapability, Target target)
      : ConvertTritonGPUToLLVMBase({computeCapability, target}) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context, target);
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    // Hack: WSMaterialization may have changed the effective number of warps,
    // in a way that isn't reflected in triton_gpu.num-warps.  If so, we have to
    // respect that here.
    if (Attribute attr = mod->getAttr("triton_gpu.num-warp-groups-per-cta")) {
      numWarps *= attr.cast<IntegerAttr>().getInt();
    }

    // Preprocess
    decomposeFp8e4b15Convert(mod);
    decomposeSplatToSharedLayout(mod, numWarps, threadsPerWarp, numCTAs);
    decomposeMmaToDotOperand(mod, numWarps, threadsPerWarp, numCTAs);
#ifdef USE_ROCM
    decomposeMfmaToDotOperand(mod, numWarps, threadsPerWarp, numCTAs);
    reduceCvtOpLDSUsage(mod);
#endif
    decomposeBlockedToDotOperand(mod);
    decomposeInsertSliceAsyncOp(mod);
    decomposeMixedModeDotOp(mod);

    // Allocate shared memory and set barrier
    ModuleAllocation allocation(mod);
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    // Lower functions
    {
      mlir::LowerToLLVMOptions option(context);
      TritonGPUToLLVMTypeConverter typeConverter(context, option);
      TritonLLVMFunctionConversionTarget funcTarget(*context, target);
      RewritePatternSet funcPatterns(context);
      funcPatterns.add<FuncOpConversion>(typeConverter, numWarps, allocation,
                                         /*benefit=*/1);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of each
    // function
    initSharedMemory(allocation, typeConverter);

    // Convert call and ret ops
    {
      mlir::LowerToLLVMOptions option(context);
      TritonGPUToLLVMTypeConverter typeConverter(context, option);
      TritonLLVMFunctionConversionTarget funcTarget(*context, target);
      RewritePatternSet funcPatterns(context);
      funcPatterns.add<CallOpConversion>(typeConverter, numWarps, allocation,
                                         /*benefit=*/1);
      funcPatterns.add<ReturnOpConversion>(typeConverter, /*benefit=*/1);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    // Emit logics to get threadId/blockIds/linearized clusterCTAId etc. and
    // cache the values. The reason to do it here is that cluster_ctaid is
    // currently implemented via inline asm, and thus cannot be CSEed.
    // clusterCTAId will be emitted only when numCTAs is larger than 1, and
    // other values will be DCEed if not used hereafter.
    OpBuilder::InsertPoint indexInsertPoint;
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo indexCacheInfo{
        &baseIndexCache, &indexCache, &indexInsertPoint};
    // TODO: enable index cache if there are multiple functions
    if (axisInfoAnalysis.getNumFunctions() > 1) {
      indexCacheInfo = {nullptr, nullptr, nullptr};
    }

    // {
    //   RewritePatternSet patterns(context);
    //   populateTritonGPUToLLVMPatterns(typeConverter, patterns, numWarps, axisInfoAnalysis, allocation, indexCacheInfo, 10);
    //   mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    //   mlir::populateGpuToROCDLConversionPatterns(typeConverter, patterns,
    //                                              mlir::gpu::amd::HIP);
    //   populatePatterns3(populateLoadStoreOpToLLVMPatterns);
    //   if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))){
    //     llvm::outs() << "fail1!\n";
    //     return signalPassFailure();
    //   }
    // }
    RewritePatternSet patterns(context);

    auto populatePatterns1 = [&](auto populateFunc) {
      populateFunc(typeConverter, patterns, numWarps, axisInfoAnalysis,
                   allocation, indexCacheInfo,
                   /*benefit*/ 10);
    };

    auto populatePatterns2 = [&](auto populateFunc) {
      populateFunc(typeConverter, patterns, numWarps, axisInfoAnalysis,
                   allocation, /*benefit*/ 10);
    };

    auto populatePatterns3 = [&](auto populateFunc) {
      populateFunc(typeConverter, patterns, numWarps, axisInfoAnalysis,
                   allocation, indexCacheInfo, /*benefit*/ 10);
    };

    auto populatePatterns4 = [&](auto populateFunc) {
      populateFunc(typeConverter, patterns, numWarps, axisInfoAnalysis,
                   allocation, indexCacheInfo, computeCapability,
                   /*benefit*/ 10);
    };

    populatePatterns1(AMD::populateTritonGPUToLLVMPatterns);
    populatePatterns1(AMD::populateConvertLayoutOpToLLVMPatterns);
    populatePatterns2(AMD::populateDotOpToLLVMPatterns);
    populatePatterns4(AMD::populateElementwiseOpToLLVMPatterns);
    populatePatterns3(AMD::populateLoadStoreOpToLLVMPatterns);
    populatePatterns4(AMD::populateReduceOpToLLVMPatterns);
    populatePatterns1(AMD::populateScanOpToLLVMPatterns);
    populatePatterns2(AMD::populateViewOpToLLVMPatterns);
    populatePatterns2(AMD::populateBarrierOpToLLVMPatterns);

    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);

    // Native lowering patterns
    switch (target) {
    case Target::NVVM:
      mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);
      break;
    case Target::ROCDL:
      mlir::populateGpuToROCDLConversionPatterns(typeConverter, patterns,
                                                 mlir::gpu::amd::HIP);
      break;
    }

    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))){
      return signalPassFailure();
    }

    // Fold CTAId when there is only 1 CTA.
    if (numCTAs == 1) {
      mod.walk([](triton::nvgpu::ClusterCTAIdOp id) {
        OpBuilder b(id);
        Value zero = LLVM::createConstantI32(id->getLoc(), b, 0);
        id.replaceAllUsesWith(zero);
      });
    }
  }

private:
  DenseMap<IndexCacheKeyT, SmallVector<Value>, CacheKeyDenseMapInfo>
      baseIndexCache;
  DenseMap<IndexCacheKeyT, SmallVector<SmallVector<Value>>,
           CacheKeyDenseMapInfo>
      indexCache;
  void initSharedMemory(ModuleAllocation &allocation,
                        TritonGPUToLLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto ctx = mod.getContext();
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    auto global = b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/0,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::kSharedMemorySpace));
    mod.walk([&](FunctionOpInterface funcOp) {
      Value funcSmem;
      b.setInsertionPointToStart(&funcOp.getFunctionBody().front());
      if (allocation.isRoot(funcOp)) {
        funcSmem = b.create<LLVM::AddressOfOp>(loc, global);
      } else {
        funcSmem = funcOp.getArgument(funcOp.getNumArguments() - 1);
      }
      auto ptrTy = LLVM::LLVMPointerType::get(
          ctx, NVVM::NVVMMemorySpace::kSharedMemorySpace);
      funcSmem = b.create<LLVM::BitcastOp>(loc, ptrTy, funcSmem);
      allocation.setFunctionSharedMemoryValue(funcOp, funcSmem);
    });
    mod->setAttr("triton_gpu.shared",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                        allocation.getSharedMemorySize()));
  }

  void decomposeFp8e4b15Convert(ModuleOp mod) const {
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      if (!getElementTypeOrSelf(cvtOp)
               .isa<mlir::Float8E4M3B11FNUZType, mlir::Float8E4M3FNType>())
        return;
      auto shape = cvtOp.getType().getShape();
      auto argEncoding = cvtOp.getSrc().getType().getEncoding();
      auto cvtEncoding = cvtOp.getType().getEncoding();
      if (argEncoding.isa<triton::gpu::DotOperandEncodingAttr>() ||
          cvtEncoding.isa<triton::gpu::DotOperandEncodingAttr>())
        return;
      auto F16Ty = builder.getF16Type();

      auto newArgType = RankedTensorType::get(shape, F16Ty, argEncoding);
      auto newCvtType = RankedTensorType::get(shape, F16Ty, cvtEncoding);
      auto newArg = builder.create<mlir::triton::FpToFpOp>(
          cvtOp.getLoc(), newArgType, cvtOp.getSrc());
      addWSNamedAttrs(newArg, cvtOp->getAttrs());
      auto newCvt = builder.create<mlir::triton::gpu::ConvertLayoutOp>(
          cvtOp.getLoc(), newCvtType, newArg);
      addWSNamedAttrs(newCvt, cvtOp->getAttrs());
      auto newRet = builder.create<mlir::triton::FpToFpOp>(
          cvtOp.getLoc(), cvtOp.getType(), newCvt.getResult());
      newRet.setRounding(
          triton::RoundingMode::RTNE); // Downcast requires rounding mode
      addWSNamedAttrs(newRet, cvtOp->getAttrs());
      cvtOp.replaceAllUsesWith(newRet.getResult());
      cvtOp.erase();
    });
  }

  void decomposeSplatToSharedLayout(ModuleOp mod, int numWarps,
                                    int threadsPerWarp, int numCTAs) const {
    // Replace `splat -> shared` with `splat -> blocked -> shared`.
    mod.walk([&](triton::SplatOp splatOp) -> void {
      auto dstType = splatOp.getType();
      auto shared =
          dstType.getEncoding().dyn_cast<triton::gpu::SharedEncodingAttr>();
      if (shared) {
        OpBuilder builder(splatOp);
        SmallVector<unsigned, 4> sizePerThread(dstType.getRank(), 1);
        auto newType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::BlockedEncodingAttr::get(
                mod.getContext(), dstType.getShape(), sizePerThread,
                getOrder(shared), numWarps, threadsPerWarp, numCTAs));
        auto newSplat = builder.create<triton::SplatOp>(
            splatOp.getLoc(), newType, splatOp.getSrc());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            splatOp.getLoc(), dstType, newSplat.getResult());
        splatOp.replaceAllUsesWith(newConvert.getResult());
        splatOp.erase();
      }
    });
  }

#ifdef USE_ROCM
  void decomposeMfmaToDotOperand(ModuleOp mod, int numWarps, int threadsPerWarp,
                                 int numCTAs) const {
    // Replace `mfma -> dot_op` with `mfma -> blocked -> dot_op`
    // unless certain conditions are met
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getSrc().getType();
      auto dstType = cvtOp.getType();
      auto srcMfma =
          srcType.getEncoding().dyn_cast<triton::gpu::MfmaEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      // llvm::outs() << isMfmaToDotShortcut(srcType, dstType) << "\n";
      if (srcMfma && dstDotOp && !isMfmaToDotShortcut(srcType, dstType)) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::BlockedEncodingAttr::get(
                mod.getContext(), srcType.getShape(), getSizePerThread(srcMfma),
                getOrder(srcMfma), numWarps, threadsPerWarp, numCTAs));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getSrc());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }

  int getCvtOpLDSUsage(triton::gpu::ConvertLayoutOp &cvtOp) const {
    unsigned inVec = 0;
    unsigned outVec = 0;
    auto smemShape = getScratchConfigForCvtLayout(cvtOp, inVec, outVec);
    unsigned elems = std::accumulate(smemShape.begin(), smemShape.end(), 1,
                                     std::multiplies{});
    auto srcType = cvtOp.getSrc().getType();
    auto bytes =
        srcType.getElementType().isa<triton::PointerType>()
            ? elems * kPtrBitWidth / 8
            : elems * std::max<int>(8, srcType.getElementTypeBitWidth()) / 8;

    return bytes;
  }

  bool isPowerOfTwo(unsigned x) const { return x && (x & (x - 1)) == 0; }

  std::vector<std::pair<int, int>> factorizePowerOf2(int n) const {
    assert(isPowerOfTwo(n));
    int x = log2(n);
    std::vector<std::pair<int, int>> pairs;

    for (int i = 0; i <= x / 2; ++i) {
      int j = x - i;
      pairs.push_back({pow(2, i), pow(2, j)});
      pairs.push_back({pow(2, j), pow(2, i)});
    }

    return pairs;
  }

  std::pair<triton::gpu::ConvertLayoutOp, triton::gpu::ConvertLayoutOp>
  createNewConvertOps(ModuleOp &mod, OpBuilder &builder,
                      triton::gpu::ConvertLayoutOp &cvtOp,
                      std::pair<unsigned, unsigned> warpsPerCta) const {
    unsigned warpsPerCtaX = warpsPerCta.first;
    unsigned warpsPerCtaY = warpsPerCta.second;
    auto srcType = cvtOp.getSrc().getType();
    auto dstType = cvtOp.getType();

    auto srcMfma =
        srcType.getEncoding().dyn_cast<triton::gpu::MfmaEncodingAttr>();
    auto newMfmaEnc = triton::gpu::MfmaEncodingAttr::get(
        mod.getContext(), srcMfma.getNonKDim(), {warpsPerCtaX, warpsPerCtaY},
        srcMfma.getIsTransposed(), srcMfma.getCTALayout());

    auto newDstType = RankedTensorType::get(
        dstType.getShape(), dstType.getElementType(), dstType.getEncoding());
    auto newSrcType = RankedTensorType::get(
        srcType.getShape(), srcType.getElementType(), newMfmaEnc);

    auto tmpCvt = builder.create<triton::gpu::ConvertLayoutOp>(
        cvtOp.getLoc(), newSrcType, cvtOp.getSrc());
    auto newEpilogueCvt = builder.create<triton::gpu::ConvertLayoutOp>(
        cvtOp.getLoc(), newDstType, tmpCvt);

    return std::make_pair(tmpCvt, newEpilogueCvt);
  }

  // Try to reduce LDS usage of cvt(mfma->blocked) op by changing the shape of
  // WarpsPerCta attribute in mfma layout. The implicit LDS usage of
  // cvt(mfma->blocked) op depends on the number of warps per CTA that mfma
  // layout uses along x dimension and block layout uses across y dimension.
  //
  // clang-format off
  //
  // LDS usage of this op is roughly calculated as:
  // LDS_USAGE = getShapePerCTA(mfma_layout)[0] * getShapePerCTA(blocked_layout)[1] * sizeof(data_type)
  // LDS_USAGE = warpsPerCTA(mfma_layout)[0] * warpsPerCta(blocked_layout)[1] * C,
  // where C = 32 * sizePerWarp(blocked_layout)[1] * threadsPerWarp(blocked_layout)[1] * sizeof(data_type)
  //
  // clang-format on
  //
  // When LDS_USAGE exceeds the size of LDS, try to lower LDS usage by
  // decomposing cvt(mfma->blocked) op into 2 conversions: cvt(mfma->mfma_tmp)
  // and cvt(mfma_tmp->blocked), where mfma_tmp has WarpsPerCta attribute that
  // minimizes uses of LDS for these conversions.
  void reduceCvtOpLDSUsage(ModuleOp mod) const {
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);

      auto srcType = cvtOp.getSrc().getType();
      auto dstType = cvtOp.getType();

      auto srcMfma =
          srcType.getEncoding().dyn_cast<triton::gpu::MfmaEncodingAttr>();
      auto dstBlocked =
          dstType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();

      if (!srcMfma || !dstBlocked) {
        return;
      }

      auto currLDSUsage = getCvtOpLDSUsage(cvtOp);
      if (currLDSUsage <= LDSSize) {
        return;
      }

      unsigned numWarps =
          srcMfma.getWarpsPerCTA()[0] * srcMfma.getWarpsPerCTA()[1];

      triton::gpu::ConvertLayoutOp tmpCvt;
      triton::gpu::ConvertLayoutOp newEpilogueCvt;

      // Find all possible shapes of WarpsPerCTA by finding all possible
      // factorizations of numWarps. Pick shape for which both conversions in
      // decomposition use LDS less than LDSSize and for which sum of LDS usage
      // is minimal. If no such shape exists, do not decompose.
      unsigned minLDSUsage = 2 * LDSSize;
      int minIdx = -1;
      auto factorizedNumWarps = factorizePowerOf2(numWarps);

      for (int i = 0; i < factorizedNumWarps.size(); i++) {
        auto warpsPerCTAPair = factorizedNumWarps[i];
        std::tie(tmpCvt, newEpilogueCvt) =
            createNewConvertOps(mod, builder, cvtOp, warpsPerCTAPair);

        int tmpCvtLDS = getCvtOpLDSUsage(tmpCvt);
        int newCvtLDS = getCvtOpLDSUsage(newEpilogueCvt);
        if (tmpCvtLDS <= LDSSize && newCvtLDS <= LDSSize) {
          int LDSUsage = tmpCvtLDS + newCvtLDS;
          if (LDSUsage < minLDSUsage) {
            minLDSUsage = LDSUsage;
            minIdx = i;
          }
        }
        newEpilogueCvt.erase();
        tmpCvt.erase();
      }

      if (minIdx == -1) {
        return;
      }

      assert(minIdx >= 0 && minIdx < factorizedNumWarps.size());
      auto warpsPerCTAPair = factorizedNumWarps[minIdx];
      std::tie(tmpCvt, newEpilogueCvt) =
          createNewConvertOps(mod, builder, cvtOp, warpsPerCTAPair);

      cvtOp.replaceAllUsesWith(newEpilogueCvt.getResult());
      cvtOp.erase();
    });
  }

#endif

  void decomposeMmaToDotOperand(ModuleOp mod, int numWarps, int threadsPerWarp,
                                int numCTAs) const {
    // Replace `mma -> dot_op` with `mma -> blocked -> dot_op`
    // unless certain conditions are met
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getSrc().getType();
      auto dstType = cvtOp.getType();
      auto srcMma =
          srcType.getEncoding().dyn_cast<triton::gpu::NvidiaMmaEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcMma && dstDotOp && !isMmaToDotShortcut(srcType, dstType)) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::BlockedEncodingAttr::get(
                mod.getContext(), srcType.getShape(), getSizePerThread(srcMma),
                getOrder(srcMma), numWarps, threadsPerWarp, numCTAs));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getSrc());
        addWSNamedAttrs(tmp, cvtOp->getAttrs());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        addWSNamedAttrs(newConvert, cvtOp->getAttrs());
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }

  void decomposeBlockedToDotOperand(ModuleOp mod) const {
    // Replace `blocked -> dot_op` with `blocked -> shared -> dot_op`
    // because the codegen doesn't handle `blocked -> dot_op` directly
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getSrc().getType();
      auto dstType = cvtOp.getType();
      auto srcBlocked =
          srcType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcBlocked && dstDotOp) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::SharedEncodingAttr::get(
                mod.getContext(), dstDotOp, srcType.getShape(),
                srcBlocked.getOrder(), srcBlocked.getCTALayout(),
                srcType.getElementType()));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getSrc());
        addWSNamedAttrs(tmp, cvtOp->getAttrs());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        addWSNamedAttrs(newConvert, cvtOp->getAttrs());
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }

  void decomposeInsertSliceAsyncOp(ModuleOp mod) const {
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    // TODO(Keren): This is a hacky knob that may cause performance regression
    // when decomposition has been performed. We should remove this knob once we
    // have thorough analysis on async wait. Currently, we decompose
    // `insert_slice_async` into `load` and `insert_slice` without knowing which
    // `async_wait` is responsible for the `insert_slice_async`. To guarantee
    // correctness, we blindly set the `async_wait` to wait for all async ops.
    //
    // There are two options to improve this:
    // 1. We can perform a dataflow analysis to find the `async_wait` that is
    // responsible for the `insert_slice_async` in the backend.
    // 2. We can modify the pipeline to perform the decomposition before the
    // `async_wait` is inserted. However, it is also risky because we don't know
    // the correct vectorized shape yet in the pipeline pass. Making the
    // pipeline pass aware of the vectorization could introduce additional
    // dependencies on the AxisInfoAnalysis and the Coalesce analysis.
    bool decomposed = false;
    // insert_slice_async %src, %dst, %idx, %mask, %other
    // =>
    // %tmp = load %src, %mask, %other
    // %res = insert_slice %tmp into %dst[%idx]
    mod.walk([&](triton::gpu::InsertSliceAsyncOp insertSliceAsyncOp) -> void {
      OpBuilder builder(insertSliceAsyncOp);

      // Get the vectorized load size
      auto src = insertSliceAsyncOp.getSrc();
      auto dst = insertSliceAsyncOp.getDst();
      auto mask = insertSliceAsyncOp.getMask();
      auto srcTy = src.getType();
      auto dstTy = dst.getType();
      auto srcBlocked =
          srcTy.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto resSharedLayout =
          dstTy.getEncoding().dyn_cast<triton::gpu::SharedEncodingAttr>();
      auto resElemTy = dstTy.getElementType();
      unsigned inVec = axisInfoAnalysis.getPtrContiguity(src);
      if (mask)
        inVec =
            std::min<unsigned>(axisInfoAnalysis.getMaskAlignment(mask), inVec);
      unsigned outVec = resSharedLayout.getVec();
      unsigned minVec = inVec;
      if (outVec > 1)
        minVec = std::min(outVec, inVec);
      auto maxBitWidth =
          std::max<unsigned>(128, resElemTy.getIntOrFloatBitWidth());
      auto vecBitWidth = resElemTy.getIntOrFloatBitWidth() * minVec;
      auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
      auto byteWidth = bitWidth / 8;

      // If the load byte width is not eligible or the current compute
      // capability does not support async copy, then we do decompose
#ifndef USE_ROCM
      if (triton::gpu::InsertSliceAsyncOp::getEligibleLoadByteWidth(
              computeCapability)
              .contains(byteWidth)) {
        return;
      }
#endif

      // load
      auto tmpTy =
          RankedTensorType::get(srcTy.getShape(), resElemTy, srcBlocked);
      auto loadOp = builder.create<triton::LoadOp>(
          insertSliceAsyncOp.getLoc(), tmpTy, insertSliceAsyncOp.getSrc(),
          insertSliceAsyncOp.getMask(), insertSliceAsyncOp.getOther(),
          // TODO(Chenggang): confirm `boundaryCheck` and `padding`
          /*boundaryCheck=*/nullptr, /*padding=*/nullptr,
          insertSliceAsyncOp.getCache(), insertSliceAsyncOp.getEvict(),
          insertSliceAsyncOp.getIsVolatile());
      addWSNamedAttrs(loadOp, insertSliceAsyncOp->getAttrs());

      // insert_slice
      auto axis = insertSliceAsyncOp.getAxis();
      auto intAttr = [&](int64_t v) { return builder.getI64IntegerAttr(v); };
      auto offsets = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(0));
      auto sizes = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(1));
      auto strides = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(1));
      offsets[axis] = insertSliceAsyncOp.getIndex();
      for (size_t i = 0; i < dstTy.getRank(); i++) {
        if (i != axis)
          sizes[i] = intAttr(dstTy.getShape()[i]);
      }
      auto insertSliceOp = builder.create<tensor::InsertSliceOp>(
          insertSliceAsyncOp.getLoc(), loadOp, insertSliceAsyncOp.getDst(),
          offsets, sizes, strides);
      addWSNamedAttrs(insertSliceOp, insertSliceAsyncOp->getAttrs());

      // Replace
      insertSliceAsyncOp.replaceAllUsesWith(insertSliceOp.getResult());
      insertSliceAsyncOp.erase();
      decomposed = true;
    });

    mod.walk([&](triton::gpu::AsyncCommitGroupOp asyncCommitGroupOp) -> void {
#ifdef USE_ROCM
      asyncCommitGroupOp.erase();
#else
      if (!triton::gpu::AsyncCommitGroupOp::isSupported(computeCapability))
        asyncCommitGroupOp.erase();
#endif
    });

    mod.walk([&](triton::gpu::AsyncWaitOp asyncWaitOp) -> void {
#ifdef USE_ROCM
      // AsyncWait is not supported for ROCM and should be removed
      asyncWaitOp.erase();
#else
      if (!triton::gpu::AsyncWaitOp::isSupported(computeCapability)) {
        // async wait is supported in Ampere and later
        asyncWaitOp.erase();
      } else if (decomposed) {
        // Wait for all previous async ops
        OpBuilder builder(asyncWaitOp);
        auto newWaitOp =
            builder.create<triton::gpu::AsyncWaitOp>(asyncWaitOp.getLoc(), 0);
        addWSNamedAttrs(newWaitOp, asyncWaitOp->getAttrs());
        asyncWaitOp.erase();
      }
#endif
    });
  }

  static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                              Type promotedType) {
    Type tensorPromotedType =
        operand.getType().cast<RankedTensorType>().cloneWith(std::nullopt,
                                                             promotedType);
    return builder.create<triton::FpToFpOp>(loc, tensorPromotedType, operand);
  }

  // promote operands of dot op if the existing combination is not natively
  // supported.
  void decomposeMixedModeDotOp(ModuleOp mod) const {
    mod.walk([](triton::DotOp dotOp) -> void {
      auto D = dotOp.getD();
      OpBuilder builder(dotOp);
      Type AElType =
          dotOp.getA().getType().getElementType();
      Type promoteType;
      NvidiaMmaEncodingAttr mmaLayout = D.getType()
                                            .getEncoding()
                                            .dyn_cast<NvidiaMmaEncodingAttr>();
      if (mmaLayout) {
        bool isNativeHopperFP8 =
            AElType.isFloat8E5M2() || AElType.isFloat8E4M3FNUZ();
        bool isFP8 = isNativeHopperFP8 || AElType.isFloat8E5M2FNUZ() ||
                     AElType.isFloat8E4M3FN() || AElType.isFloat8E4M3B11FNUZ();
        if (!isFP8 || (isNativeHopperFP8 && mmaLayout.isHopper()))
          return;
        promoteType = builder.getF16Type();
#ifdef USE_ROCM
      } else if (MfmaEncodingAttr mfmaLayout =
                     D.getType()
                         .getEncoding()
                         .dyn_cast<MfmaEncodingAttr>()) {
        Type BElType =
            dotOp.getB().getType().getElementType();

        auto maxBitWidth = std::max(AElType.getIntOrFloatBitWidth(),
                                    BElType.getIntOrFloatBitWidth());

        // TODO check mfma tensor core version compatibility
        if (maxBitWidth == 8)
          return;

        if (AElType == BElType)
          return;

        if (maxBitWidth < 16)
          promoteType = builder.getF16Type();
        else if (maxBitWidth <= 32)
          promoteType = builder.getF32Type();
#endif
      } else {
        // FMA case.
        Type AElType =
            dotOp.getA().getType().getElementType();
        Type DElType = D.getType().getElementType();
        if (AElType == DElType)
          return;
        promoteType = DElType;
      }
      Location loc = dotOp.getLoc();
      Value promotedA = promoteOperand(builder, loc, dotOp.getA(), promoteType);
      Value promotedB = promoteOperand(builder, loc, dotOp.getB(), promoteType);
      dotOp.setOperand(0, promotedA);
      dotOp.setOperand(1, promotedB);
    });
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonAMDGPUToLLVMPass() {
  return std::make_unique<ConvertTritonGPUToLLVM>(90, triton::ROCDL);
}

} // namespace triton
} // namespace mlir
