#include "triton/Conversion/TritonGPUToSPIRV/TritonGPUToSPIRVPass.h"

#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRV.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "triton/Analysis/Membar.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "TritonGPUToSPIRV.h"
#include "ViewOpToSPIRV.h"
#include "ElementwiseOpToSPIRV.h"
#include "LoadStoreOpToSPIRV.h"
#include "ReduceOpToSPIRV.h"
#include "ConvertLayoutOpToSPIRV.h"
#include "TypeConverter.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Conversion/Passes.h.inc"

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getThreadsPerCTA;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;


struct FuncOpConversionBase : public OpConversionPattern<func::FuncOp> {
protected:
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter provided
  // to this legalization pattern.
  spirv::FuncOp
  convertFuncOpToSPIRVFuncOp(func::FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) const {

    return nullptr;
  }

};

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncOpConversion : public FuncOpConversionBase {
  FuncOpConversion(SPIRVTypeConverter &converter, MLIRContext *context, int numWarps,
                   PatternBenefit benefit)
          : FuncOpConversionBase(converter, context, benefit), NumWarps(numWarps) {}

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp mod = dyn_cast<ModuleOp>(funcOp->getParentOp());
    if (!mod)
      return failure();

    auto shared = mod->getAttrOfType<mlir::IntegerAttr>("triton_gpu.shared");
    int shared_memory = shared.getInt();

    auto fnType = funcOp.getFunctionType();
    if (fnType.getNumResults() > 1)
      return failure();

    int num_inputs = fnType.getNumInputs();
    if (shared_memory)
      num_inputs +=1;
    TypeConverter::SignatureConversion signatureConverter(num_inputs);
    for (const auto &argType : enumerate(fnType.getInputs())) {
      auto convertedType = getTypeConverter()->convertType(argType.value());
      if (!convertedType)
        return failure();
      signatureConverter.addInputs(argType.index(), convertedType);
    }

    if (shared_memory) {
      Type int8_ty = rewriter.getIntegerType(8);
//      ::llvm::DebugFlag = true;
//      ::llvm::setCurrentDebugType("mlir-spirv-conversion");
      Type elemTy = getTypeConverter()->convertType(int8_ty);
//      ::llvm::DebugFlag = false;
      spirv::PointerType pointTy = spirv::PointerType::get(elemTy, spirv::StorageClass::Workgroup);
      signatureConverter.addInputs(num_inputs - 1, pointTy);
    }

    Type resultType;
    if (fnType.getNumResults() == 1) {
      resultType = getTypeConverter()->convertType(fnType.getResult(0));
      if (!resultType)
        return failure();
    }

    // Create the converted spv.func op.
    auto newFuncOp = rewriter.create<spirv::FuncOp>(
            funcOp.getLoc(), funcOp.getName(),
            rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                                     resultType ? TypeRange(resultType)
                                                : TypeRange()));

    // Set the SPIRV kernel entry point
    newFuncOp->setAttr(spirv::getEntryPointABIAttrName(), spirv::EntryPointABIAttr::get(getContext(), nullptr, std::nullopt));

    // Copy over all attributes other than the function name and type.
    for (const auto &namedAttr : funcOp->getAttrs()) {
        if (namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
            namedAttr.getName() != SymbolTable::getSymbolAttrName() &&
            namedAttr.getName() != funcOp.getArgAttrsAttrName())
        newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    ArrayAttr attrs = funcOp.getAllArgAttrs();
    for(int i = 0; i < attrs.size(); i++) {
      if (attrs[i].isa<mlir::DictionaryAttr>()) {
        newFuncOp.setArgAttrs(i, attrs[i].dyn_cast<mlir::DictionaryAttr>());
      }
    }


    if (shared_memory) {
      newFuncOp.setArgAttr(num_inputs - 1, "tt.scratch_memory_size",
                           mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 32),
                                                  shared_memory));
    }

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(
            &newFuncOp.getBody(), *getTypeConverter(), &signatureConverter)))
      return failure();
    rewriter.eraseOp(funcOp);
    return success();

  }

private:
  int NumWarps{0};
};

class TritonSPIRVFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonSPIRVFunctionConversionTarget(MLIRContext &ctx, SPIRVTypeConverter& typeConverter)
          : ConversionTarget(ctx) {
    addLegalDialect<spirv::SPIRVDialect>();
    addIllegalOp<mlir::func::FuncOp>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonSPIRVConversionTarget : public ConversionTarget {
public:
  explicit TritonSPIRVConversionTarget(MLIRContext &ctx, SPIRVTypeConverter& typeConverter)
          : ConversionTarget(ctx) {
    addLegalDialect<spirv::SPIRVDialect>();
//    addIllegalDialect<triton::TritonDialect>();
//    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addIllegalDialect<mlir::func::FuncDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    addDynamicallyLegalOp<mlir::func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
  }
};

class ConvertTritonGPUToSPIRV
        : public ConvertTritonGPUToSPIRVBase<ConvertTritonGPUToSPIRV> {

private:

  void decomposeMmaToDotOperand(ModuleOp mod, int numWarps, int threadsPerWarp) const {
    // Replace `mma -> dot_op` with `mma -> blocked -> dot_op`
    // unless certain conditions are met
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcMma =
              srcType.getEncoding().dyn_cast<triton::gpu::MmaEncodingAttr>();
      auto dstDotOp =
              dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcMma && dstDotOp && !isMmaToDotShortcut(srcMma, dstDotOp)) {
        assert(0 && "no dot op so far");
        auto tmpType = RankedTensorType::get(
                dstType.getShape(), dstType.getElementType(),
                triton::gpu::BlockedEncodingAttr::get(
                        mod.getContext(), srcType.getShape(), getSizePerThread(srcMma),
                        getOrder(srcMma), numWarps, threadsPerWarp));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
                cvtOp.getLoc(), tmpType, cvtOp.getOperand());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
                cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }

  void decomposeBlockedToDotOperand(ModuleOp mod) {
    // replace `blocked -> dot_op` with `blocked -> shared -> dot_op`
    // because the codegen doesn't handle `blocked -> dot_op` directly
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcBlocked =
              srcType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto dstDotOp =
              dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcBlocked && dstDotOp) {
        assert(0 && "no dot op so far");
        auto tmpType = RankedTensorType::get(
                dstType.getShape(), dstType.getElementType(),
                triton::gpu::SharedEncodingAttr::get(
                        mod.getContext(), dstDotOp, srcType.getShape(),
                        getOrder(srcBlocked), srcType.getElementType()));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
                cvtOp.getLoc(), tmpType, cvtOp.getOperand());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
                cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }

  LogicalResult decomposeInsertSliceAsyncOp(ModuleOp mod) const {
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    AxisInfoAnalysis *axisInfoAnalysis = solver->load<AxisInfoAnalysis>();
    if (failed(solver->initializeAndRun(mod)))
      return failure();
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
      assert(0 && "no triton::gpu::InsertSliceAsyncOp");
      OpBuilder builder(insertSliceAsyncOp);

      // Get the vectorized load size
      auto src = insertSliceAsyncOp.getSrc();
      auto dst = insertSliceAsyncOp.getDst();
      auto srcTy = src.getType().cast<RankedTensorType>();
      auto dstTy = dst.getType().cast<RankedTensorType>();
      auto srcBlocked =
          srcTy.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto resSharedLayout =
          dstTy.getEncoding().dyn_cast<triton::gpu::SharedEncodingAttr>();
      auto resElemTy = dstTy.getElementType();
      unsigned inVec = axisInfoAnalysis->getPtrContiguity(src);
      unsigned outVec = resSharedLayout.getVec();
      unsigned minVec = std::min(outVec, inVec);
      auto maxBitWidth =
          std::max<unsigned>(128, resElemTy.getIntOrFloatBitWidth());
      auto vecBitWidth = resElemTy.getIntOrFloatBitWidth() * minVec;
      auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
      auto byteWidth = bitWidth / 8;

      // If the load byte width is not eligible or the current compute
      // capability does not support async copy, then we do decompose
      if (triton::gpu::InsertSliceAsyncOp::getEligibleLoadByteWidth(
              computeCapability)
              .contains(byteWidth))
        return;

      // load
      auto tmpTy =
          RankedTensorType::get(srcTy.getShape(), resElemTy, srcBlocked);
      auto loadOp = builder.create<triton::LoadOp>(
          insertSliceAsyncOp.getLoc(), tmpTy, insertSliceAsyncOp.getSrc(),
          insertSliceAsyncOp.getMask(), insertSliceAsyncOp.getOther(),
          insertSliceAsyncOp.getCache(), insertSliceAsyncOp.getEvict(),
          insertSliceAsyncOp.getIsVolatile());

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

      // Replace
      insertSliceAsyncOp.replaceAllUsesWith(insertSliceOp.getResult());
      insertSliceAsyncOp.erase();
      decomposed = true;
    });

    mod.walk([&](triton::gpu::AsyncCommitGroupOp asyncCommitGroupOp) -> void {
      if (!triton::gpu::AsyncCommitGroupOp::isSupported(computeCapability))
        asyncCommitGroupOp.erase();
    });

    mod.walk([&](triton::gpu::AsyncWaitOp asyncWaitOp) -> void {
      if (!triton::gpu::AsyncWaitOp::isSupported(computeCapability)) {
        // async wait is supported in Ampere and later
        asyncWaitOp.erase();
      } else if (decomposed) {
        // Wait for all previous async ops
        OpBuilder builder(asyncWaitOp);
        builder.create<triton::gpu::AsyncWaitOp>(asyncWaitOp.getLoc(), 0);
        asyncWaitOp.erase();
      }
    });
    return success();
  }

public:
  explicit ConvertTritonGPUToSPIRV(int computeCapability)
          : computeCapability(computeCapability) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    spirv::Capability caps_opencl[] = {
            spirv::Capability::Addresses,
            spirv::Capability::Float16Buffer,
            spirv::Capability::Int64,
            spirv::Capability::Int16,
            spirv::Capability::Int8,
            spirv::Capability::Kernel,
            spirv::Capability::Linkage,
            spirv::Capability::Vector16,
            spirv::Capability::GenericPointer,
            spirv::Capability::Groups,
            spirv::Capability::Float16,
            spirv::Capability::Float64,
            spirv::Capability::AtomicFloat32AddEXT,
            spirv::Capability::ExpectAssumeKHR,
    };
    spirv::Extension exts_opencl[] = {
            spirv::Extension::SPV_EXT_shader_atomic_float_add,
            spirv::Extension::SPV_KHR_expect_assume};
    auto triple = spirv::VerCapExtAttr::get(
            spirv::Version::V_1_0, caps_opencl, exts_opencl, context);
    auto targetAttr = spirv::TargetEnvAttr::get(
            triple, spirv::getDefaultResourceLimits(context),
            spirv::ClientAPI::OpenCL,
            spirv::Vendor::Unknown,
            spirv::DeviceType::Unknown,
            spirv::TargetEnvAttr::kUnknownDeviceID);

    mod->setAttr(spirv::getTargetEnvAttrName(), targetAttr);

    SPIRVConversionOptions options;
    // TODO: need confirm
    options.use64bitIndex = false;
    TritonGPUToSPIRVTypeConverter spirvTypeConverter(targetAttr, options);
    TritonSPIRVFunctionConversionTarget funcTarget(*context, spirvTypeConverter);
    TritonSPIRVConversionTarget spirvTarget(*context, spirvTypeConverter);

    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    // Step 1: Decompose unoptimized layout conversions to use shared memory
    // Step 2: Decompose insert_slice_async to use load + insert_slice for
    //   pre-Ampere architectures or unsupported vectorized load sizes
    // Step 3: Allocate shared memories and insert barriers
    // Step 4: Convert SCF to CFG
    // Step 5: Get axis and shared memory info
    // Step 6: Convert FuncOp to spirv::FuncOp via partial conversion
    // Step 7: Convert the rest of ops via partial conversion
    //
    // The reason for putting step 3 before step 4 is that the membar
    // analysis currently only supports SCF but not CFG. The reason for a
    // separation between 5/7 is that, step 6 is out of the scope of Dialect
    // Conversion, thus we need to make sure the smem is not revised during the
    // conversion of step 7.

    // Step 1
    decomposeMmaToDotOperand(mod, numWarps, threadsPerWarp);
    decomposeBlockedToDotOperand(mod);

    // Step 2
    if (failed(decomposeInsertSliceAsyncOp(mod)))
      return signalPassFailure();

    // Step 3
    Allocation allocation(mod);
    MembarAnalysis membarPass(&allocation);
    membarPass.run();

    // Step 4
    RewritePatternSet scf_patterns(context);
    mlir::populateSCFToControlFlowConversionPatterns(scf_patterns);
    mlir::ConversionTarget scf_target(*context);
    scf_target.addIllegalOp<scf::ForOp, scf::IfOp, scf::ParallelOp,
            scf::WhileOp, scf::ExecuteRegionOp>();
    scf_target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(
            applyPartialConversion(mod, scf_target, std::move(scf_patterns))))
      return signalPassFailure();

    // Step 5 - get axis and shared memory info
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    AxisInfoAnalysis *axisInfoAnalysis = solver->load<AxisInfoAnalysis>();
    if (failed(solver->initializeAndRun(mod)))
      return signalPassFailure();
    mod->setAttr("triton_gpu.shared",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32),
                                        allocation.getSharedMemorySize()));

    // Step 6
    RewritePatternSet func_patterns(context);
    func_patterns.add<FuncOpConversion>(spirvTypeConverter, context, numWarps, 1 /*benefit*/);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(func_patterns))))
      return signalPassFailure();

    initSharedMemory(allocation.getSharedMemorySize(), spirvTypeConverter);

    // Step 7 - rewrite rest of ops
    // We set a higher benefit here to ensure triton's patterns runs before
    // arith patterns for some encoding not supported by the community
    // patterns.
    OpBuilder::InsertPoint indexInsertPoint;
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo indexCacheInfo{
            &baseIndexCache, &indexCache, &indexInsertPoint};

    RewritePatternSet patterns(context);
    // Normal conversions
    populateTritonGPUToSPIRVPatterns(spirvTypeConverter, context, patterns, numWarps,
                                     *axisInfoAnalysis, &allocation, smem,
                                    indexCacheInfo, /*benefit=*/10);
    // ConvertLayoutOp
    populateConvertLayoutOpToSPIRVPatterns(spirvTypeConverter, context, patterns, numWarps,
                                          *axisInfoAnalysis, &allocation, smem,
                                          indexCacheInfo, /*benefit=*/10);
    // DotOp
//    populateDotOpToLLVMPatterns(typeConverter, patterns, numWarps,
//                                *axisInfoAnalysis, &allocation, smem,
//            /*benefit=*/10);
    // ElementwiseOp
    populateElementwiseOpToSPIRVPatterns(spirvTypeConverter, context, patterns, numWarps,
                                        *axisInfoAnalysis, &allocation, smem,
            /*benefit=*/10);
    // LoadStoreOp
    populateLoadStoreOpToSPIRVPatterns(spirvTypeConverter, context, patterns, numWarps,
                                      *axisInfoAnalysis, &allocation, smem,
                                      indexCacheInfo, /*benefit=*/10);
    // ReduceOp
    populateReduceOpToSPIRVPatterns(spirvTypeConverter, context, patterns, numWarps,
                                   *axisInfoAnalysis, &allocation, smem,
                                   indexCacheInfo, /*benefit=*/10);
    // ViewOp
    populateViewOpToSPIRVPatterns(spirvTypeConverter, context, patterns, numWarps,
                                 *axisInfoAnalysis, &allocation, smem,
            /*benefit=*/10);

    // Add arith/math's patterns to help convert scalar expression to SPIRV.
    mlir::arith::populateArithToSPIRVPatterns(spirvTypeConverter,
                                                            patterns);
    mlir::populateMathToSPIRVPatterns(spirvTypeConverter, patterns);
    mlir::populateFuncToSPIRVPatterns(spirvTypeConverter, patterns);
    mlir::populateGPUToSPIRVPatterns(spirvTypeConverter, patterns);
    mlir::cf::populateControlFlowToSPIRVPatterns(spirvTypeConverter, patterns);

    if (failed(applyPartialConversion(mod, spirvTarget, std::move(patterns))))
      return signalPassFailure();
  }

protected:

  void initSharedMemory(size_t size,
                        TritonGPUToSPIRVTypeConverter &typeConverter);

  using IndexCacheKeyT = std::pair<Attribute, RankedTensorType>;
  DenseMap<IndexCacheKeyT, SmallVector<Value>, CacheKeyDenseMapInfo>
          baseIndexCache;
  DenseMap<IndexCacheKeyT, SmallVector<SmallVector<Value>>,
          CacheKeyDenseMapInfo>
          indexCache;

  Value smem;

  int computeCapability{};
};


void ConvertTritonGPUToSPIRV::initSharedMemory(
        size_t size, TritonGPUToSPIRVTypeConverter &typeConverter) {
  ModuleOp mod = getOperation();

  SmallVector<spirv::FuncOp> funcs;
  mod.walk([&](spirv::FuncOp func) { funcs.push_back(func); });
  assert(funcs.size() == 1 &&
         "spirv funcion conversion is expected before initSharedMemory");

  auto funcOp = funcs[0];

  if (size) {
    auto& bb0 = funcOp.getBody().getBlocks();
    auto& value = *(bb0.begin()->args_rbegin());
    smem = value;
  }
}


namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToSPIRVPass(int computeCapability) {
  return std::make_unique<::ConvertTritonGPUToSPIRV>(computeCapability);
}

} // namespace triton
} // namespace mlir