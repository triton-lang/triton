#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM//ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "ConvertLayoutOpToLLVM.h"
#include "DotOpToLLVM.h"
#include "ElementwiseOpToLLVM.h"
#include "LoadStoreOpToLLVM.h"
#include "ReduceOpToLLVM.h"
#include "TritonGPUToLLVM.h"
#include "TypeConverter.h"
#include "ViewOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Conversion/Passes.h.inc"

namespace mlir {

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addIllegalOp<mlir::func::FuncOp>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

} // namespace mlir

namespace {

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncOpConversion : public FuncOpConversionBase {
  FuncOpConversion(LLVMTypeConverter &converter, int numWarps,
                   PatternBenefit benefit)
      : FuncOpConversionBase(converter, benefit), numWarps(numWarps) {}

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (!newFuncOp)
      return failure();

    auto ctx = funcOp->getContext();

    // Set an attribute to indicate this function is a kernel entry.
    newFuncOp->setAttr("nvvm.kernel",
                       rewriter.getIntegerAttr(type::u1Ty(ctx), 1));

    // Set an attribute for maxntidx, it could be used in latter LLVM codegen
    // for `nvvm.annotation` metadata.
    newFuncOp->setAttr("nvvm.maxntid",
                       rewriter.getIntegerAttr(i32_ty, 32 * numWarps));

    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  int numWarps{0};
};

class ConvertTritonGPUToLLVM
    : public ConvertTritonGPUToLLVMBase<ConvertTritonGPUToLLVM> {

public:
  explicit ConvertTritonGPUToLLVM(int computeCapability)
      : computeCapability(computeCapability) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMFunctionConversionTarget funcTarget(*context);
    TritonLLVMConversionTarget target(*context);

    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);

    // Step 1: Decompose unoptimized layout conversions to use shared memory
    // Step 2: Decompose insert_slice_async to use load + insert_slice for
    //   pre-Ampere architectures or unsupported vectorized load sizes
    // Step 3: Allocate shared memories and insert barriers
    // Step 4: Convert SCF to CFG
    // Step 5: Convert FuncOp to LLVMFuncOp via partial conversion
    // Step 6: Get axis and shared memory info
    // Step 7: Convert the rest of ops via partial conversion
    //
    // The reason for putting step 3 before step 4 is that the membar
    // analysis currently only supports SCF but not CFG. The reason for a
    // separation between 5/7 is that, step 6 is out of the scope of Dialect
    // Conversion, thus we need to make sure the smem is not revised during the
    // conversion of step 7.

    // Step 1
    decomposeMmaToDotOperand(mod, numWarps);
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

    // Step 5
    RewritePatternSet func_patterns(context);
    func_patterns.add<FuncOpConversion>(typeConverter, numWarps, /*benefit=*/1);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(func_patterns))))
      return signalPassFailure();

    // Step 6 - get axis and shared memory info
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    AxisInfoAnalysis *axisInfoAnalysis = solver->load<AxisInfoAnalysis>();
    if (failed(solver->initializeAndRun(mod)))
      return signalPassFailure();
    initSharedMemory(allocation.getSharedMemorySize(), typeConverter);
    mod->setAttr("triton_gpu.shared",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32),
                                        allocation.getSharedMemorySize()));

    // Step 7 - rewrite rest of ops
    // We set a higher benefit here to ensure triton's patterns runs before
    // arith patterns for some encoding not supported by the community
    // patterns.
    OpBuilder::InsertPoint indexInsertPoint;
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo indexCacheInfo{
        &baseIndexCache, &indexCache, &indexInsertPoint};

    RewritePatternSet patterns(context);

    // Normal conversions
    populateTritonGPUToLLVMPatterns(typeConverter, patterns, numWarps,
                                    *axisInfoAnalysis, &allocation, smem,
                                    indexCacheInfo, /*benefit=*/10);
    // ConvertLayoutOp
    populateConvertLayoutOpToLLVMPatterns(typeConverter, patterns, numWarps,
                                          *axisInfoAnalysis, &allocation, smem,
                                          indexCacheInfo, /*benefit=*/10);
    // DotOp
    populateDotOpToLLVMPatterns(typeConverter, patterns, numWarps,
                                *axisInfoAnalysis, &allocation, smem,
                                /*benefit=*/10);
    // ElementwiseOp
    populateElementwiseOpToLLVMPatterns(typeConverter, patterns, numWarps,
                                        *axisInfoAnalysis, &allocation, smem,
                                        /*benefit=*/10);
    // LoadStoreOp
    populateLoadStoreOpToLLVMPatterns(typeConverter, patterns, numWarps,
                                      *axisInfoAnalysis, &allocation, smem,
                                      indexCacheInfo, /*benefit=*/10);
    // ReduceOp
    populateReduceOpToLLVMPatterns(typeConverter, patterns, numWarps,
                                   *axisInfoAnalysis, &allocation, smem,
                                   indexCacheInfo, /*benefit=*/10);
    // ViewOp
    populateViewOpToLLVMPatterns(typeConverter, patterns, numWarps,
                                 *axisInfoAnalysis, &allocation, smem,
                                 /*benefit=*/10);

    // Add arith/math's patterns to help convert scalar expression to LLVM.
    mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter,
                                                            patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }

private:
  Value smem;

  using IndexCacheKeyT = std::pair<Attribute, SmallVector<int64_t>>;
  DenseMap<IndexCacheKeyT, SmallVector<Value>, CacheKeyDenseMapInfo>
      baseIndexCache;
  DenseMap<IndexCacheKeyT, SmallVector<SmallVector<Value>>,
           CacheKeyDenseMapInfo>
      indexCache;

  int computeCapability{};

  void initSharedMemory(size_t size,
                        TritonGPUToLLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    auto global = b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/0,
        mlir::gpu::GPUDialect::getWorkgroupAddressSpace());
    SmallVector<LLVM::LLVMFuncOp> funcs;
    mod.walk([&](LLVM::LLVMFuncOp func) { funcs.push_back(func); });
    assert(funcs.size() == 1 &&
           "Inliner pass is expected before TritonGPUToLLVM");
    b.setInsertionPointToStart(&funcs[0].getBody().front());
    smem = b.create<LLVM::AddressOfOp>(loc, global);
    auto ptrTy =
        LLVM::LLVMPointerType::get(typeConverter.convertType(b.getI8Type()), 3);
    smem = b.create<LLVM::BitcastOp>(loc, ptrTy, smem);
  }

  void decomposeMmaToDotOperand(ModuleOp mod, int numWarps) const {
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
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::BlockedEncodingAttr::get(
                mod.getContext(), srcType.getShape(), getSizePerThread(srcMma),
                getOrder(srcMma), numWarps));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getOperand());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
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
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcBlocked =
          srcType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcBlocked && dstDotOp) {
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
      OpBuilder builder(insertSliceAsyncOp);

      // Get the vectorized load size
      auto src = insertSliceAsyncOp.src();
      auto dst = insertSliceAsyncOp.dst();
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
          insertSliceAsyncOp.getLoc(), tmpTy, insertSliceAsyncOp.src(),
          insertSliceAsyncOp.mask(), insertSliceAsyncOp.other(),
          insertSliceAsyncOp.cache(), insertSliceAsyncOp.evict(),
          insertSliceAsyncOp.isVolatile());

      // insert_slice
      auto axis = insertSliceAsyncOp.axis();
      auto intAttr = [&](int64_t v) { return builder.getI64IntegerAttr(v); };
      auto offsets = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(0));
      auto sizes = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(1));
      auto strides = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(1));
      offsets[axis] = insertSliceAsyncOp.index();
      for (size_t i = 0; i < dstTy.getRank(); i++) {
        if (i != axis)
          sizes[i] = intAttr(dstTy.getShape()[i]);
      }
      auto insertSliceOp = builder.create<tensor::InsertSliceOp>(
          insertSliceAsyncOp.getLoc(), loadOp, insertSliceAsyncOp.dst(),
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
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int computeCapability) {
  return std::make_unique<::ConvertTritonGPUToLLVM>(computeCapability);
}

} // namespace triton
} // namespace mlir
