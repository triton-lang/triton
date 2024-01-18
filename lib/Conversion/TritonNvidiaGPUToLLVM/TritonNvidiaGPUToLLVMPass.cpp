#include "triton/Conversion/TritonNvidiaGPUToLLVM/TritonNvidiaGPUToLLVMPass.h"
#include "../lib/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
using namespace mlir;
using namespace mlir::triton;
#define GEN_PASS_CLASSES
#include "triton/Conversion/TritonNvidiaGPUToLLVM/Passes.h.inc"

namespace {

// TODO[goostavz]: GetThreadIdOp/GetClusterCTAIdOp is a temporary solution
// before async dialect is done. These concepts should appear in ttgpu
// level, and they are planned to be deprecated along with ttgpu.mbarrier_xxx
// ops.
struct GetThreadIdOpConversion : public ConvertTritonGPUOpToLLVMPattern<
                                     triton::nvidia_gpu::GetThreadIdOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::GetThreadIdOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::GetThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, getThreadId(rewriter, op->getLoc()));
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

    // Collect TMA information.
    unsigned numTMALoad = 0;
    funcOp.walk(
        [&numTMALoad](triton::nvidia_gpu::InsertSliceTMAOp insertSliceOp) {
          numTMALoad++;
        });
    unsigned numTMAStore = 0;
    funcOp.walk(
        [&numTMAStore](triton::nvidia_gpu::StoreAsyncTMAOp storeAsyncOp) {
          numTMAStore++;
        });
    unsigned numTMA = numTMALoad + numTMAStore;

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
    // Set an attribute for maxntidx, it could be used in latter LLVM codegen
    // for `nvvm.annotation` metadata.
    newFuncOp->setAttr("nvvm.maxntid",
                       rewriter.getDenseI32ArrayAttr(32 * numWarps));
    // The call graph is updated by mapping the old function to the new one.
    allocation.mapFuncOp(funcOp, newFuncOp);

    // Append arguments to receive TMADesc in global memory in the runtime
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 1);
    auto numArgs = newFuncOp.getBody().front().getNumArguments();
    auto funcTy = newFuncOp.getFunctionType().cast<LLVM::LLVMFunctionType>();
    SmallVector<Type> newInputsTy(funcTy.getParams().begin(),
                                  funcTy.getParams().end());
    for (unsigned i = 0; i < numTMA; ++i) {
      newFuncOp.getBody().front().addArgument(ptrTy, funcOp.getLoc());
      newInputsTy.push_back(ptrTy);
    }
    newFuncOp.setType(
        LLVM::LLVMFunctionType::get(funcTy.getReturnType(), newInputsTy));
    // required by AxisInfoAnalysis
    for (unsigned i = 0; i < numTMA; ++i) {
      newFuncOp.setArgAttr(numArgs + i, "tt.divisibility",
                           rewriter.getIntegerAttr(i32_ty, 1));
    }

    newFuncOp->setAttr(kAttrNumTMALoadDescsName,
                       rewriter.getIntegerAttr(i32_ty, numTMALoad));
    newFuncOp->setAttr(kAttrNumTMAStoreDescsName,
                       rewriter.getIntegerAttr(i32_ty, numTMAStore));

    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  int numWarps{0};
  ModuleAllocation &allocation;
};

struct GetCanonicalWarpIdConversion
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::nvidia_gpu::GetCanonicalWarpId> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::GetCanonicalWarpId>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::GetCanonicalWarpId op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, GetCanonicalWarpId(rewriter, op->getLoc()));
    return success();
  }
};

struct GetClusterCTAIdOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::nvidia_gpu::GetClusterCTAIdOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::GetClusterCTAIdOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::GetClusterCTAIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, getClusterCTAId(rewriter, op->getLoc()));
    return success();
  }
};

class FoldSplatMaskInInsertAsync : public mlir::RewritePattern {

public:
  FoldSplatMaskInInsertAsync(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            triton::nvidia_gpu::InsertSliceTMAOp::getOperationName(), 1,
            context) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto insertOp = cast<triton::nvidia_gpu::InsertSliceTMAOp>(op);
    if (!insertOp.getMask())
      return failure();
    auto splatOp = insertOp.getMask().getDefiningOp<triton::SplatOp>();
    if (!splatOp)
      return failure();
    rewriter.updateRootInPlace(insertOp, [&]() {
      insertOp.getMaskMutable().assign(splatOp->getOperand(0));
    });
    return mlir::success();
  }
};

class ConvertTritonNvidiaGPUToLLVM
    : public ConvertTritonNvidiaGPUToLLVMBase<ConvertTritonNvidiaGPUToLLVM> {

public:
  explicit ConvertTritonNvidiaGPUToLLVM() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    RewritePatternSet patterns(context);

    /* Get tensorPtrMap before conversion */
    TensorPtrMapT tensorPtrMap;
    mod.walk(
        [&tensorPtrMap](mlir::triton::nvidia_gpu::InsertSliceTMAOp insertOp) {
          auto src = insertOp.getSrc();
          auto ptrTy = src.getType().dyn_cast<triton::PointerType>();
          if (ptrTy && ptrTy.getPointeeType().isa<RankedTensorType>()) {
            auto makeTensorPtrOp = getMakeTensorPtrOp(insertOp.getSrc());
            tensorPtrMap[insertOp.getOperation()] = makeTensorPtrOp;
          }
        });

    mod.walk(
        [&tensorPtrMap](mlir::triton::nvidia_gpu::StoreAsyncTMAOp storeOp) {
          auto dst = storeOp.getDst();
          auto ptrTy = dst.getType().dyn_cast<triton::PointerType>();
          if (ptrTy && ptrTy.getPointeeType().isa<RankedTensorType>()) {
            auto makeTensorPtrOp = getMakeTensorPtrOp(storeOp.getDst());
            tensorPtrMap[storeOp.getOperation()] = makeTensorPtrOp;
          }
        });

    // Hack: cleanup
    {
      RewritePatternSet patterns(context);
      patterns.add<FoldSplatMaskInInsertAsync>(context);
      SmallVector<Operation *> insertSlices;
      mod.walk([&insertSlices](triton::nvidia_gpu::InsertSliceTMAOp op) {
        insertSlices.push_back(op);
      });
      if (applyOpPatternsAndFold(insertSlices, std::move(patterns)).failed())
        signalPassFailure();
    }

    // Hack: replace threadId w/ (threadId%128) if warp specialization is
    // enabled
    mod.walk([&tensorPtrMap](mlir::gpu::ThreadIdOp tid) {
      auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
      auto newTid = tid;
      if (ttng::TritonNvidiaGPUDialect::getWSSupportedAttr(mod)) {
        Value _128 = rewriter.create<arith::ConstantIntOp>(loc, 128, 32);
        newTid = rewriter.create<arith::RemSIOp>(loc, tid, _128);
      }
      tid.replaceAllUsesWith(newTid);
    });

    // Add patterns
    if (applyPatternsAndFoldGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonNvidiaGPUToLLVMPass() {
  return std::make_unique<::ConvertTritonNvidiaGPUToLLVM>();
}

} // namespace triton
} // namespace mlir
