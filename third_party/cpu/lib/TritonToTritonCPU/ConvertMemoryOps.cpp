#include "TypeConverter.h"

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
#define GEN_PASS_DEF_CONVERTMEMORYOPS
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

struct LoadOpConversion : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();
    auto mask = loadOp.getMask();
    auto ptr = loadOp.getPtr();
    auto boundaryChecks = loadOp.getBoundaryCheck();

    if (!triton::isTensorPointerType(ptr.getType())) {
      return lowerToScalarLoads(loadOp, rewriter);
    }

    // TODO: support masks.
    if (mask) {
      llvm_unreachable("unsupported load op");
    }

    auto memRef = rewriter.getRemappedValue(ptr);
    auto rank = dyn_cast<MemRefType>(memRef.getType()).getRank();
    auto resTy = dyn_cast<VectorType>(
        getTypeConverter()->convertType(loadOp.getResult().getType()));
    auto indices = rewriter.create<ExtractIndicesOp>(loc, ptr).getResults();
    SmallVector<bool, 4> inBounds(rank, true);
    for (auto dim : boundaryChecks) {
      inBounds[dim] = false;
    }
    auto vecRead = rewriter.create<vector::TransferReadOp>(loc, resTy, memRef,
                                                           indices, inBounds);
    rewriter.replaceOp(loadOp, vecRead);
    return success();
  }

  LogicalResult lowerToScalarLoads(triton::LoadOp loadOp,
                                   ConversionPatternRewriter &rewriter) const {
    // Scalar loads and boundary checks are not expected.
    assert(loadOp.getBoundaryCheck().empty());
    assert(isa<RankedTensorType>(loadOp.getType()));

    auto loc = loadOp.getLoc();
    auto vecTy =
        dyn_cast<VectorType>(getTypeConverter()->convertType(loadOp.getType()));
    auto ptrs = rewriter.getRemappedValue(loadOp.getPtr());
    auto mask = loadOp.getMask() ? rewriter.getRemappedValue(loadOp.getMask())
                                 : nullptr;
    auto ptrTy =
        dyn_cast<RankedTensorType>(loadOp.getPtr().getType()).getElementType();
    auto cache = loadOp.getCache();
    auto evict = loadOp.getEvict();
    auto isVolatile = loadOp.getIsVolatile();

    Value defaultVal = loadOp.getOther();
    if (!defaultVal)
      defaultVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(vecTy.getElementType()));
    Value dst = rewriter.create<vector::BroadcastOp>(loc, vecTy, defaultVal);

    int64_t numElems = vecTy.getNumElements();
    auto strides = computeStrides(vecTy.getShape());
    for (auto idx = 0; idx < numElems; ++idx) {
      auto indices = delinearize(idx, strides);
      Block *headerBlock = rewriter.getBlock();
      Block *condBlock = nullptr;
      Value origDst = dst;
      // Create a conditional block for load if there is a mask.
      if (mask) {
        condBlock =
            rewriter.splitBlock(headerBlock, rewriter.getInsertionPoint());
        rewriter.setInsertionPointToStart(condBlock);
      }

      Value ptr = rewriter.create<vector::ExtractOp>(loc, ptrs, indices);
      ptr = rewriter.create<IntToPtrOp>(loc, ptrTy, ptr);
      Value val =
          rewriter.create<triton::LoadOp>(loc, ptr, cache, evict, isVolatile);
      dst = rewriter.create<vector::InsertOp>(loc, val, dst, indices);

      // Add predicate and branches.
      if (mask) {
        Block *footerBlock =
            rewriter.splitBlock(condBlock, rewriter.getInsertionPoint());
        Value resDst = dst;
        dst = footerBlock->addArgument(dst.getType(), dst.getLoc());
        rewriter.setInsertionPointToEnd(headerBlock);
        auto predicate = rewriter.create<vector::ExtractOp>(loc, mask, indices);
        rewriter.create<cf::CondBranchOp>(loc, predicate, condBlock,
                                          footerBlock, origDst);
        rewriter.setInsertionPointToEnd(condBlock);
        rewriter.create<cf::BranchOp>(loc, footerBlock, resDst);
        rewriter.setInsertionPointToStart(footerBlock);
      }
    }

    rewriter.replaceOp(loadOp, dst);

    return success();
  }
};

struct StoreOpConversion : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = storeOp.getLoc();
    auto mask = storeOp.getMask();
    auto ptr = storeOp.getPtr();
    auto boundaryChecks = storeOp.getBoundaryCheck();

    if (!triton::isTensorPointerType(ptr.getType())) {
      return lowerToScalarStores(storeOp, rewriter);
    }

    // TODO: support masks.
    if (mask) {
      llvm_unreachable("unsupported store op");
    }

    auto value = rewriter.getRemappedValue(storeOp.getValue());
    auto memRef = rewriter.getRemappedValue(ptr);
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

  LogicalResult lowerToScalarStores(triton::StoreOp storeOp,
                                    ConversionPatternRewriter &rewriter) const {
    // Scalar stores and boundary checks are not expected.
    assert(storeOp.getBoundaryCheck().empty());
    assert(isa<RankedTensorType>(storeOp.getValue().getType()));

    auto loc = storeOp.getLoc();
    auto ptrs = rewriter.getRemappedValue(storeOp.getPtr());
    auto mask = storeOp.getMask() ? rewriter.getRemappedValue(storeOp.getMask())
                                  : nullptr;
    auto vals = rewriter.getRemappedValue(storeOp.getValue());
    auto tensorTy = dyn_cast<RankedTensorType>(storeOp.getPtr().getType());
    auto ptrTy = tensorTy.getElementType();
    auto cache = storeOp.getCache();
    auto evict = storeOp.getEvict();

    int64_t numElems = tensorTy.getNumElements();
    auto strides = computeStrides(tensorTy.getShape());
    for (auto idx = 0; idx < numElems; ++idx) {
      auto indices = delinearize(idx, strides);
      Block *headerBlock = rewriter.getBlock();
      Block *condBlock = nullptr;
      // Create a conditional block for store if there is a mask.
      if (mask) {
        condBlock =
            rewriter.splitBlock(headerBlock, rewriter.getInsertionPoint());
        rewriter.setInsertionPointToStart(condBlock);
      }

      Value ptr = rewriter.create<vector::ExtractOp>(loc, ptrs, indices);
      ptr = rewriter.create<IntToPtrOp>(loc, ptrTy, ptr);
      Value val = rewriter.create<vector::ExtractOp>(loc, vals, indices);
      rewriter.create<triton::StoreOp>(loc, ptr, val, cache, evict);

      // Add predicate and branches.
      if (mask) {
        Block *footerBlock =
            rewriter.splitBlock(condBlock, rewriter.getInsertionPoint());
        rewriter.setInsertionPointToEnd(headerBlock);
        auto predicate = rewriter.create<vector::ExtractOp>(loc, mask, indices);
        rewriter.create<cf::CondBranchOp>(loc, predicate, condBlock,
                                          footerBlock);
        rewriter.setInsertionPointToEnd(condBlock);
        rewriter.create<cf::BranchOp>(loc, footerBlock);
        rewriter.setInsertionPointToStart(footerBlock);
      }
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
    addLegalDialect<cf::ControlFlowDialect>();
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
    : public triton::impl::ConvertMemoryOpsBase<ConvertMemoryOps> {
  using ConvertMemoryOpsBase::ConvertMemoryOpsBase;

  ConvertMemoryOps() : ConvertMemoryOpsBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    MemoryOpConversionTarget convTarget(*context);
    TritonToTritonCPUTypeConverter pointerConverter;
    RewritePatternSet patterns(context);
    patterns.add<LoadOpConversion>(pointerConverter, context);
    patterns.add<StoreOpConversion>(pointerConverter, context);

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

} // namespace cpu
} // namespace triton
} // namespace mlir
