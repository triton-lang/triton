//===----------------------------------------------------------------------===//
//
// Copyright (c) Triton Project Contributors.
//
//===----------------------------------------------------------------------===//

#include "triton/Analysis/UseAnalysis.h"
#include "triton/Conversion/TritonToLinalg/TritonToLinalg.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-to-linalg"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton/Conversion/TritonToLinalg/Passes.h.inc"

namespace {

class TritonTypeConverter : public TypeConverter {
public:
  TritonTypeConverter() {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      return UnrankedMemRefType::get(ptrType.getPointeeType(), 0);
    });
    addConversion([](TensorType tensorType) -> Type {
      auto elemType = tensorType.getElementType();
      if (auto ptrType = elemType.dyn_cast<triton::PointerType>()) {
        elemType = ptrType.getPointeeType();
      }
      return MemRefType::get(tensorType.getShape(), elemType);
    });
  }
};

struct TritonToLinalgPass : public TritonToLinalgBase<TritonToLinalgPass> {

  static unsigned int constexpr LAUNCH_GRID_RANK = 3;

  // Add additional I32 arguments to represent program
  // ID, one for each dimension of the launch grid
  static void addProgramId(triton::FuncOp func) {
    OpBuilder b(func);

    auto origFuncType = func.getFunctionType();
    auto origInputTypes = origFuncType.getInputs();
    SmallVector<Type> newInputTypes(origInputTypes);
    newInputTypes.append(LAUNCH_GRID_RANK, b.getI32Type());

    auto newFuncType =
        b.getFunctionType(newInputTypes, origFuncType.getResults());

    func.setFunctionType(newFuncType);

    // Add empty attributes for each new argument if needed
    if (func.getAllArgAttrs()) {
      SmallVector<DictionaryAttr> newArgAttrs;
      func.getAllArgAttrs(newArgAttrs);
      newArgAttrs.append(LAUNCH_GRID_RANK, DictionaryAttr());
      func.setAllArgAttrs(newArgAttrs);
    }

    // Add the corresponding arguments to function body
    for (unsigned int i = 0; i < LAUNCH_GRID_RANK; i++) {
      func.getBody().front().addArgument(b.getI32Type(), func.getLoc());
    }
  }

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, AffineDialect, scf::SCFDialect,
                    tensor::TensorDialect, bufferization::BufferizationDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    {
      RewritePatternSet patterns(&getContext());
      populateTritonToLinalgCanonicalizationPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
        signalPassFailure();
      }
    }

    moduleOp.walk([this](triton::FuncOp op) {
      if (failed(runUseAnalysis(op))) {
        signalPassFailure();
      }
    });

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonTypeConverter tritonTypeConverter;

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, memref::MemRefDialect>();

    target.addLegalOp<ModuleOp>();

    target.addIllegalDialect<triton::TritonDialect>();

    // triton.reduce will be lowered to linalg.reduce. Unfortunately, mlir
    // inserts the ops inside triton.reduce's region BEFORE triton.reduce
    // itself, so the conversion algorithm will visit triton.reduce_return
    // first. Without marking this op as legal, the conversion process will fail
    // because there's no legalization pattern for triton.reduce_return.
    target.addLegalOp<triton::ReduceReturnOp>();

    target.addLegalOp<triton::ReturnOp>();

    // Update function signature to use memrefs
    target.addDynamicallyLegalOp<triton::FuncOp>([&](triton::FuncOp op) {
      return tritonTypeConverter.isSignatureLegal(op.getFunctionType());
    });

    // Lower dense constant to linalg.fill
    target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) {
      if (!isa<RankedTensorType>(op.getResult().getType())) {
        return true;
      }

      if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (denseAttr.isSplat() &&
            isa<FloatType, IntegerType>(denseAttr.getElementType())) {
          return false;
        }
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::ForOp, scf::YieldOp>([](Operation *op) {
      return llvm::all_of(op->getOperandTypes(), [](Type t) {
        if (isa<triton::PointerType>(t)) {
          return false;
        }
        if (auto shapedType = dyn_cast<ShapedType>(t)) {
          return shapedType.getElementType().isIntOrFloat();
        }
        assert(t.isIntOrIndexOrFloat());
        return true;
      });
    });

    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
        [](Operation *op) {
          if (op->hasAttr("MetaUse")) {
            return false;
          }

          if (isa<arith::ConstantOp>(op)) {
            return true;
          }

          bool operateOnTensors =
              llvm::all_of(op->getOperandTypes(), [](Type type) {
                return type.isa<RankedTensorType>();
              });

          return !operateOnTensors;
        });

    triton::populateTritonToLinalgConversionPatterns(
        tritonTypeConverter, patterns, LAUNCH_GRID_RANK);

    for (auto func : getOperation().getOps<triton::FuncOp>())
      addProgramId(func);

    if (failed(applyFullConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();

    // Convert tt.func and tt.return into func's counterparts
    moduleOp.walk([&](triton::FuncOp func) {
      OpBuilder builder(func);

      auto name = func.getName();
      auto type = func.getFunctionType();

      SmallVector<DictionaryAttr> argAttrs, resAttrs;
      func.getAllArgAttrs(argAttrs);
      func.getAllResultAttrs(resAttrs);

      auto funcFunc = builder.create<func::FuncOp>(func.getLoc(), name, type);
      funcFunc.setAllArgAttrs(argAttrs);
      funcFunc.setAllResultAttrs(resAttrs);

      auto &funcFuncBody = funcFunc.getBody();
      auto &funcBody = func.getBody();

      IRMapping map;
      funcBody.cloneInto(&funcFuncBody, map);

      for (Block &block : funcFuncBody.getBlocks()) {
        auto term = block.getTerminator();
        builder.setInsertionPoint(term);
        builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
        term->erase();
      }
      func.erase();
    });

    // Erase dead code and fold constants created during lowering
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonToLinalgPass() {
  return std::make_unique<TritonToLinalgPass>();
}
