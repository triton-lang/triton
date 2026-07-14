#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Support/WalkResult.h"
#include "triton/Dialect/Triton/Transforms/ArithTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/FunctionTypeConversion.h"

#include "triton/Dialect/Triton/Transforms/TensorDescriptorConversionUtils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCONVERTTENSORDESCRIPTOR
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

class TritonAMDGPUConvertTensorDescriptorPass
    : public impl::TritonAMDGPUConvertTensorDescriptorBase<
          TritonAMDGPUConvertTensorDescriptorPass> {

public:
  using Base::Base;

  void runOnOperation() override {
    auto module = getOperation();

    mlir::ConversionTarget target(getContext());
    target.addIllegalOp<triton::DescriptorReduceOp>();

    // Create a custom attribute to tag ops that must be converted
    auto needsConversionAttr =
        mlir::StringAttr::get(&getContext(), "triton.illegal_desc");
    auto unitAttr = mlir::UnitAttr::get(&getContext());

    // Ops are marked as legal or illegal ops explicitly.
    bool reduceOnly = archGenerationName == "gfx1250";
    if (reduceOnly) {
      llvm::DenseSet<Value> illegalDescriptors;
      module.walk([&illegalDescriptors](triton::DescriptorReduceOp reduceOp) {
        illegalDescriptors.insert(reduceOp.getDesc());
      });

      auto legalReduce = module.walk([&](triton::DescriptorReduceOp op) {
        auto funcOp = op->getParentOfType<triton::FuncOp>();
        if (funcOp->hasAttr("noinline") &&
            funcOp->getAttrOfType<BoolAttr>("noinline").getValue()) {
          return mlir::WalkResult::interrupt();
        }

        return mlir::WalkResult::advance();
      });

      if (legalReduce.wasInterrupted()) {
        signalPassFailure();
      }

      // If a descriptor is used by reduce and used in by other operations in a
      // loop, no matter reduce is inside a loop or not, bail out for now.
      auto reduceDescriptorInLoop = module.walk([&](scf::ForOp forOp) {
        for (auto opnd : forOp.getOperands()) {
          auto type = opnd.getType();
          if (llvm::isa<mlir::triton::TensorDescType>(type) &&
              illegalDescriptors.contains(opnd))
            return mlir::WalkResult::interrupt();
        }

        return mlir::WalkResult::advance();
      });

      if (reduceDescriptorInLoop.wasInterrupted()) {
        module->emitError("Not supported if reduce descriptor in the loop.");
        signalPassFailure();
      }

      // Tag the operations which define or use the descriptor
      module.walk([&](triton::DescriptorReduceOp reduceOp) {
        reduceOp->setAttr(needsConversionAttr, unitAttr);

        Value desc = reduceOp.getDesc();

        if (Operation *defOp = desc.getDefiningOp()) {
          defOp->setAttr(needsConversionAttr, unitAttr);
        } else if (auto blockArg = dyn_cast<BlockArgument>(desc)) {
          auto parentOp = blockArg.getOwner()->getParentOp();
          if (auto funcOp = dyn_cast<triton::FuncOp>(parentOp)) {
            funcOp->setAttr(needsConversionAttr, unitAttr);
          }
        }

        for (Operation *user : desc.getUsers()) {
          user->setAttr(needsConversionAttr, unitAttr);
        }
      });

      target.addDynamicallyLegalOp<triton::MakeTensorDescOp,
                                   triton::DescriptorLoadOp,
                                   triton::DescriptorStoreOp>(
          [needsConversionAttr](mlir::Operation *op) {
            if (!op->hasAttr(needsConversionAttr))
              return true;

            auto isDescType = [](mlir::Type t) {
              return llvm::isa<mlir::triton::TensorDescType>(t);
            };

            bool hasDescOperand =
                llvm::any_of(op->getOperandTypes(), isDescType);
            bool hasDescResult = llvm::any_of(op->getResultTypes(), isDescType);
            return !hasDescOperand && !hasDescResult;
          });

      // Tag FuncOp separately because it depends on the signature of funcop.
      target.addDynamicallyLegalOp<triton::FuncOp>(
          [needsConversionAttr](triton::FuncOp funcOp) {
            if (!funcOp->hasAttr(needsConversionAttr)) {
              return true;
            }

            auto isDescType = [](mlir::Type t) {
              return llvm::isa<mlir::triton::TensorDescType>(t);
            };

            auto funcType = funcOp.getFunctionType();
            bool hasDescArg = llvm::any_of(funcType.getInputs(), isDescType);
            bool hasDescRet = llvm::any_of(funcType.getResults(), isDescType);

            return !hasDescArg && !hasDescRet;
          });

      target.addDynamicallyLegalDialect<mlir::arith::ArithDialect,
                                        mlir::triton::TritonDialect>(
          [](mlir::Operation *op) { return true; });
    } else {
      target.addDynamicallyLegalDialect<
          mlir::arith::ArithDialect, mlir::scf::SCFDialect,
          mlir::triton::TritonDialect>([](mlir::Operation *op) {
        return !mlir::triton::hasATensorDescriptorType(op->getOperandTypes()) &&
               !mlir::triton::hasATensorDescriptorType(op->getResultTypes());
      });

      target.addDynamicallyLegalOp<triton::FuncOp>([](triton::FuncOp funcOp) {
        return !mlir::triton::hasATensorDescriptorType(
                   funcOp.getFunctionType().getInputs()) &&
               !mlir::triton::hasATensorDescriptorType(
                   funcOp.getFunctionType().getResults());
      });

      target
          .addIllegalOp<triton::DescriptorLoadOp, triton::DescriptorStoreOp,
                        triton::DescriptorScatterOp, triton::DescriptorGatherOp,
                        triton::MakeTensorDescOp>();
    }

    auto converter = mlir::triton::createDescTypeConverter();
    auto renamer = mlir::triton::createFunArgRenamer();

    mlir::RewritePatternSet patterns(module->getContext());

    // Populate conversion patterns to handle loops, function calls, and arith
    // ops.
    triton::populateFunctionTypeConversions(converter, renamer, patterns);
    mlir::scf::populateSCFStructuralTypeConversions(converter, patterns);
    triton::populateArithTypeConversions(converter, patterns);

    mlir::triton::populateMakeTensorDescriptorPattern(patterns, converter);
    mlir::triton::populateLoadTensorDescriptorPattern(patterns, converter);
    mlir::triton::populateStoreTensorDescriptorPattern(patterns, converter);
    mlir::triton::populateGatherDescriptorPattern(patterns, converter);
    mlir::triton::populateScatterTensorDescriptorPattern(patterns, converter);
    mlir::triton::populateReduceTensorDescriptorPattern(patterns, converter);

    ConversionConfig config;
    config.buildMaterializations = false;

    if (mlir::failed(mlir::applyPartialConversion(
            module, target, std::move(patterns), config))) {
      signalPassFailure();
    }

    // The code to do the descriptor conversion is in the common path and the
    // custom attribute is copied to the converted ops. So we do the clean up
    // after conversion.
    if (reduceOnly) {
      module.walk(
          [&](mlir::Operation *op) { op->removeAttr(needsConversionAttr); });
    }
  }
};
} // namespace
} // namespace mlir
