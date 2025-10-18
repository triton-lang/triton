#include <iterator>
#include <numeric>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonamdgpu-implicit-convert-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUIMPLICITCONVERTLAYOUT
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

struct ImplicitConvertLayoutPass : public impl::TritonAMDGPUImplicitConvertLayoutBase<ImplicitConvertLayoutPass> {
  static Type getNewType(Type type, Attribute encoding) {
    RankedTensorType tensorType = cast<RankedTensorType>(type);
    return tensorType.cloneWithEncoding(encoding);
  }

  void coalesceOp(Attribute srcEncoding, Attribute dstEncoding, Operation *op) {
    OpBuilder builder(op);
    // Convert operands
    // For load/store with tensor pointers, we don't have to change the
    // operands' type, we do this by changing the outputs' type of
    // `make_tensor_ptr`
    SmallVector<Value, 4> newArgs;
    for (auto operand : op->getOperands()) {
      auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
      if (tensorType &&
          !isa<ttg::SharedEncodingTrait>(tensorType.getEncoding())) {
        Type newType = getNewType(tensorType, srcEncoding);
        newArgs.push_back(builder.create<ttg::ConvertLayoutOp>(
            op->getLoc(), newType, operand));
      } else {
        newArgs.push_back(operand);
      }
    }

    // Convert output types
    SmallVector<Type, 4> newTypes;
    for (auto t : op->getResultTypes()) {
      bool isAsync = isa<ttg::AsyncCopyGlobalToLocalOp>(op);
      newTypes.push_back(isAsync ? t : getNewType(t, dstEncoding));
    }

    // Construct new op with the new encoding
    Operation *newOp =
        builder.create(op->getLoc(), op->getName().getIdentifier(), newArgs,
                       newTypes, op->getAttrs());

    // Cast the results back to the original layout
    for (size_t i = 0; i < op->getNumResults(); i++) {
      Value newResult = newOp->getResult(i);
      if (newTypes[i] != op->getResultTypes()[i]) {
        newResult = builder.create<ttg::ConvertLayoutOp>(
            op->getLoc(), op->getResult(i).getType(), newResult);
      }
      op->getResult(i).replaceAllUsesWith(newResult);
    }
    op->erase();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    moduleOp.walk([&](Operation *cur) {
      if (auto loadOp = dyn_cast<tt::LoadOp>(cur)) {
        auto type = loadOp.getResult().getType();
        if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
          auto encoding = tensorTy.getEncoding();
          llvm::dbgs() << "load op: " << loadOp << "\n";
          llvm::dbgs() << "type: " << type << "\n";
          llvm::dbgs() << "encoding: " << encoding << "\n";
          if (auto blockedEncoding = dyn_cast<ttg::BlockedEncodingAttr>(encoding)) {
            llvm::dbgs() << "blocked encoding to linear layout: " << blockedEncoding.toLinearLayout(tensorTy.getShape()) << "\n";
          }
        }
      }

      auto dot = dyn_cast<tt::DotOp>(cur);
      if (!dot)
        return;

      // 1. Check if the dot operand satisfies the implicit conversion conditions
      auto BOperand = dot.getB();
      RankedTensorType BOperandTy = BOperand.getType();
      auto opEncoding = dyn_cast<ttg::DotOperandEncodingAttr>(BOperandTy.getEncoding());
      if (!opEncoding)
        return;

      // Get backward slices util load op
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      auto filter = [&dot](Operation *op) {
        return op->getParentRegion() == dot->getParentRegion();
      };
      opt.filter = filter;
      opt.inclusive = true;
      llvm::SetVector<Operation *> backwardSlices;
      llvm::SmallVector<Operation *> reversedBackwardSlices;
      (void)getBackwardSlice(BOperand, &backwardSlices, opt);
      for (auto sliceIter = backwardSlices.rbegin(); sliceIter != backwardSlices.rend(); sliceIter++) {
        reversedBackwardSlices.emplace_back(*sliceIter);
        if (isa<tt::LoadOp>(*sliceIter)) {
          break;
        }
      }
      if (reversedBackwardSlices.empty() || !isa<tt::LoadOp>(reversedBackwardSlices.back())) {
        return;
      }

      // Get vectorization factor of load op
      tt::LoadOp loadOp = dyn_cast<tt::LoadOp>(reversedBackwardSlices.back());
      auto loadTy = loadOp.getType();
      int vecFactor = 1;
      const int MIN_KWIDTH = 4;
      if (auto loadTensorTy = dyn_cast<RankedTensorType>(loadTy)) {
        if (auto loadBlockedLayout = dyn_cast<ttg::BlockedEncodingAttr>(loadTensorTy.getEncoding())) {
          auto sizePerThread = loadBlockedLayout.getSizePerThread();
          auto loadOrder = loadBlockedLayout.getOrder();
          vecFactor = sizePerThread[loadOrder[0]];
        }
      }
      if (vecFactor < MIN_KWIDTH) {
        return;
      }

      // 2. Infer backward layout conversion "#tt.dot -> #tt.load"
      // `layoutMap` maps backward slices to their input layouts
      llvm::MapVector<Operation *, Attribute> layoutMap;
      auto newBOpLayout = ttg::DotOperandEncodingAttr::get(
        BOperandTy.getContext(), 1, opEncoding.getParent(), vecFactor);
      ttg::LinearEncodingAttr curLayout = ttg::LinearEncodingAttr::get(
        BOperandTy.getContext(), newBOpLayout.toLinearLayout(BOperandTy.getShape()));
      Attribute lastLayout = newBOpLayout;
      for (auto slice : reversedBackwardSlices) {
        if (!isa<ttg::LocalLoadOp, ttg::LocalAllocOp>(slice)) {
          tt::LinearLayout linearLayout = curLayout.getLinearLayout();
          auto resultTy = dyn_cast<RankedTensorType>(slice->getResult(0).getType());
          if (auto transOp = dyn_cast<tt::TransOp>(slice)) {
            auto transOrder = to_vector(transOp.getOrder());
            auto originOrder = transOrder;
            for (int i = 0; i < transOrder.size(); i++) {
              originOrder[transOrder[i]] = i;
            }
            linearLayout = transposeLinearLayout(curLayout.getLinearLayout(), originOrder);
          }
          else if (auto reshapeOp = dyn_cast<tt::ReshapeOp>(slice)) {
            auto originShape = reshapeOp.getSrc().getType().getShape();
            linearLayout = reshapeLayout(slice->getContext(), curLayout.getLinearLayout(), originShape);
          }
          // Make sure only valid instructions are included
          // else if (!(isa<ttg::ConvertLayoutOp, tt::LoadOp>(slice) 
          //     || slice->hasTrait<OpTrait::SameOperandsAndResultEncoding>() 
          //     || slice->hasTrait<OpTrait::Elementwise>())) {
          //   llvm::dbgs() << "slice: " << *slice << "\n";
          //   assert(false && "unsupported op");
          // }
          lastLayout = curLayout;
          curLayout = ttg::LinearEncodingAttr::get(BOperandTy.getContext(), linearLayout);
          layoutMap[slice] = curLayout;
          llvm::dbgs() << "slice: " << *slice << " \n-> input layout: " << layoutMap[slice] << "\n";
        }
        else {
          assert(false && "local load/alloc should not appear in implicit convert layout");
        }
      }

      // 3. Propagate layout to forward slices (backward slices 
      // should be handled by `remove_layout_conversions` pass)
      for (auto it = reversedBackwardSlices.rbegin(); it != reversedBackwardSlices.rend(); it++) {
        Operation *slice = *it;
        if (isa<ttg::ConvertLayoutOp>(slice)) {
          Value srcVal = slice->getOperand(0);
          Value dstVal = slice->getResult(0);
          dstVal.replaceAllUsesWith(srcVal);
          slice->erase();
          layoutMap.erase(slice);
        }
        else {
          OpBuilder rewriter(slice);
          Attribute srcEncoding = layoutMap[slice];
          Attribute dstEncoding = inferDstEncoding(slice, srcEncoding);
          if (slice == reversedBackwardSlices.front()) {
            dstEncoding = newBOpLayout;
          }
          llvm::dbgs() << "op: " << *slice << "\n";
          llvm::dbgs() << "src encoding: " << srcEncoding << "\n";
          llvm::dbgs() << "dst encoding: " << dstEncoding << "\n";

          // `coalesceOp` will insert convert layout before and after `slice`, 
          // and we will remove them in `remove_layout_conversions` pass
          coalesceOp(srcEncoding, dstEncoding, slice);
        }
      }

      // 4. Replace layout of operand B
      BOperand = dot.getB();
      OpBuilder rewriter(BOperand.getDefiningOp());
      rewriter.setInsertionPointAfter(BOperand.getDefiningOp());
      auto newBType = RankedTensorType::get(
        BOperand.getType().getShape(),
        BOperand.getType().getElementType(),
        newBOpLayout
      );
      auto newBOperand = rewriter.create<ttg::ConvertLayoutOp>(
        BOperand.getDefiningOp()->getLoc(), newBType, BOperand);
      BOperand.replaceAllUsesExcept(newBOperand, newBOperand);

      llvm::dbgs() << "dot op: " << dot << "\n";
      llvm::dbgs() << "B tensor type: " << BOperandTy << "\n";
      llvm::dbgs() << "encoding: " << opEncoding << "\n";
      // llvm::dbgs() << "linear layout: " << opEncoding.toLinearLayout(tensorTy.getShape()) << "\n";
      // llvm::dbgs() << "reversed backward slices:\n";
      // for (auto slice : reversedBackwardSlices) {
      //   llvm::dbgs() << *slice << "\n";
      // }
      llvm::dbgs() << "vectorization factor: " << vecFactor << "\n";
      llvm::dbgs() << "new layout: " << newBOpLayout << "\n";
      llvm::dbgs() << "BOperand: " << BOperand << "\n";
      llvm::dbgs() << "BOperand defining op: " << *BOperand.getDefiningOp() << "\n";

      llvm::dbgs() << "\n";

      // 5. Replace layout of operand A
      auto AOperand = dot.getA();
      auto AOperandTy = AOperand.getType();
      opEncoding = dyn_cast<ttg::DotOperandEncodingAttr>(AOperandTy.getEncoding());
      if (!opEncoding)
        return;
      auto newAOpLayout = ttg::DotOperandEncodingAttr::get(
        AOperandTy.getContext(), 0, opEncoding.getParent(), vecFactor);

      // Assume A{#dot_op(0)} is defined by `A = #ttg.local_load ...`,
      // we change the output layout of #ttg.local_load directly
      auto localLoadOp = dyn_cast<ttg::LocalLoadOp>(AOperand.getDefiningOp());
      assert(localLoadOp && "A should be defined by local load");
      rewriter.setInsertionPointAfter(localLoadOp);
      auto newLocalLoadOp = rewriter.clone(*localLoadOp);
      AOperandTy = AOperandTy.cloneWithEncoding(newAOpLayout);
      newLocalLoadOp->getResult(0).setType(AOperandTy);
      AOperand.replaceAllUsesWith(newLocalLoadOp->getResult(0));
      localLoadOp->erase();

      // llvm::dbgs() << "current function:\n";
      // llvm::dbgs() << *cur->getParentOfType<tt::FuncOp>() << "\n";
    });
  }
};

}
} // namespace mlir
