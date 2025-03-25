//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// TODO[dyq]: Pass Description
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUCREATEGM2LM
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

bool replaceAtomicOp(mlir::ModuleOp m) {
  bool getAtomicRMWOp = false;

  m.walk([&](triton::AtomicRMWOp atomicRMWOp) {
    getAtomicRMWOp = true;
    OpBuilder builder(atomicRMWOp);
    auto loc = atomicRMWOp.getLoc();
    Value ptr = atomicRMWOp.getPtr();
    Value val = atomicRMWOp.getVal();
    Value mask = atomicRMWOp.getMask();
    Value emptyBufPtr;
    RMWOp atomic_rmw_op = atomicRMWOp.getAtomicRmwOp();
    auto dtype = val.getType();
    auto loadOp = builder.create<triton::xpu::LoadOp>(
        loc, dtype, ptr, mask, Value(), Value(), 1, -1, false, false, false);

    Operation *arithOp;
    switch (atomic_rmw_op) {
    case RMWOp::AND: {
      arithOp = builder.create<arith::AndIOp>(loc, loadOp.getResult(), val);
      break;
    }
    case RMWOp::OR: {
      arithOp = builder.create<arith::OrIOp>(loc, loadOp.getResult(), val);
      break;
    }
    case RMWOp::XOR: {
      arithOp = builder.create<arith::XOrIOp>(loc, loadOp.getResult(), val);
      break;
    }
    case RMWOp::ADD: {
      arithOp = builder.create<arith::AddIOp>(loc, loadOp.getResult(), val);
      break;
    }
    case RMWOp::FADD: {
      arithOp = builder.create<arith::AddFOp>(loc, loadOp.getResult(), val);
      break;
    }
    case RMWOp::MAX: {
      arithOp = builder.create<arith::MaxSIOp>(loc, loadOp.getResult(), val);
      break;
    }
    case RMWOp::MIN: {
      arithOp = builder.create<arith::MinSIOp>(loc, loadOp.getResult(), val);
      break;
    }
    case RMWOp::UMAX: {
      arithOp = builder.create<arith::MaxUIOp>(loc, loadOp.getResult(), val);
      break;
    }
    case RMWOp::UMIN: {
      arithOp = builder.create<arith::MinUIOp>(loc, loadOp.getResult(), val);
      break;
    }
    case RMWOp::XCHG: {
      assert(0 && "The RMWOp::XCHG is not supported in RMWOp");
      break;
    }
    default: {
      assert(0 && "The atomic_rmw_op only could be 1-10 in RMWOp");
    }
    }

    auto storeOp = builder.create<triton::xpu::StoreOp>(
        loc, ptr, arithOp->getResults()[0], mask, Value(), -1, false);
    atomicRMWOp.erase();
  });

  return getAtomicRMWOp;
}

Attribute getOneCoreGEncoding(Operation *op, ArrayRef<int64_t> shape) {
  Attribute newEncoding;
  unsigned rank = shape.size();
  llvm::SmallVector<unsigned> sizePerBank;
  llvm::SmallVector<unsigned> coresPerGroup;
  llvm::SmallVector<unsigned> groupsPerCluster;
  llvm::SmallVector<unsigned> order;
  bool isReduceOpt = false;

  if (rank == 1) {
    sizePerBank = {static_cast<unsigned>(shape[0])};
    coresPerGroup = {1};
    groupsPerCluster = {1};
    order = {0};
  } else if (rank == 2) {
    sizePerBank = {1, static_cast<unsigned>(shape[1])};
    coresPerGroup = {1, 1};
    groupsPerCluster = {1, 1};
    order = {0, 1};
  } else {
    llvm_unreachable("AtomicOp Simulation With Rank > 2 Unsupported");
  }

  newEncoding = triton::xpu::ClusterLayoutAttr::get(
      op->getContext(), sizePerBank, coresPerGroup, groupsPerCluster, order,
      isReduceOpt);
  return newEncoding;
}

bool atomicSimulation(mlir::ModuleOp m) {

  // Step 1. Replace AtomicRMWOp with GM2LMOp + Arith.xxx + LM2GMOp
  if (replaceAtomicOp(m)) {
    // Step 2. Modify All Op Encoding
    m.walk([&](mlir::Operation *op) {
      auto opResult = op->getResults();
      if (opResult.size() == 1) { // SSA Assert
        // Only TensorType Has Encoding
        if (auto resTy =
                mlir::dyn_cast<RankedTensorType>(opResult[0].getType())) {
          auto shape = resTy.getShape();
          auto elemTy = resTy.getElementType();
          auto encoding = resTy.getEncoding();
          Attribute newEncoding; // newEncoding

          auto globalEncoding =
              mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(encoding);
          auto sliceEncoding =
              mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(encoding);

          if (globalEncoding) {
            newEncoding = getOneCoreGEncoding(op, shape);
          } else if (sliceEncoding) {
            // must be globalEncoding
            auto parentEncoding =
                mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(
                    sliceEncoding.getParent());

            if (parentEncoding) {
              auto newParentEncoding = getOneCoreGEncoding(op, shape);
              newEncoding = triton::gpu::SliceEncodingAttr::get(
                  op->getContext(), sliceEncoding.getDim(), newParentEncoding);
            } else {
              llvm_unreachable("Unsupported SliceEncoding's Parent Attribute");
            }
          } else {
            llvm_unreachable("Unsupported Encoding Attribute");
          }

          auto newResTy = RankedTensorType::get(shape, elemTy, newEncoding);
          opResult[0].setType(newResTy);
        }
      }
    });

    // Step 3. Special Modification For [constOp]
    // Step 3.1. ConstOp: value's encoding is not modified before this walk
    m.walk([&](arith::ConstantOp constOp) {
      auto newValue = constOp.getValue();
      if (auto attr =
              mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue())) {
        newValue = DenseElementsAttr::getFromRawBuffer(
            mlir::cast<ShapedType>(constOp.getType()), attr.getRawData());
      }
      OpBuilder builder(constOp);
      auto loc = constOp.getLoc();
      auto newConstOp = builder.create<mlir::arith::ConstantOp>(
          loc, constOp.getType(), newValue);
      constOp.replaceAllUsesWith(newConstOp.getResult());
      constOp.erase();
    });

    // Step 3.2. ExpandDimsOp: it expands the data dimension, so its prev
    // cvtOp's correct encoding should be inferd by its operand. cvtOp is
    // actually generated after expandDimsOp, so we need to modify the
    // encoding of the previous cvtOp after determining the shape of
    // expandDimsOp.
    m.walk([&](triton::ExpandDimsOp expandOp) {
      auto expandOpType = mlir::cast<RankedTensorType>(expandOp.getType());
      auto globalEncoding = mlir::cast<triton::xpu::ClusterLayoutAttr>(
          expandOpType.getEncoding());

      if (auto cvtOp =
              expandOp.getSrc().getDefiningOp<triton::xpu::ConvertLayoutOp>()) {
        auto cvtOpType = mlir::cast<RankedTensorType>(cvtOp.getType());
        auto sliceEncoding =
            mlir::cast<triton::gpu::SliceEncodingAttr>(cvtOpType.getEncoding());

        auto newSliceEncoding = triton::gpu::SliceEncodingAttr::get(
            expandOp->getContext(), sliceEncoding.getDim(), globalEncoding);
        auto newResTy = RankedTensorType::get(
            cvtOpType.getShape(), cvtOpType.getElementType(), newSliceEncoding);

        cvtOp->getResult(0).setType(newResTy);
      } else {
        llvm_unreachable("ExpandDimsOp With Error Operand");
      }
    });

    // Step 3.3. ForOp: we need to modify forOp's argTy, args can't be
    m.walk([&](scf::ForOp forOp) {
      auto forBody = forOp.getBody();
      // modify forOp's argTy
      auto forArgs = forBody->getArguments();
      for (auto forArg : forArgs) {
        if (auto argTy = mlir::dyn_cast<RankedTensorType>(forArg.getType())) {
          auto shape = argTy.getShape();
          auto elemTy = argTy.getElementType();
          auto argEncoding =
              mlir::cast<triton::xpu::ClusterLayoutAttr>(argTy.getEncoding());

          auto newArgEncoding = getOneCoreGEncoding(forOp, shape);
          auto newArgTy = RankedTensorType::get(shape, elemTy, newArgEncoding);

          forArg.setType(newArgTy);
        }
      }

      // modify forOp's resTy
      auto forResults = forOp->getResults();
      for (auto forRes : forResults) {
        if (auto argTy = mlir::dyn_cast<RankedTensorType>(forRes.getType())) {
          auto shape = argTy.getShape();
          auto elemTy = argTy.getElementType();
          auto argEncoding =
              mlir::cast<triton::xpu::ClusterLayoutAttr>(argTy.getEncoding());

          auto newArgEncoding = getOneCoreGEncoding(forOp, shape);
          auto newArgTy = RankedTensorType::get(shape, elemTy, newArgEncoding);

          forRes.setType(newArgTy);
        }
      }
    });

    // Step 3.4. ReduceOp: it reduces the data dimension, so its correct
    // encoding should be inferd by its input type.
    m.walk([&](triton::ReduceOp redOp) {
      llvm_unreachable("TODO[dyq]: new reduceOp has multi operands and "
                       "results, we need to modify all Tys");
      auto resTy = mlir::cast<RankedTensorType>(redOp.getType(0));
      auto srcTy = mlir::cast<RankedTensorType>(redOp.getOperand(0).getType());

      auto resSliceEncoding =
          mlir::cast<triton::gpu::SliceEncodingAttr>(resTy.getEncoding());
      auto srcGlobalEncoding =
          mlir::cast<triton::xpu::ClusterLayoutAttr>(srcTy.getEncoding());

      auto newEncoding = triton::gpu::SliceEncodingAttr::get(
          redOp.getContext(), resSliceEncoding.getDim(), srcGlobalEncoding);
      auto newResTy = RankedTensorType::get(
          resTy.getShape(), resTy.getElementType(), newEncoding);

      redOp->getResult(0).setType(newResTy);
    });

    return true;
  }

  return false;
}

struct TritonXPUCreateGM2LMPass
    : public impl::TritonXPUCreateGM2LMBase<TritonXPUCreateGM2LMPass> {

  using impl::TritonXPUCreateGM2LMBase<
      TritonXPUCreateGM2LMPass>::TritonXPUCreateGM2LMBase;

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    bool hasAtomicSim = false;

    // Replace AtomicRMWOp with GM2LMOp + Arith.xxx + LM2GMOp(Embedding
    // Backward)
    if (atomicSim)
      hasAtomicSim = atomicSimulation(m);

    llvm::SmallSetVector<Operation *, 4> opToErase;
    Value emptyBufPtr;

    // FIXME: Sometimes (test_core.py::test_bin_op_constexpr)
    // `triton::LoadOp` and `triton::StoreOp` can not be replaced
    // with `triton::xpu::LoadOp` and `triton::xpu::StoreOp` in
    // TritonToTritonXPUPass, So we workaround to replace it here.
    m.walk([&](triton::LoadOp loadOp) {
      auto loc = loadOp.getLoc();
      OpBuilder builder(loadOp);
      auto newLoadOp = builder.create<triton::xpu::LoadOp>(
          loc, loadOp.getType(), loadOp.getPtr(), loadOp.getMask(),
          loadOp.getOther(), Value(), 1, -1, false, false, false);
      loadOp.replaceAllUsesWith(newLoadOp.getResult());
      opToErase.insert(loadOp);
    });

    m.walk([&](triton::StoreOp storeOp) {
      auto loc = storeOp.getLoc();
      OpBuilder builder(storeOp);
      auto newLoadOp = builder.create<triton::xpu::StoreOp>(
          loc, storeOp.getPtr(), storeOp.getValue(), storeOp.getMask(), Value(),
          -1, false);
      opToErase.insert(storeOp);
    });

    m.walk([&](triton::xpu::LoadOp loadOp) {
      OpBuilder builder(loadOp);
      auto loc = loadOp.getLoc();
      auto lmPtrType = addrspaceCast(loadOp.getPtr().getType(), 0);
      if (xpuArch == 2) {
        if (loadOp.getResult().hasOneUse()) {
          if (auto extFOp = dyn_cast<arith::ExtFOp>(*(loadOp->user_begin()))) {
            auto gm2lmOp = builder.create<triton::xpu::GM2LMOp>(
                loc, lmPtrType, loadOp.getPtr(), loadOp.getMask(), emptyBufPtr,
                static_cast<int32_t>(OffsetState::Unknown), -1, -1, -1, -1, -1,
                false, false, hasAtomicSim);
            loadOp.setOperand(0, gm2lmOp.getResult());
            loadOp.getResult().setType(extFOp.getType());
            extFOp.getResult().replaceAllUsesWith(loadOp.getResult());
            opToErase.insert(extFOp);
            return;
          }
        }
      }
      auto gm2lmOp = builder.create<triton::xpu::GM2LMOp>(
          loc, lmPtrType, loadOp.getPtr(), loadOp.getMask(), emptyBufPtr,
          static_cast<int32_t>(OffsetState::Unknown), -1, -1, -1, -1, -1, false,
          false, hasAtomicSim);
      loadOp.setOperand(0, gm2lmOp.getResult());
    });

    m.walk([&](triton::xpu::StoreOp storeOp) {
      OpBuilder builder(storeOp);
      auto loc = storeOp.getLoc();
      auto storeVal = storeOp.getValue();
      if (xpuArch == 2) {
        if (storeVal.getDefiningOp()) {
          if (auto truncFOp =
                  dyn_cast<arith::TruncFOp>(storeVal.getDefiningOp())) {
            storeVal = truncFOp.getIn();
          }
        }
      }
      storeOp->setOperand(1, storeVal);
      auto lm2gmOp = builder.create<triton::xpu::LM2GMOp>(
          loc, storeOp.getPtr(), storeVal, storeOp.getMask(), emptyBufPtr,
          static_cast<int32_t>(OffsetState::Unknown), -1, -1, -1, hasAtomicSim);
      lm2gmOp->moveAfter(storeOp);
    });

    for (auto op : opToErase) {
      op->erase();
    }
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
