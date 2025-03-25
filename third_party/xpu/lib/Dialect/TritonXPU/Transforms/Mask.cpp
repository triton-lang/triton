//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// TODO: Pass Description
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#include "mlir/IR/IRMapping.h"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUMASK
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUMaskPass : public impl::TritonXPUMaskBase<TritonXPUMaskPass> {

public:
  using impl::TritonXPUMaskBase<TritonXPUMaskPass>::TritonXPUMaskBase;

  void getOpChain(llvm::SetVector<Operation *> &opChain, Operation *op) {
    if (!op || opChain.contains(op))
      return;

    opChain.insert(op);

    if (op->use_empty())
      return;

    for (Operation *user : op->getUsers()) {
      getOpChain(opChain, user);
    }
  }

  bool isFindUserOpImpl(Operation *startingOp, Operation *targetOp,
                        llvm::SetVector<Operation *> &visitedOps) {
    if (startingOp == targetOp) {
      return true;
    }

    if (!startingOp || visitedOps.contains(startingOp)) {
      return false;
    }

    visitedOps.insert(startingOp);

    for (auto userOp : startingOp->getUsers()) {
      if (isFindUserOpImpl(userOp, targetOp, visitedOps)) {
        return true;
      }
    }

    return false;
  }

  bool isFindUserOp(Operation *startingOp, Operation *targetOp) {
    llvm::SetVector<Operation *> visitedOps;
    return isFindUserOpImpl(startingOp, targetOp, visitedOps);
  }

  // Block Access Optimization Degradation
  // If the loadOp/storeOp .len() is not computed by subIOp, the block
  // read/write optimizations cannot be applied.
  void blockAccessOptDeGrad(mlir::ModuleOp m) {
    m.walk([&](triton::xpu::GM2LMOp gm2lmOp) {
      auto len = gm2lmOp.getLen();
      if (len) {
        auto subIOp = len.getDefiningOp<mlir::arith::SubIOp>();
        if (!subIOp) {
          OpBuilder builder(gm2lmOp);
          auto loc = gm2lmOp->getLoc();
          gm2lmOp->setAttr("offsetState",
                           builder.getSI32IntegerAttr(
                               static_cast<int32_t>(OffsetState::Unknown)));
        }
      }
    });

    m.walk([&](triton::xpu::LM2GMOp lm2gmOp) {
      auto len = lm2gmOp.getLen();
      if (len) {
        auto subIOp = len.getDefiningOp<mlir::arith::SubIOp>();
        if (!subIOp) {
          OpBuilder builder(lm2gmOp);
          auto loc = lm2gmOp->getLoc();
          lm2gmOp->setAttr("offsetState",
                           builder.getSI32IntegerAttr(
                               static_cast<int32_t>(OffsetState::Unknown)));
        }
      }
    });

    m.walk([&](triton::xpu::SM2GMOp sm2gmOp) {
      auto len = sm2gmOp.getLen();
      if (len) {
        auto subIOp = len.getDefiningOp<mlir::arith::SubIOp>();
        if (!subIOp) {
          OpBuilder builder(sm2gmOp);
          auto loc = sm2gmOp->getLoc();
          sm2gmOp->setAttr("offsetState",
                           builder.getSI32IntegerAttr(
                               static_cast<int32_t>(OffsetState::Unknown)));
        }
      }
    });
  }

  void addThreadIdMask(mlir::scf::IfOp ifOp, triton::xpu::LoadOp loadOp) {

    auto resTensorType = mlir::cast<RankedTensorType>(loadOp.getType());

    OpBuilder builder(ifOp);
    auto loc = ifOp->getLoc();

    // Step 1. get val
    auto rcvVal = builder.create<mlir::triton::xpu::ExtractOp>(
        loc, builder.getI64Type(), builder.getI32IntegerAttr(0), loadOp);
    auto rcvValI32 = builder.create<mlir::arith::TruncIOp>(
        loc, builder.getI32Type(), rcvVal);

    // Step 2. get ThreadNum
    auto clusterNum = builder.create<mlir::gpu::GridDimOp>(
        loc, builder.getIndexType(), mlir::gpu::Dimension::x);
    auto clusterNum_cast = builder.create<mlir::arith::IndexCastOp>(
        loc, builder.getI32Type(), clusterNum);
    auto coreNum = builder.create<mlir::gpu::BlockDimOp>(
        loc, builder.getIndexType(), mlir::gpu::Dimension::x);
    auto coreNum_cast = builder.create<mlir::arith::IndexCastOp>(
        loc, builder.getI32Type(), coreNum);
    auto threadNum = builder.create<mlir::arith::MulIOp>(
        loc, builder.getI32Type(), clusterNum_cast, coreNum_cast);

    // Step 3. val % ThreadNum
    auto remFOp =
        builder.create<mlir::arith::RemSIOp>(loc, rcvValI32, threadNum);

    // Step 4. get ThreadId
    auto threadIdOp = builder.create<mlir::triton::xpu::GetThreadIdOp>(
        loc, builder.getI32Type(), builder.getSI32IntegerAttr(1));

    // Step 5. threadId == val % ThreadNum
    auto threadCond = builder.create<mlir::arith::CmpIOp>(
        loc, builder.getI1Type(), mlir::arith::CmpIPredicate::eq, remFOp,
        threadIdOp);

    auto originCondOp = ifOp.getCondition();

    auto newCondOp = builder.create<mlir::arith::AndIOp>(
        loc, builder.getI1Type(), originCondOp, threadCond);

    ifOp->setOperand(0, newCondOp);
  }

  void atoNaiveMask(mlir::ModuleOp m) {
    m.walk([&](scf::IfOp ifOp) {
      OpBuilder builder(ifOp);
      auto loc = ifOp->getLoc();

      auto threadIdOp = builder.create<mlir::triton::xpu::GetCoreIdOp>(
          loc, builder.getI32Type());

      auto core0Op = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);

      auto atomicCondOp = builder.create<mlir::arith::CmpIOp>(
          loc, builder.getI1Type(), mlir::arith::CmpIPredicate::eq, core0Op,
          threadIdOp);

      auto originCondOp = ifOp.getCondition();

      auto newCondOp = builder.create<mlir::arith::AndIOp>(
          loc, builder.getI1Type(), originCondOp, atomicCondOp);

      ifOp->setOperand(0, newCondOp);
    });
  }

  void atoOptimizationMask(mlir::ModuleOp m) {
    /*********************** Atomic Mask Opt **************************
    * %cst_0 = arith.constant dense<-1> : tensor<2048xi64>
    * %cst_1 = arith.constant dense<0.000000e+00> : tensor<2048xf32>
    *
    * Before Optimization:
    * %1 = tt.load: tensor<2048xi64>
    * %2 = tt.load: tensor<2048xf32>
    * %3 = arith.cmpi eq, %1, %cst_0 -> tensor<2048xi1>
    * %4 = arith.select %3, %cst_1, %2 -> tensor<2048xf32>,
    * %5 = tt.atomic_add %4 -> tensor<2048xf32>

    * After Optimization:
    * %1 = tt.load: tensor<2048xi64>
    * %3 = arith.cmpi eq, %1, %cst_0 -> tensor<2048xi1>
    * %4 = arith.neg %3        ------>     negativeCond
    * scf.if %4 {
    *   %2 = tt.load: tensor<2048xf32>
    *   %5 = tt.atomic_add %4 -> tensor<2048xf32>
    * }
    *
    *****************************************************************/
    arith::CmpIOp rcvCmpIOp;
    triton::xpu::LoadOp rcvLoadOp;
    arith::SelectOp rcvSelectOp;
    AtomicMaskCond atoMaskCond = AtomicMaskCond::NonActivate;
    bool atoMaskOpt = false;

    m.walk([&](arith::CmpIOp cmpIOp) {
      if (cmpIOp.getPredicate() == arith::CmpIPredicate::eq) {
        auto lhs = cmpIOp.getLhs();
        auto rhs = cmpIOp.getRhs();
        auto res = cmpIOp.getResult();

        // Assert Only Have One SelectOp User
        auto cmpiop_user_begin = cmpIOp->user_begin();
        auto cmpiop_user_end = cmpIOp->user_end();
        if (std::distance(cmpiop_user_begin, cmpiop_user_end) != 1)
          return;

        if (auto selectOp =
                dyn_cast<mlir::arith::SelectOp>(*cmpiop_user_begin)) {

          // Assert Only Have One AddFOp User
          auto selectop_user_begin = selectOp->user_begin();
          auto selectop_user_end = selectOp->user_end();
          if (std::distance(selectop_user_begin, selectop_user_end) != 1)
            return;

          if (auto addFOp = dyn_cast<arith::AddFOp>(*selectop_user_begin)) {
            auto trueVal = selectOp.getTrueValue();
            auto falseVal = selectOp.getFalseValue();

            auto trueLoadOp =
                dyn_cast<triton::xpu::LoadOp>(trueVal.getDefiningOp());
            auto falseConstOp =
                dyn_cast<arith::ConstantOp>(falseVal.getDefiningOp());

            auto trueConstOp =
                dyn_cast<arith::ConstantOp>(trueVal.getDefiningOp());
            auto falseLoadOp =
                dyn_cast<triton::xpu::LoadOp>(falseVal.getDefiningOp());

            if (trueLoadOp && falseConstOp) {
              if (auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(
                      falseConstOp.getValue())) {
                float constValue = *denseAttr.getValues<float>().begin();
                if (constValue == 0.0f)
                  atoMaskCond = AtomicMaskCond::PostiveCond;
              }
            } else if (trueConstOp && falseLoadOp) {
              if (auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(
                      trueConstOp.getValue())) {
                float constValue = *denseAttr.getValues<float>().begin();
                if (constValue == 0.0f)
                  atoMaskCond = AtomicMaskCond::NegativeCond;
              }
            }

            auto lhsLoadOp = dyn_cast<triton::xpu::LoadOp>(lhs.getDefiningOp());
            auto rhsConstOp = dyn_cast<arith::ConstantOp>(rhs.getDefiningOp());

            auto lhsConstOp = dyn_cast<arith::ConstantOp>(lhs.getDefiningOp());
            auto rhsLoadOp = dyn_cast<triton::xpu::LoadOp>(rhs.getDefiningOp());

            if ((lhsLoadOp && rhsConstOp) || (lhsConstOp && rhsLoadOp)) {
              rcvCmpIOp = cmpIOp;
              rcvLoadOp = lhsLoadOp ? lhsLoadOp : rhsLoadOp;
              rcvSelectOp = selectOp;
              atoMaskOpt = true;
            }
          }
        }
      }
    });

    // Step 1. Create If
    if (rcvCmpIOp && rcvLoadOp && atoMaskOpt &&
        atoMaskCond != AtomicMaskCond::NonActivate) {
      Operation *nextOp = rcvLoadOp->getNextNode();
      // Move the cmpi operation right after the load operation
      OpBuilder builder(rcvLoadOp);
      auto loc = rcvLoadOp->getLoc();
      builder.setInsertionPointAfter(rcvLoadOp);

      auto newCmpiOp = builder.create<mlir::arith::CmpIOp>(
          loc, rcvCmpIOp.getType(), mlir::arith::CmpIPredicate::eq,
          rcvCmpIOp.getLhs(), rcvCmpIOp.getRhs());
      rcvCmpIOp.replaceAllUsesWith(newCmpiOp.getResult());
      rcvCmpIOp.erase();

      auto trueValue = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getI1Type(),
          builder.getIntegerAttr(builder.getI1Type(), 1));

      auto posCond = builder.create<mlir::triton::xpu::ExtractOp>(
          loc, builder.getI1Type(), builder.getI32IntegerAttr(0), newCmpiOp);

      auto negCond =
          builder.create<mlir::arith::XOrIOp>(loc, trueValue, posCond);

      // Create the SCF if operation
      scf::IfOp scfIfOp;
      if (atoMaskCond == AtomicMaskCond::PostiveCond) {
        scfIfOp = builder.create<mlir::scf::IfOp>(loc, posCond,
                                                  /*withElseRegion=*/false);
      } else if (atoMaskCond == AtomicMaskCond::NegativeCond) {
        scfIfOp = builder.create<mlir::scf::IfOp>(loc, negCond,
                                                  /*withElseRegion=*/false);
      }

      addThreadIdMask(scfIfOp, rcvLoadOp);

      // Move subsequent operations inside the ifOp's then block
      builder.setInsertionPointToStart(&scfIfOp.getThenRegion().front());
      Operation *yieldOp = scfIfOp.getThenRegion().front().getTerminator();
      while (nextOp) {
        Operation *currentOp = nextOp;
        nextOp = nextOp->getNextNode();
        if (!isa<scf::YieldOp>(currentOp)) {
          currentOp->moveBefore(yieldOp);
        }
      }
    }

    // Step 2. Eliminate Unvalid SelectOp
    if (rcvCmpIOp && rcvLoadOp && atoMaskOpt &&
        atoMaskCond != AtomicMaskCond::NonActivate) {
      auto true_value = rcvSelectOp.getTrueValue();
      auto false_value = rcvSelectOp.getFalseValue();

      if (atoMaskCond == AtomicMaskCond::PostiveCond) {
        rcvSelectOp.replaceAllUsesWith(true_value);
      } else if (atoMaskCond == AtomicMaskCond::NegativeCond) {
        rcvSelectOp.replaceAllUsesWith(false_value);
      }
      rcvSelectOp->erase();
    }
  }

  bool isOptMask(mlir::ModuleOp m) {
    bool isOpt = false;
    m.walk([&](triton::xpu::LM2GMOp lm2gmOp) {
      auto op = findDefOpBwd<mlir::arith::SelectOp>(lm2gmOp.getValue());
      if (op) {
        auto selectOp = cast<mlir::arith::SelectOp>(op);
        if (auto cmpIOp = dyn_cast<mlir::arith::CmpIOp>(
                selectOp.getCondition().getDefiningOp())) {
          if (cmpIOp.getPredicate() == arith::CmpIPredicate::eq) {
            isOpt = true;
          }
        }
      }
    });
    return isOpt;
  }

  // Add AtomicOp Simulation Condition
  // We must use the specifiy core to avoid access race
  void addAtomicSimulationCond(mlir::ModuleOp m) {
    bool atomicSim = false;
    m->walk([&](triton::xpu::GM2LMOp gm2lmOp) {
      auto tensorTy = gm2lmOp.getResult().getType();
      if (auto rankTensorTy = mlir::dyn_cast<RankedTensorType>(tensorTy)) {
        auto gEncoding = mlir::cast<triton::xpu::ClusterLayoutAttr>(
            rankTensorTy.getEncoding());
        auto coresPerGroup = gEncoding.getCoresPerGroup();
        auto groupsPerCluster = gEncoding.getGroupsPerCluster();

        atomicSim = (llvm::find_if(coresPerGroup,
                                   [](unsigned int num) { return num != 1; }) ==
                     coresPerGroup.end()) &&
                    (llvm::find_if(groupsPerCluster, [](unsigned int num) {
                       return num != 1;
                     }) == groupsPerCluster.end());

        if (findUserOp<triton::ReduceOp>(gm2lmOp) ||
            findUserOp<triton::xpu::ReduceOp>(gm2lmOp)) {
          atomicSim = false;
        }
      }
    });

    if (!atomicSim)
      return;

    AtomicMaskType atoMaskTy = AtomicMaskType::NaiveMask;
    if (maskValue != -1 && isOptMask(m)) {
      atoMaskTy = AtomicMaskType::OptimizationMask;
    }

    switch (atoMaskTy) {
    case AtomicMaskType::NaiveMask: {
      atoNaiveMask(m);
      break;
    }
    case AtomicMaskType::OptimizationMask: {
      atoOptimizationMask(m);
      break;
    }
    default:
      llvm_unreachable("Unknown Atomic Mask Type");
    }
  }

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();

    // Check Core Tiling Optimization
    // TODO[dyq]: Open core tiling pass
    // m.walk([&](triton::ReduceOp redOp) {
    //   isReduceOpt = isReduceOptimized(redOp.operand().getType());

    //   if (auto reduceSrcTy =
    //           redOp.operand().getType().dyn_cast<RankedTensorType>()) {
    //     auto sizePerCore = reduceSrcTy.getEncoding()
    //                            .cast<triton::xpu::GlobalEncodingAttr>()
    //                            .getSizePerCore();
    //     rowsPerCore = sizePerCore[0];
    //   }
    // });

    // Step 0. Convert CmpOp+SplatOp to SplatOp+CmpOp
    m.walk([&](triton::SplatOp splatOp) {
      if (auto splatDefOp = splatOp.getSrc().getDefiningOp()) {
        if (auto cmpiOp = dyn_cast<arith::CmpIOp>(splatDefOp)) {
          OpBuilder builder(splatOp);
          auto loc = cmpiOp->getLoc();
          auto lhs = cmpiOp.getLhs();
          auto rhs = cmpiOp.getRhs();
          auto lhsElemTy = lhs.getType();
          if (auto lhsTensorTy = dyn_cast<RankedTensorType>(lhs.getType())) {
            lhsElemTy = lhsTensorTy.getElementType();
          }
          auto resTy = cast<RankedTensorType>(splatOp.getType());
          auto newSplatTy = RankedTensorType::get(resTy.getShape(), lhsElemTy,
                                                  resTy.getEncoding());
          auto lhsSplatOp =
              builder.create<triton::SplatOp>(loc, newSplatTy, lhs);
          auto rhsSplatOp =
              builder.create<triton::SplatOp>(loc, newSplatTy, rhs);
          auto newCmpiOp = builder.create<arith::CmpIOp>(
              loc, resTy, cmpiOp.getPredicate(), lhsSplatOp, rhsSplatOp);
          splatOp.replaceAllUsesWith(newCmpiOp.getResult());
        }
      }
    });

    // Step 1. Replace CmiOp with SubOp(replace mask with len)
    llvm::DenseMap<Operation *, llvm::SetVector<Operation *>> maskUsersMap;
    llvm::SetVector<Operation *> cmpiOps;

    m.walk([&](mlir::arith::CmpIOp cmpiOp) {
      llvm::SetVector<Operation *> maskUsers;
      llvm::SetVector<Operation *> cmpiOpUsers;
      if (cmpiOp.getPredicate() == arith::CmpIPredicate::slt ||
          cmpiOp.getPredicate() == arith::CmpIPredicate::ult) {
        getOpChain(cmpiOpUsers, cmpiOp);
        for (auto user : cmpiOpUsers) {
          Value mask;
          if (auto loadOp = dyn_cast<triton::xpu::GM2LMOp>(user)) {
            mask = loadOp.getLen();
          }
          if (auto storeOp = dyn_cast<triton::xpu::LM2GMOp>(user)) {
            mask = storeOp.getLen();
          }
          if (auto storeOp = dyn_cast<triton::xpu::SM2GMOp>(user)) {
            mask = storeOp.getLen();
          }
          if (mask) {
            if (mask == cmpiOp.getResult()) {
              cmpiOps.insert(cmpiOp);
              maskUsers.insert(user);
            }
            if (auto andIOp =
                    dyn_cast<mlir::arith::AndIOp>(mask.getDefiningOp())) {
              if (isFindUserOp(cmpiOp, andIOp.getLhs().getDefiningOp())) {
                if (auto cmpiOpDef =
                        findDefOpBwd<triton::ExpandDimsOp>(andIOp.getLhs())) {
                  cmpiOps.insert(cmpiOp);
                  auto expandDimOp = cast<triton::ExpandDimsOp>(cmpiOpDef);
                  //  expand axis = 0 means inner dim
                  if (expandDimOp.getAxis() == 0) {
                    maskUsers.insert(user);
                  }
                }
              } else if (isFindUserOp(cmpiOp,
                                      andIOp.getRhs().getDefiningOp())) {
                if (auto cmpiOpDef =
                        findDefOpBwd<triton::ExpandDimsOp>(andIOp.getRhs())) {
                  cmpiOps.insert(cmpiOp);
                  auto expandDimOp = cast<triton::ExpandDimsOp>(cmpiOpDef);
                  //  expand axis = 0 means inner dim
                  if (expandDimOp.getAxis() == 0) {
                    maskUsers.insert(user);
                  }
                }
              }
            }
          }
        }
        if (!maskUsers.empty()) {
          maskUsersMap[cmpiOp] = maskUsers;
        }
      }
    });

    // Step 2 Replace the mask of LoadOp/StoreOp
    for (const auto &pair : maskUsersMap) {
      Value rhs;
      Value lhs;
      auto op = pair.first;
      auto users = pair.second;

      if (auto cmpiOp = dyn_cast<mlir::arith::CmpIOp>(op)) {
        rhs = cmpiOp.getRhs();
        lhs = cmpiOp.getLhs();
      } else {
        llvm_unreachable(
            "cmpiOp only is mlir::arith::CmpIOp/triton::gpu::CmpIOp");
      }

      OpBuilder builder(op);
      auto loc = op->getLoc();
      // TODO[dyq]: Open coretiling pass
      //   if (isReduceOpt && rowsPerCore != 1) {
      //     auto resTensorType = rhs.getType().cast<RankedTensorType>();
      //     SmallVector<Attribute, 4> values(
      //         resTensorType.getNumElements(),
      //         builder.getI32IntegerAttr(rowsPerCore));
      //     auto denseValues = DenseElementsAttr::get(resTensorType, values);
      //     auto rowsPerCoreValue =
      //         builder.create<arith::ConstantOp>(loc, denseValues);
      //     rhs = builder.create<mlir::arith::MulIOp>(loc, rhs,
      //     rowsPerCoreValue);
      //   }

      auto elemLenOp =
          builder.create<mlir::arith::SubIOp>(loc, rhs.getType(), rhs, lhs);

      // get maskValue from subIOp.rhs()
      if (rhs.getDefiningOp()) {
        if (auto rMaskConstOp =
                dyn_cast<arith::ConstantOp>(rhs.getDefiningOp())) {
          auto resTy = rMaskConstOp.getResult().getType();
          if (auto tensorType = mlir::dyn_cast<RankedTensorType>(resTy)) {
            if (auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(
                    rMaskConstOp.getValue())) {
              auto values = denseAttr.getValues<mlir::APInt>();
              if (!values.empty()) {
                maskValue = values[0].getZExtValue();
              }
            }
          }
        }
      }

      for (auto user : users) {
        if (auto loadOp = dyn_cast<triton::xpu::GM2LMOp>(user)) {
          loadOp.setOperand(1, elemLenOp.getResult());
        } else if (auto storeOp = dyn_cast<triton::xpu::LM2GMOp>(user)) {
          storeOp.setOperand(2, elemLenOp.getResult());
        } else if (auto storeOp = dyn_cast<triton::xpu::SM2GMOp>(user)) {
          storeOp.setOperand(1, elemLenOp.getResult());
        }
      }
    }

    // Step 3. Create scf::IfOp to deal with tailing
    cmpiOps = multiRootTopologicalSort(cmpiOps);
    llvm::SmallVector<Operation *> sortedCmpiOps;
    for (auto it = cmpiOps.begin(); it != cmpiOps.end(); ++it) {
      sortedCmpiOps.emplace_back(*it);
    }
    llvm::SmallVector<llvm::SetVector<Operation *>> ifBlockTrees;
    for (int i = 0; i < sortedCmpiOps.size(); ++i) {
      Operation *op = sortedCmpiOps[i];
      OpBuilder builder(op);
      mlir::Block *block = op->getBlock();
      auto loc = builder.getUnknownLoc();
      // ops to be moved into ifblock and later earsed
      llvm::SetVector<Operation *> opsToMoveAndErase;
      // Get the ops that from current to the end of the block
      mlir::Operation *terminator = block->getTerminator();
      auto it = op->getIterator();
      ++it;
      for (; &*it != terminator; ++it) {
        opsToMoveAndErase.insert(&*it);
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
        if (yieldOp.getResults().size() > 0) {
          opsToMoveAndErase.insert(terminator);
        }
      }
      auto sortedOpsToMoveAndErase = sortOpTreeBwd(opsToMoveAndErase);
      ifBlockTrees.emplace_back(sortedOpsToMoveAndErase);
      // Create scf::IfOp
      builder.setInsertionPoint(terminator);
      mlir::scf::IfOp newIfOp;
      if (auto cmpiOp = dyn_cast<mlir::arith::CmpIOp>(op)) {
        auto cond = builder.create<triton::xpu::ExtractOp>(
            loc, builder.getI1Type(), builder.getI32IntegerAttr(0),
            cmpiOp.getResult());
        if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
          if (yieldOp.getResults().size() > 0) {
            newIfOp = builder.create<mlir::scf::IfOp>(
                loc, yieldOp.getResults().getType(), cond,
                /*withElseRegion=*/true);
            builder.setInsertionPointToStart(newIfOp.thenBlock());
          } else {
            newIfOp = builder.create<mlir::scf::IfOp>(loc, cond,
                                                      /*withElseRegion=*/false);
            builder.setInsertionPointToStart(newIfOp.thenBlock());
          }
        } else if (auto funcRetureOp = dyn_cast<triton::ReturnOp>(terminator)) {
          newIfOp = builder.create<mlir::scf::IfOp>(loc, cond,
                                                    /*withElseRegion=*/false);
          builder.setInsertionPointToStart(newIfOp.thenBlock());
        }
      } else {
        llvm_unreachable("cmpiOp only is mlir::arith::CmpIOp");
      }

      mlir::IRMapping mapping;
      for (int j = sortedOpsToMoveAndErase.size() - 1; j >= 0; --j) {
        auto bodyOp = sortedOpsToMoveAndErase[j];
        auto newBodyOp = builder.clone(*bodyOp, mapping); // Clone bodyOps
        // TODO[dyq]: Open core tiling pass
        // if (auto reduceOp = dyn_cast<triton::ReduceOp>(bodyOp)) {
        //   // for shared memory init
        //   auto newReduceOp = cast<triton::ReduceOp>(newBodyOp);
        //   ReduceOpHelper helper(reduceOp);
        //   ReduceOpHelper newHelper(newReduceOp);
        //   newHelper.setReduceId(helper.getReduceId());
        //   newHelper.setOriginResShape(helper.getOriginResShape());
        // } else
        if (auto storeOp = dyn_cast<triton::xpu::StoreSMOp>(bodyOp)) {
          SMHelper helper(storeOp);
          SMHelper newHelper(newBodyOp);
          newHelper.setOffset(helper.getOffset());
        } else if (auto sm2gmOp = dyn_cast<triton::xpu::SM2GMOp>(bodyOp)) {
          SMHelper helper(sm2gmOp);
          SMHelper newHelper(newBodyOp);
          newHelper.setOffset(helper.getOffset());
        }
      }

      if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
        if (yieldOp.getResults().size() > 0) {
          builder.setInsertionPointToStart(newIfOp.elseBlock());
          auto block = yieldOp->getBlock();
          auto *region = block->getParent();
          auto *parentOp = region->getParentOp();
          if (auto parentForOp = dyn_cast<scf::ForOp>(parentOp)) {
            // Create YiledOp For newIfOp
            if (parentForOp.getRegionIterArgs().size() > 0) {
              builder.create<mlir::scf::YieldOp>(
                  loc, parentForOp.getRegionIterArgs());
            }
            // Create YiledOp For YieldOp's ParentOp
            builder.setInsertionPointToEnd(parentForOp.getBody());
            auto ifResults = newIfOp.getResults();
            builder.create<mlir::scf::YieldOp>(loc, ifResults);
          } else if (auto parentIfOp = dyn_cast<scf::IfOp>(parentOp)) {
            // Create YiledOp For newIfOp
            auto elseYieldOp = parentIfOp.elseYield();
            auto elseResults = elseYieldOp.getResults();
            builder.create<mlir::scf::YieldOp>(loc, elseResults);
            // Create YiledOp For YieldOp's ParentOp
            builder.setInsertionPointToEnd(parentIfOp.getBody());
            builder.create<mlir::scf::YieldOp>(loc, newIfOp.getResults());
          } else {
            llvm_unreachable("Unknown Mask YiledOp Pattern");
          }
        }
      }

      // update next iteration of sortedCmpiOps
      for (int k = i + 1; k < sortedCmpiOps.size(); ++k) {
        auto mappedMaskVal =
            mapping.lookupOrDefault(sortedCmpiOps[k]->getResult(0));
        sortedCmpiOps[k] = mappedMaskVal.getDefiningOp();
      }
      // Erase Old Ops
      for (auto op : sortedOpsToMoveAndErase) {
        if (op->getParentOp() != nullptr) {
          op->erase();
        }
      }
    }

    // Step 3. Block Read/Write Optimizations Degradation
    blockAccessOptDeGrad(m);

    // Step 4. Add AtomicOp Simulation Conditon
    addAtomicSimulationCond(m);
  }

private:
  int32_t maskValue = -1;
};

} // namespace xpu
} // namespace triton
} // namespace mlir
