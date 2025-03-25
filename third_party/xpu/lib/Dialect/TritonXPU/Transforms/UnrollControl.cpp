//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#define DEBUG_TYPE "tritonxpu-unroll-control"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUUNROLLCONTROL
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

#define COMBINE_OP                                                             \
  arith::AddFOp, arith::MulFOp, arith::MaxNumFOp, arith::MinNumFOp,            \
      arith::OrIOp, arith::XOrIOp, arith::AndIOp

template <typename OP> struct COMOp;

#define COMOP(SrcType, DstType)                                                \
  template <> struct COMOp<SrcType> {                                          \
    typedef DstType type;                                                      \
  };

COMOP(arith::AddFOp, triton::xpu::VvaddFOp);
COMOP(arith::MulFOp, triton::xpu::VvmulFOp);
COMOP(arith::MaxNumFOp, triton::xpu::VvmaxNumFOp);
COMOP(arith::MinNumFOp, triton::xpu::VvminNumFOp);
COMOP(arith::OrIOp, triton::xpu::VvorIOp);
COMOP(arith::XOrIOp, triton::xpu::VvxorIOp);
COMOP(arith::AndIOp, triton::xpu::VvandIOp);

struct TritonXPUUnrollControl
    : public impl::TritonXPUUnrollControlBase<TritonXPUUnrollControl> {

public:
  using impl::TritonXPUUnrollControlBase<
      TritonXPUUnrollControl>::TritonXPUUnrollControlBase;

  template <typename T> static decltype(auto) createCombineVectorizedOp(T op) {
    OpBuilder builder(op);
    return builder.create<typename COMOp<T>::type>(
        op.getLoc(), op.getResult().getType(), op.getLhs(), op.getRhs());
  }

  void processOpVecTy(ModuleOp &m) {
    m.walk([&](Operation *op) {
      TypeSwitch<Operation *>(op).Case<COMBINE_OP>([&](auto combineOp) {
        if (auto tensorTy =
                dyn_cast<RankedTensorType>(combineOp.getResult().getType())) {
          if (isa<VectorType>(getElementTypeOrSelf(tensorTy))) {
            auto vecOp = createCombineVectorizedOp(combineOp);
            combineOp.replaceAllUsesWith(vecOp.getResult());
            combineOp.erase();
          }
        }
      });
    });
  }

  bool isAncestorOf(Operation *op1, Operation *op2) {
    Block *block1 = op1->getBlock();
    for (Block *block2 = op2->getBlock(); block2 != nullptr;) {
      if (block1 == block2) {
        return true;
      }
      Operation *parentOp = block2->getParentOp();
      if (parentOp == nullptr) {
        break;
      }
      block2 = parentOp->getBlock();
    }
    return false;
  }

  void getUnrollTree(Operation *op, SetVector<Operation *> &opTree,
                     SetVector<Operation *> &visitedOps,
                     SetVector<Operation *> &excludeChainOps, Operation *rootOp,
                     bool isTop2Bottom = true) {
    if (!op || visitedOps.count(op) ||
        isa<triton::xpu::GM2LMOp, triton::xpu::LM2GMOp, scf::YieldOp,
            triton::xpu::ReduceOp, triton::xpu::ReduceReturnOp>(op)) {
      return;
    }

    visitedOps.insert(op);
    if (isAncestorOf(op, rootOp) || op->getBlock() == rootOp->getBlock()) {
      opTree.insert(op);
    }

    // Search definedOp of childOp
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // Then
      auto &ifThenBlock = ifOp.getThenRegion().front();
      for (auto &inBlockOp : ifThenBlock) {
        getUnrollTree(&inBlockOp, opTree, visitedOps, excludeChainOps, rootOp,
                      isTop2Bottom);
      }
      // Else
      auto &ifElseRegion = ifOp.getElseRegion();
      if (!ifElseRegion.empty()) {
        auto &ifElseBlock = ifElseRegion.front();
        for (auto &inBlockOp : ifElseBlock) {
          getUnrollTree(&inBlockOp, opTree, visitedOps, excludeChainOps, rootOp,
                        isTop2Bottom);
        }
      }
    }

    // from bottom to top
    if (isa<triton::xpu::LoadOp, arith::ConstantOp, triton::xpu::VConstOp>(
            op)) {
    } else if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(op)) {
      auto defOp = storeOp.getValue().getDefiningOp();
      getUnrollTree(defOp, opTree, visitedOps, excludeChainOps, rootOp,
                    isTop2Bottom);
    } else {
      for (auto operand : op->getOperands()) {
        auto defOp = operand.getDefiningOp();
        getUnrollTree(defOp, opTree, visitedOps, excludeChainOps, rootOp,
                      isTop2Bottom);
      }
    }

    if (isTop2Bottom) {
      // from top to bottom
      if (excludeChainOps.count(op) ||
          isa<arith::ConstantOp, triton::xpu::ConvertLayoutOp,
              triton::xpu::VConstOp>(op)) {
      } else {
        for (auto userOp : op->getUsers()) {
          getUnrollTree(userOp, opTree, visitedOps, excludeChainOps, rootOp,
                        isTop2Bottom);
        }
      }
    }
    return;
  }

  int64_t getNumCol(Type type) {
    if (auto tensorTy = dyn_cast<RankedTensorType>(type))
      return tensorTy.getShape().back();
    else
      return 1;
  }

  int64_t getNumInVector(Type type) {
    if (auto vecType = dyn_cast<VectorType>(type))
      return vecType.getNumElements();
    else
      return 1;
  }

  int64_t getnumUnroll(Type type) {
    int64_t numUnroll = numUnrollPerCore * 64;
    if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
      auto clusterEncoding =
          cast<triton::xpu::ClusterLayoutAttr>(tensorTy.getEncoding());
      numUnroll = numUnrollPerCore * clusterEncoding.getCoresPerGroup().back();
    }
    return numUnroll;
  }

  Type createPointerType(Type type, int64_t vecSize) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      Type elemType = getElementTypeOrSelf(tensorType);
      Type elemScalarType = getElementTypeOrSelf(elemType);
      Type pointerType = triton::PointerType::get(elemScalarType, 0);
      auto shape = tensorType.getShape().vec();
      shape[shape.size() - 1] = shape.back() * vecSize;
      return RankedTensorType::get(shape, pointerType,
                                   tensorType.getEncoding());
    } else {
      return triton::PointerType::get(type, 0);
    }
  }

  triton::xpu::ClusterLayoutAttr
  createEncoding(MLIRContext *context, triton::xpu::ClusterLayoutAttr &encoding,
                 int64_t iterNum) const {
    auto sizePerCore = encoding.getSizePerCore().vec();
    sizePerCore[sizePerCore.size() - 1] =
        ceil<int64_t>(sizePerCore.back(), iterNum);
    auto newEncoding = triton::xpu::ClusterLayoutAttr::get(
        context, sizePerCore, encoding.getCoresPerGroup(),
        encoding.getGroupsPerCluster(), encoding.getOrder(),
        encoding.getIsReduceOpt());
    return newEncoding;
  }

  void setTensorType(MLIRContext *context, Operation *op, int64_t iterNum,
                     bool isOuter, bool sliceShape = true) const {
    for (auto [i, resTy] : llvm::enumerate(op->getResultTypes())) {
      if (isa<RankedTensorType>(resTy) && !isOuter) {
        auto tensorTy = cast<RankedTensorType>(resTy);
        auto shape = tensorTy.getShape().vec();
        if (sliceShape) {
          shape[shape.size() - 1] = ceil<int64_t>(shape.back(), iterNum);
        }
        RankedTensorType controledTensorTy;
        if (auto sliceEncoding = dyn_cast<triton::gpu::SliceEncodingAttr>(
                tensorTy.getEncoding())) {
          auto clusterEncoding =
              cast<triton::xpu::ClusterLayoutAttr>(sliceEncoding.getParent());
          auto newClusterEncoding =
              createEncoding(context, clusterEncoding, iterNum);
          auto newEncoding = triton::gpu::SliceEncodingAttr::get(
              context, sliceEncoding.getDim(), newClusterEncoding);
          controledTensorTy = RankedTensorType::get(
              shape, tensorTy.getElementType(), newEncoding);
        } else {
          auto clusterEncoding =
              cast<triton::xpu::ClusterLayoutAttr>(tensorTy.getEncoding());
          auto newClusterEncoding =
              createEncoding(context, clusterEncoding, iterNum);
          controledTensorTy = RankedTensorType::get(
              shape, tensorTy.getElementType(), newClusterEncoding);
        }
        op->getResult(i).setType(controledTensorTy);
      }
    }
  }

  triton::xpu::ExtractSliceOp
  getExtractedOperand(MLIRContext *context, OpBuilder &builder, Location &loc,
                      mlir::Operation *op, unsigned operandIndex,
                      int64_t iterNum) const {
    auto resTy = op->getOperand(operandIndex).getType();
    RankedTensorType tensorTy;
    if (isa<RankedTensorType>(resTy)) {
      tensorTy = cast<RankedTensorType>(resTy);
    }
    auto shape = tensorTy.getShape().vec();
    shape[shape.size() - 1] = ceil<int64_t>(shape.back(), iterNum);
    auto clusterEncoding =
        cast<triton::xpu::ClusterLayoutAttr>(tensorTy.getEncoding());
    auto newClusterEncoding = createEncoding(context, clusterEncoding, iterNum);

    RankedTensorType controledTensorTy = RankedTensorType::get(
        shape, tensorTy.getElementType(), newClusterEncoding);
    triton::xpu::ExtractSliceOp extractSliceOp =
        builder.create<triton::xpu::ExtractSliceOp>(
            loc, controledTensorTy, op->getOperand(operandIndex));
    return extractSliceOp;
  }

  // Determine whether the operand has been hoisted
  bool isOperandOperationInSameForBlock(mlir::Operation *op,
                                        unsigned operandIndex) {
    auto *parentOp = op->getParentOp();
    while (parentOp && !llvm::isa<mlir::scf::ForOp>(parentOp)) {
      parentOp = parentOp->getParentOp();
    }
    if (!parentOp)
      return false;

    auto forOp = llvm::cast<mlir::scf::ForOp>(parentOp);
    mlir::Value operand = op->getOperand(operandIndex);
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
      mlir::Block *block = forOp.getBody()->front().getBlock();
      return blockArg.getOwner() == block;
    } else {
      mlir::Operation *definingOp = operand.getDefiningOp();
      if (definingOp) {
        return definingOp->getBlock()->getParentOp() == forOp.getOperation();
      }
    }
    return false;
  }

  void insertIndex(Operation *op, Value idxVar) {
    OpBuilder builder(op);
    auto operandSegmentSizesAttr =
        op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
    SmallVector<int, 4> operandSegmentSizes(
        operandSegmentSizesAttr.asArrayRef());
    // LoadOp: 0: ptr, 1: mask, 2: other, 3: index
    // StoreOp: 0: ptr, 1: value, 2: mask, 3: index
    // MakeRangeOp: 0: loopIndex, 1: unrollIndex
    // InterleaveOp: 0: loopIndex, 1: unrollIndex
    ++operandSegmentSizes[operandSegmentSizes.size() - 1];
    op->setAttr("operandSegmentSizes",
                builder.getDenseI32ArrayAttr(operandSegmentSizes));
    op->insertOperands(op->getNumOperands(), {idxVar});
  }

  void getOuterChain(llvm::SetVector<Operation *> &allOpTree,
                     llvm::SetVector<Operation *> &outerChain) {
    for (auto op : allOpTree) {
      if (auto expandDimOp = dyn_cast<triton::ExpandDimsOp>(op)) {
        auto src = expandDimOp.getSrc();
        auto result = expandDimOp.getResult();
        if (auto srcTy = dyn_cast<RankedTensorType>(src.getType())) {
          if (auto resTy = dyn_cast<RankedTensorType>(result.getType())) {
            if (expandDimOp.getAxis() == 1) {
              getOpChainBwd(outerChain, expandDimOp);
              outerChain.remove(expandDimOp);
            }
          }
        }
      }
      if (auto broadcastOp = dyn_cast<triton::xpu::BroadcastOp>(op)) {
        auto src = broadcastOp.getSrc();
        auto result = broadcastOp.getResult();
        if (auto srcTy = dyn_cast<RankedTensorType>(src.getType())) {
          if (auto resTy = dyn_cast<RankedTensorType>(result.getType())) {
            int64_t srcElemNum = 1;
            if (auto vecTy =
                    dyn_cast<VectorType>(getElementTypeOrSelf(srcTy))) {
              srcElemNum = vecTy.getNumElements();
            }
            int64_t resElemNum = 1;
            if (auto vecTy =
                    dyn_cast<VectorType>(getElementTypeOrSelf(resTy))) {
              resElemNum = vecTy.getNumElements();
            }
            auto srcShape = srcTy.getShape();
            auto resShape = resTy.getShape();
            int64_t srcInnerNum = srcElemNum * srcShape.back();
            int64_t resInnerNum = resElemNum * resShape.back();
            if (srcInnerNum != resInnerNum) { // unequal dim 1 shape means in
                                              // the inner axis op chain
              getOpChainBwd(outerChain, broadcastOp);
              outerChain.remove(broadcastOp);
            }
          }
        }
      }
    }
  }

  void
  getOuterChains(const SmallVector<llvm::SetVector<Operation *>> &allOpTrees,
                 SmallVector<llvm::SetVector<Operation *>> &outerChains) {
    for (auto allOpTree : allOpTrees) {
      SetVector<Operation *> outerChain;
      getOuterChain(allOpTree, outerChain);
      outerChains.emplace_back(outerChain);
    }
  }

  void getDAG(Operation *op, SetVector<Operation *> &visitedOps,
              SmallVector<SetVector<Operation *>> &unrollOpTrees,
              SetVector<Operation *> &excludeChainOps,
              bool isTop2Bottom = true) {
    SetVector<Operation *> opTree;
    getUnrollTree(op, opTree, visitedOps, excludeChainOps, op, isTop2Bottom);
    if (!opTree.empty()) {
      SetVector<Operation *> sortedOpTree = sortOpTree(opTree);
      unrollOpTrees.push_back(sortedOpTree);
    }
  }

  void createFor(OpBuilder &builder, Location &loc, int64_t start,
                 int64_t iterNum, scf::ForOp &forOp, arith::IndexCastOp &idxVar,
                 ValueRange &iterArgs) {
    auto lower = builder.create<arith::ConstantIndexOp>(loc, start);
    auto upper = builder.create<arith::ConstantIndexOp>(loc, iterNum);
    auto step = builder.create<arith::ConstantIndexOp>(loc, 1);
    forOp = builder.create<scf::ForOp>(loc, lower, upper, step, iterArgs);
    builder.setInsertionPointToStart(forOp.getBody());
    idxVar = builder.create<arith::IndexCastOp>(loc, builder.getI32Type(),
                                                forOp.getInductionVar());
  }

  void createLoopBody(MLIRContext *context, OpBuilder &builder, Location &loc,
                      int64_t iterNum, SetVector<Operation *> &unrollOpTree,
                      SetVector<Operation *> &outerChain,
                      arith::IndexCastOp &idxVar, IRMapping &mapping) {
    for (auto op : unrollOpTree) {
      bool isOuter = inOpChain(outerChain, op);
      auto newOp = builder.clone(*op, mapping);
      setTensorType(context, newOp, iterNum, isOuter);
      TypeSwitch<Operation *>(newOp)
          .Case<triton::xpu::LoadOp>([&](auto loadOp) {
            if (auto tensorTy =
                    dyn_cast<RankedTensorType>(loadOp.getPtr().getType())) {
              if (!loadOp.getSVOpt() && !loadOp.getIsDiscrete()) {
                insertIndex(newOp, idxVar);
              }
            }
          })
          .Case<triton::xpu::StoreOp>([&](auto storeOp) {
            if (auto tensorTy =
                    dyn_cast<RankedTensorType>(storeOp.getPtr().getType())) {
              insertIndex(newOp, idxVar);
            }
          })
          .Case<triton::xpu::MakeRangeOp>([&](auto makeRangeOp) {
            if (auto tensorTy =
                    dyn_cast<RankedTensorType>(op->getResults()[0].getType())) {
              insertIndex(newOp, idxVar);
            }
          })
          .Case<triton::xpu::InterleaveOp>([&](auto interleaveOp) {
            if (auto tensorTy =
                    dyn_cast<RankedTensorType>(op->getResults()[0].getType())) {
              insertIndex(newOp, idxVar);
            }
          })
          .Case<triton::AddPtrOp>([&](auto addPtrOp) {
            auto ptr = addPtrOp.getPtr();
            auto offset = addPtrOp.getOffset();
            if (ptr.getType() != offset.getType()) {
              auto extractOp = builder.create<triton::xpu::ExtractOp>(
                  loc, getElementTypeOrSelf(ptr), builder.getI32IntegerAttr(0),
                  ptr);
              auto splatOp = builder.create<triton::SplatOp>(loc, ptr.getType(),
                                                             extractOp);
              setTensorType(context, splatOp, iterNum, isOuter);
              addPtrOp.setOperand(0, splatOp);
              addPtrOp->moveAfter(splatOp);
            }
          })
          .Case<arith::ConstantOp>([&](auto constantOp) {
            auto value = constantOp.getValue();
            if (auto attr = dyn_cast<DenseElementsAttr>(value)) {
              value = DenseElementsAttr::getFromRawBuffer(
                  cast<ShapedType>(constantOp.getType()), attr.getRawData());
            }
            constantOp.setValueAttr(value);
          })
          .Case<scf::IfOp>([&](auto ifOp) {
            // Set IfOp's childOp Type(Then)
            auto &ifThenBlock = ifOp.getThenRegion().front();
            for (auto &inBlockOp : ifThenBlock) {
              setTensorType(context, &inBlockOp, iterNum, isOuter);
            }
            // Set IfOp's childOp Type(Else)
            auto &ifElseRegion = ifOp.getElseRegion();
            if (!ifElseRegion.empty()) {
              auto &ifElseBlock = ifElseRegion.front();
              for (auto &inBlockOp : ifElseBlock) {
                setTensorType(context, &inBlockOp, iterNum, isOuter);
              }
            }
          });
      if (scf::IfOp ifOp = dyn_cast<scf::IfOp>(newOp)) {
        auto &ifThenBlock = ifOp.getThenRegion().front();
        for (auto &inBlockOp : ifThenBlock) {
          unsigned numifOpResults = ifOp.getNumResults();
          if (auto yieldOp = llvm::dyn_cast<scf::YieldOp>(&inBlockOp)) {
            // 1. needExtract denotes Extraction is required if the operand of
            // YieldOp does not match the type expected by the result of IfOp
            // 2. isSame denotes whether the operand of YieldOp is in the same
            // ForBlock as IfOp.
            unsigned numyieldOpOperands = yieldOp.getNumOperands();
            assert(
                (numifOpResults == numyieldOpOperands) &&
                "The number of IfOp results and YieldOp operands must match.");
            for (unsigned i = 0; i < numyieldOpOperands; ++i) {
              bool needExtract = false;
              bool isSame = true;
              Value result = ifOp.getResult(i);
              Type resultType = result.getType();
              Value operand = yieldOp.getOperand(i);
              Type operandType = operand.getType();
              if (resultType != operandType) {
                needExtract = true;
              }
              isSame = isOperandOperationInSameForBlock(&inBlockOp, i);
              if (!isSame && needExtract) {
                assert(isa<arith::ConstantOp>(
                           inBlockOp.getOperand(i).getDefiningOp()) &&
                       "Unable to extract the non-constant operand.");
                auto extractSliceOp = getExtractedOperand(context, builder, loc,
                                                          yieldOp, i, iterNum);
                extractSliceOp->moveBefore(ifOp);
                inBlockOp.setOperand(i, extractSliceOp->getResult(0));
              }
            }
          }
        }
      }
    }
  }

  void eraseDAG(SetVector<Operation *> &unrollOpTree) {
    SetVector<Operation *> eraseOpTree(unrollOpTree.rbegin(),
                                       unrollOpTree.rend());
    for (auto op : eraseOpTree) {
      SetVector<Operation *> users;
      for (auto user : op->getUsers()) {
        if (isa<triton::xpu::ReduceReturnOp>(user)) {
          users.insert(user);
        }
      }
      for (auto user : users) {
        user->erase();
      }
      if (op->use_empty()) {
        op->erase();
      }
    }
  }

  void moveAllocaAndGM2LM(scf::ForOp forOp) {
    ModuleOp m = getOperation();

    SmallVector<triton::xpu::GM2LMOp> gm2lmOps;
    m.walk([&](triton::xpu::GM2LMOp gm2lmOp) { gm2lmOps.push_back(gm2lmOp); });

    for (auto gm2lmOp : gm2lmOps) {
      if (gm2lmOp->getBlock() != forOp->getBlock())
        continue;

      if (gm2lmOp->isBeforeInBlock(forOp))
        continue;

      auto allocaOp = gm2lmOp.getBufPtr().getDefiningOp();

      allocaOp->moveBefore(forOp);
      gm2lmOp->moveBefore(forOp);
    }
  }

  void unrollControl(MLIRContext *context,
                     SmallVector<SetVector<Operation *>> &unrollOpTrees) {
    // Get outerChains
    SmallVector<SetVector<Operation *>> outerChains;
    getOuterChains(unrollOpTrees, outerChains);

    for (int i = 0; i < unrollOpTrees.size(); ++i) {
      auto outerChain = outerChains[i];
      auto unrollOpTree = unrollOpTrees[i];
      // 1. Prepare for unroll control
      int64_t numCol = 1;
      int64_t numUnroll = 1;
      triton::xpu::StoreOp insertPt;
      SmallVector<triton::xpu::StoreOp> allStoreOps;
      for (auto op : unrollOpTree) {
        // 1.1 Get insertPt and tensor num
        if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(op)) {
          auto type = storeOp.getValue().getType();
          numUnroll = numUnroll == 1 ? getnumUnroll(type)
                                     : std::min(numUnroll, getNumCol(type));
          numCol =
              numCol == 1 ? getNumCol(type) : std::min(numCol, getNumCol(type));
          allStoreOps.emplace_back(storeOp);
          //[TODO] To deal with the case that storeOps are in more than one
          // block
          if (insertPt && insertPt->getBlock() != storeOp->getBlock()) {
            return;
          }
          if (!insertPt || storeOp->isBeforeInBlock(insertPt)) {
            insertPt = storeOp;
          }
        }
      }
      if (insertPt) {
        auto loc = insertPt.getLoc();
        int64_t iterNum = ceil<int64_t>(numCol, numUnroll);
        if (iterNum <= 1)
          return;
        LLVM_DEBUG(llvm::dbgs()
                   << "[Unroll Control] Hit Unroll Control Pointwise\n");
        // 2. Unroll control
        // 2.1 Create forOp
        OpBuilder builder(insertPt);
        scf::ForOp forOp;
        arith::IndexCastOp idxVar;
        ValueRange iterArgs;
        createFor(builder, loc, 0, iterNum, forOp, idxVar, iterArgs);
        // 2.2 Set Tensor Type
        IRMapping mapping;
        createLoopBody(context, builder, loc, iterNum, unrollOpTree, outerChain,
                       idxVar, mapping);

        // 3. Erase old DAG
        eraseDAG(unrollOpTree);

        // 4. Move Alloca & GM2LM Op before ForOp
        moveAllocaAndGM2LM(forOp);
      }
    }
  }

  void unrollControlReduce(MLIRContext *context,
                           SetVector<Operation *> &unrollOpTree,
                           Operation *insertPt, ValueRange &iterArgs,
                           ValueRange &returnOperands) {
    SetVector<Operation *> outerChain;
    getOuterChain(unrollOpTree, outerChain);
    if (auto reduceOp = dyn_cast<triton::xpu::ReduceOp>(insertPt)) {
      int64_t numCol = 1, numUnroll = 1;
      getUnrollInfoReduce(reduceOp, numCol, numUnroll);
      int64_t iterNum = ceil<int64_t>(numCol, numUnroll);
      if (iterNum <= 1)
        return;
      OpBuilder builder(reduceOp);
      auto loc = reduceOp.getLoc();
      // 1. Prepare for unroll control
      // Insert ExtractSliceOp for TensorType
      SmallVector<Value> newIterArgs(iterArgs.size());
      for (int i = 0; i < iterArgs.size(); ++i) {
        auto iterArgDefOp = iterArgs[i].getDefiningOp();
        bool isOuter = inOpChain(outerChain, iterArgDefOp);
        auto extractSliceOp = builder.create<triton::xpu::ExtractSliceOp>(
            loc, iterArgs[i].getType(), iterArgs[i]);
        setTensorType(context, extractSliceOp, iterNum, isOuter);
        auto inUnrollOpTree = [&](OpOperand &operand) {
          return unrollOpTree.count(operand.getOwner());
        };
        iterArgs[i].replaceUsesWithIf(extractSliceOp.getResult(),
                                      inUnrollOpTree);
        newIterArgs[i] = extractSliceOp.getResult();
      }
      // 2. Unroll control
      // 2.1 Create forOp
      scf::ForOp forOp;
      arith::IndexCastOp idxVar;
      ValueRange newIterArgsRange(newIterArgs);
      createFor(builder, loc, 1, iterNum, forOp, idxVar, newIterArgsRange);
      // 2.2 Set Tensor Type
      IRMapping mapping;
      createLoopBody(context, builder, loc, iterNum, unrollOpTree, outerChain,
                     idxVar, mapping);
      bool isOuterReduce = inOpChain(outerChain, reduceOp);
      setTensorType(context, reduceOp, iterNum, isOuterReduce, false);
      // 2.3 Modify users and defs
      // replace initArgs with iterArgs
      auto inForOp = [&](OpOperand &operand) {
        return forOp == operand.getOwner()->getBlock()->getParentOp();
      };
      auto forBody = forOp.getBody();
      auto forArgs = forBody->getArguments();
      for (int i = 0; i < forOp.getInitArgs().size(); ++i) {
        forOp.getInitArgs()[i].replaceUsesWithIf(forArgs[i + 1], inForOp);
      }
      SmallVector<Value> mapRes;
      for (int i = 0; i < returnOperands.size(); ++i) {
        mapRes.emplace_back(mapping.lookup(returnOperands[i]));
      }
      builder.create<scf::YieldOp>(loc, mapRes);
      auto isReduceOp = [&](OpOperand &operand) {
        return reduceOp == operand.getOwner();
      };
      for (int i = 0; i < forOp.getResults().size(); ++i) {
        reduceOp.getOperands()[i].replaceUsesWithIf(forOp.getResults()[i],
                                                    isReduceOp);
      }
      // 3. Erase old DAG
      eraseDAG(unrollOpTree);
    }
  }

  void getExcludeChainOps(ModuleOp &m,
                          SetVector<Operation *> &excludeChainOps) {
    m.walk([&](Operation *op) {
      TypeSwitch<const Operation *>(op)
          .Case<XPU_MEMORY_OP>([&](auto memoryOp) {
            getOpChainBwd(excludeChainOps, memoryOp.getPtr().getDefiningOp());
            if (memoryOp.getLen()) {
              getOpChainBwd(excludeChainOps, memoryOp.getLen().getDefiningOp());
            }
          })
          .Case<triton::xpu::LoadOp, triton::xpu::StoreOp>([&](auto acessOp) {
            if (acessOp.getMask()) {
              getOpChainBwd(excludeChainOps, acessOp.getMask().getDefiningOp());
            }
          });
    });
  }

  void findDiscretePtrChain(SetVector<Operation *> &unrollOpTree,
                            SetVector<Operation *> &newUnrollOpTree) {
    for (auto op : unrollOpTree) {
      if (auto loadOp = dyn_cast<triton::xpu::LoadOp>(op)) {
        bool isDiscrete = loadOp.getIsDiscrete();
        if (isDiscrete) {
          OpBuilder builder(loadOp);
          auto loc = loadOp.getLoc();
          auto resType = loadOp.getResult().getType();
          int64_t numCol = getNumCol(resType);
          int64_t numUnroll = getnumUnroll(resType);
          if (numCol > numUnroll && numCol % numUnroll == 0) {
            auto lmPtr = loadOp.getPtr();
            auto gm2lmOp = cast<triton::xpu::GM2LMOp>(
                findDefOpBwd<triton::xpu::GM2LMOp>(lmPtr));
            auto gmPtrOp = cast<triton::AddPtrOp>(
                findDefOpBwd<triton::AddPtrOp>(gm2lmOp.getPtr()));
            auto offset = gmPtrOp.getOffset();
            auto newLmPtr = builder.create<triton::AddPtrOp>(
                loc, lmPtr.getType(), lmPtr, offset);
            SetVector<Operation *> ptrVisitedOps;
            SetVector<Operation *> ptrExcludeChainOps;
            getUnrollTree(newLmPtr, newUnrollOpTree, ptrVisitedOps,
                          ptrExcludeChainOps, newLmPtr, false);
            if (!newUnrollOpTree.empty()) {
              newUnrollOpTree = sortOpTree(newUnrollOpTree);
            }
            gm2lmOp->setAttr("offsetState",
                             builder.getSI32IntegerAttr(static_cast<int32_t>(
                                 OffsetState::Continuous)));
            loadOp.setOperand(0, newLmPtr);
          }
        }
      }
    }
  }

  void
  findDiscretePtrChains(SmallVector<SetVector<Operation *>> &unrollOpTrees,
                        SmallVector<SetVector<Operation *>> &newUnrollOpTrees) {
    for (auto [i, unrollOpTree] : llvm::enumerate(unrollOpTrees)) {
      findDiscretePtrChain(unrollOpTree, newUnrollOpTrees[i]);
    }
  }

  void createDiscreteOffset(ModuleOp &m) {
    m.walk([&](triton::xpu::LoadOp loadOp) {
      bool isDiscrete = loadOp.getIsDiscrete();
      if (isDiscrete) {
        OpBuilder builder(loadOp);
        auto loc = builder.getUnknownLoc();
        auto lmPtr = loadOp.getPtr();
        auto lmAddPtr =
            cast<triton::AddPtrOp>(findDefOpBwd<triton::AddPtrOp>(lmPtr));
        auto lmOffset = lmAddPtr.getOffset();
        auto gm2lmOp = cast<triton::xpu::GM2LMOp>(
            findDefOpBwd<triton::xpu::GM2LMOp>(lmPtr));
        auto gmPtrOp = cast<triton::AddPtrOp>(
            findDefOpBwd<triton::AddPtrOp>(gm2lmOp.getPtr()));
        auto gmOffset = gmPtrOp.getOffset();
        auto extractOp = builder.create<triton::xpu::ExtractOp>(
            loc, getElementTypeOrSelf(gmOffset), builder.getI32IntegerAttr(0),
            gmOffset);
        auto splatOp =
            builder.create<triton::SplatOp>(loc, lmOffset.getType(), extractOp);
        auto offset = builder.create<arith::SubIOp>(loc, lmOffset.getType(),
                                                    lmOffset, splatOp);
        lmAddPtr.setOperand(1, offset);
        lmAddPtr->moveAfter(offset);
        if (gm2lmOp->getOperand(0) == lmAddPtr.getResult())
          gm2lmOp->moveAfter(lmAddPtr);
      }
    });
  }

  void pointwiseUnrollControl(ModuleOp &m, MLIRContext *context) {
    // 1. Data-flow Analysis: get load -> store DAG
    //    (op in ptrChain/lenChain/maskChain will not walk from top to down)
    // 1.1 Get excludeChainOps
    SetVector<Operation *> excludeChainOps;
    getExcludeChainOps(m, excludeChainOps);
    // 1.2 Get load -> store DAG
    SetVector<Operation *> visitedOps;
    SmallVector<SetVector<Operation *>> unrollOpTrees;
    m.walk([&](triton::xpu::StoreOp storeOp) {
      auto valType = storeOp.getValue().getType();
      int64_t numCol = getNumCol(valType);
      int64_t numUnroll = getnumUnroll(valType);
      if (numCol > numUnroll && numCol % numUnroll == 0) {
        getDAG(storeOp, visitedOps, unrollOpTrees, excludeChainOps);
      }
      for (auto visitedOp : visitedOps) {
        if (isa<arith::ConstantOp>(visitedOp)) {
          visitedOps.remove(visitedOp);
        }
      }
    });
    if (unrollOpTrees.size() == 0)
      return;

    // 1.3 Find ptr chain of discrete for moving to loop body
    SmallVector<SetVector<Operation *>> newUnrollOpTrees(unrollOpTrees);
    findDiscretePtrChains(unrollOpTrees, newUnrollOpTrees);

    // 2. Deal with unroll opTrees
    unrollControl(context, newUnrollOpTrees);

    // 3. Calculate discrete offset in the runtime
    createDiscreteOffset(m);
  }

  void createLoadStore(scf::ForOp &forOp, scf::YieldOp &yieldOp, Value &yield,
                       int i, Block &block,
                       SmallVector<Operation *> &storeOps) {
    OpBuilder builder(yieldOp);
    auto loc = yieldOp->getLoc();
    Type yieldType = yield.getType();
    Type yieldElemType = getElementTypeOrSelf(yieldType);
    int64_t vecSize = getNumInVector(yieldElemType);
    Type ptrTy = createPointerType(yieldType, vecSize);
    int64_t tensorSize = getNumCol(yieldType);
    if (!forOp.getResults()[i].use_empty()) {
      // Create Alloca Store for Init Args
      auto initForArg = forOp.getInitArgs()[i];
      auto newAllocaOp = builder.create<triton::xpu::AllocaOp>(
          loc, ptrTy, tensorSize * vecSize);
      auto initStoreOp = builder.create<triton::xpu::StoreOp>(
          loc, newAllocaOp, initForArg, Value(), Value(), -1, false);
      newAllocaOp->moveBefore(forOp);
      initStoreOp->moveBefore(forOp);
      // Create Load for Input
      auto inputLoadOp = builder.create<triton::xpu::LoadOp>(
          loc, yieldType, newAllocaOp, Value(), Value(), Value(), 1, -1, false,
          false, false);
      auto notUsedForYield = [&](OpOperand &operand) {
        return !isa<scf::YieldOp>(operand.getOwner());
      };
      auto forArg = forOp.getRegionIterArgs()[i];
      forArg.replaceUsesWithIf(inputLoadOp, notUsedForYield);
      inputLoadOp->moveBefore(&block.front());
      // Create Store for Output
      auto outputStoreOp = builder.create<triton::xpu::StoreOp>(
          loc, newAllocaOp, yield, Value(), Value(), -1, false);
      outputStoreOp->moveBefore(yieldOp);
      storeOps.emplace_back(outputStoreOp);
      // Create Load for Reduce
      auto reduceLoadOp = builder.create<triton::xpu::LoadOp>(
          loc, yieldType, newAllocaOp, Value(), Value(), Value(), 1, -1, false,
          false, false);
      // Move Load closed to For user
      reduceLoadOp->moveAfter(forOp);
      Operation *insertPt = nullptr;
      for (auto user : forOp.getResults()[i].getUsers()) {
        if (!insertPt) {
          insertPt = user;
        } else {
          if (insertPt->getBlock() == user->getBlock()) {
            if (user->isBeforeInBlock(insertPt)) {
              insertPt = user;
            }
          }
        }
      }
      if (insertPt) {
        reduceLoadOp->moveBefore(insertPt);
      }
      // Replace For Result with Load
      auto notReduceLoadOp = [&](OpOperand &operand) {
        return reduceLoadOp != operand.getOwner();
      };
      forOp.getResults()[i].replaceUsesWithIf(reduceLoadOp, notReduceLoadOp);

      // Discard Yield by setting initForArg to operand
      yieldOp->setOperand(i, initForArg);
    }
  }

  void getUnrollInfoReduce(triton::xpu::ReduceOp &reduceOp, int64_t &numCol,
                           int64_t &numUnroll) {
    auto types = reduceOp.getOperandTypes();
    assert(types.size() > 1);
    for (int i = 0; i < types.size() - 1; ++i) {
      if (i == 0) {
        numCol = getNumCol(types[i]);
        numUnroll = getnumUnroll(types[i]);
      } else {
        assert(numCol == getNumCol(types[i]));
        assert(numUnroll == getnumUnroll(types[i]));
      }
    }
  }

  void forUnrollControl(ModuleOp &m, MLIRContext *context) {
    SetVector<Operation *> excludeChainOps;
    getExcludeChainOps(m, excludeChainOps);
    SetVector<Operation *> vistedForOps;
    // 1. Create Store Load
    m.walk([&](triton::xpu::ReduceOp reduceOp) {
      int64_t numCol = 1, numUnroll = 1;
      getUnrollInfoReduce(reduceOp, numCol, numUnroll);
      if (numCol > numUnroll && numCol % numUnroll == 0) {
        LLVM_DEBUG(llvm::dbgs() << "[Unroll Control] Hit Unroll Control For\n");
        for (auto operand : reduceOp.getOperands()) {
          if (auto forOp = dyn_cast<scf::ForOp>(operand.getDefiningOp())) {
            if (!vistedForOps.count(forOp)) {
              vistedForOps.insert(forOp);
              auto &forBlock = forOp.getRegion().front();
              bool hasIf = false;
              SetVector<Operation *> visitedOps;
              for (auto &inForBlockOp : forBlock) {
                if (auto ifOp = dyn_cast<scf::IfOp>(inForBlockOp)) {
                  SmallVector<Operation *> storeOps;
                  auto &ifBlock = ifOp.getThenRegion().front();
                  auto yieldOp = cast<scf::YieldOp>(ifBlock.getTerminator());
                  for (auto [i, yield] :
                       llvm::enumerate(yieldOp.getOperands())) {
                    createLoadStore(forOp, yieldOp, yield, i, ifBlock,
                                    storeOps);
                  }
                  // Unroll control
                  for (auto storeOp : storeOps) {
                    SmallVector<SetVector<Operation *>> unrollOpTrees;
                    getDAG(storeOp, visitedOps, unrollOpTrees, excludeChainOps);
                    // Find ptr chain of discrete for moving to loop body
                    SmallVector<SetVector<Operation *>> newUnrollOpTrees(
                        unrollOpTrees);
                    findDiscretePtrChains(unrollOpTrees, newUnrollOpTrees);
                    unrollControl(context, newUnrollOpTrees);
                  }
                  hasIf = true;
                }
              }
              if (!hasIf) {
                SmallVector<Operation *> storeOps;
                auto yieldOp = cast<scf::YieldOp>(forBlock.getTerminator());
                for (auto [i, yield] : llvm::enumerate(yieldOp.getOperands())) {
                  createLoadStore(forOp, yieldOp, yield, i, forBlock, storeOps);
                }
                // Unroll control
                for (auto storeOp : storeOps) {
                  SmallVector<SetVector<Operation *>> unrollOpTrees;
                  getDAG(storeOp, visitedOps, unrollOpTrees, excludeChainOps);
                  // Find ptr chain of discrete for moving to loop body
                  SmallVector<SetVector<Operation *>> newUnrollOpTrees(
                      unrollOpTrees);
                  findDiscretePtrChains(unrollOpTrees, newUnrollOpTrees);
                  unrollControl(context, newUnrollOpTrees);
                }
              }
            }
          }
        }
      }
    });
  }

  void getInlineInfo(SetVector<Operation *> &inlineOps, Operation *startOp,
                     ValueRange &returnOperands) {
    Operation *op = startOp;
    while (!isa<triton::xpu::ReduceReturnOp>(op)) {
      inlineOps.insert(op);
      op = op->getNextNode();
    }
    returnOperands = op->getOperands();
  }

  void createReduceWithinCore(ModuleOp &m, MLIRContext *context) {
    SetVector<Operation *> excludeChainOps;
    getExcludeChainOps(m, excludeChainOps);
    m.walk([&](triton::xpu::ReduceOp reduceOp) {
      ReduceOpHelper helper(reduceOp);
      OpBuilder builder(reduceOp);
      auto loc = reduceOp->getLoc();
      SetVector<Operation *> visitedOps;
      auto reduceOperandNum = reduceOp.getNumOperands() - 1;
      SmallVector<SetVector<Operation *>> copyOpTrees;
      SetVector<Operation *> unrollOpTree;
      int64_t numCol = 1, numUnroll = 1;
      getUnrollInfoReduce(reduceOp, numCol, numUnroll);
      if (numCol > numUnroll && numCol % numUnroll == 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[Unroll Control] Hit Unroll Control Reduction\n");
        for (int i = 0; i < reduceOperandNum; ++i) {
          if (auto reduceDefOp = reduceOp.getOperands()[i].getDefiningOp()) {
            getDAG(reduceDefOp, visitedOps, copyOpTrees, excludeChainOps,
                   false);
          }
        }
        // 1. Copy Defined Op Chain of Reduce Operand for InitArgs
        IRMapping mapping;
        for (auto &copyOpTree : copyOpTrees) {
          for (auto &copyOp : copyOpTree) {
            auto newOp = builder.clone(*copyOp, mapping);
            unrollOpTree.insert(newOp);
          }
        }
        // 2. Inline Combine Op of Reduce
        // Clone Region
        IRRewriter rewriter(builder);
        Block *currentBlock = rewriter.getBlock();
        Region &parent = *currentBlock->getParent();
        rewriter.cloneRegionBefore(reduceOp.getCombineOp(), &parent.front());
        auto &newReduce = parent.front();
        // Set Type for Cloned Ops
        auto tensorTy = reduceOp.getInputTypes()[0];
        auto shape = tensorTy.getShape();
        for (auto &op : newReduce) {
          if (auto cmpfOp = dyn_cast<arith::CmpFOp>(op)) {
            auto tensorTy0 = cmpfOp.getODSOperands(0)[0].getType();
            auto tensorTy1 = cmpfOp.getODSOperands(1)[0].getType();
            int operandIndexNeedModify;
            mlir::Type operandNeedReserved;
            if (tensorTy0 != tensorTy1) {
              if ((mlir::isa<mlir::FloatType>(tensorTy0) ||
                   mlir::isa<mlir::IntegerType>(tensorTy0)) &&
                  mlir::isa<mlir::TensorType>(tensorTy1)) {
                operandIndexNeedModify = 0;
                operandNeedReserved = tensorTy1;
              } else if ((mlir::isa<mlir::FloatType>(tensorTy1) ||
                          mlir::isa<mlir::IntegerType>(tensorTy1)) &&
                         mlir::isa<mlir::TensorType>(tensorTy0)) {
                operandIndexNeedModify = 1;
                operandNeedReserved = tensorTy0;
              }
              assert(isa<arith::ConstantOp>(
                         cmpfOp.getOperand(operandIndexNeedModify)
                             .getDefiningOp()) &&
                     "Unable to extract the non-constant operand.");
              auto splatOp = builder.create<triton::SplatOp>(
                  loc, operandNeedReserved,
                  cmpfOp.getOperand(operandIndexNeedModify));
              splatOp->moveBefore(&op);
              cmpfOp.setOperand(operandIndexNeedModify, splatOp.getResult());
            }
          } else if (auto selOp = dyn_cast<arith::SelectOp>(op)) {
            auto tensorTy1 = selOp.getODSOperands(1)[0].getType();
            auto tensorTy2 = selOp.getODSOperands(2)[0].getType();
            int operandIndexNeedModify;
            mlir::Type operandNeedReserved;
            if (tensorTy1 != tensorTy2) {
              if ((mlir::isa<mlir::FloatType>(tensorTy1) ||
                   mlir::isa<mlir::IntegerType>(tensorTy1)) &&
                  mlir::isa<mlir::TensorType>(tensorTy2)) {
                operandIndexNeedModify = 1;
                operandNeedReserved = tensorTy2;
              } else if ((mlir::isa<mlir::FloatType>(tensorTy2) ||
                          mlir::isa<mlir::IntegerType>(tensorTy2)) &&
                         mlir::isa<mlir::TensorType>(tensorTy1)) {
                operandIndexNeedModify = 2;
                operandNeedReserved = tensorTy1;
              }
              assert(isa<arith::ConstantOp>(
                         selOp.getOperand(operandIndexNeedModify)
                             .getDefiningOp()) &&
                     "Unable to extract the non-constant operand.");

              auto splatOp = builder.create<triton::SplatOp>(
                  loc, operandNeedReserved,
                  selOp.getOperand(operandIndexNeedModify));
              splatOp->moveBefore(&op);
              selOp.setOperand(operandIndexNeedModify, splatOp.getResult());
            }
          }
          for (auto [i, resTy] : llvm::enumerate(op.getResultTypes())) {
            auto inlineTensorTy =
                RankedTensorType::get(shape, resTy, tensorTy.getEncoding());
            op.getResult(i).setType(inlineTensorTy);
          }
        }
        // Inline Ops
        llvm::SmallVector<Value> combineArgs(2 * reduceOperandNum);
        for (unsigned i = 0; i < reduceOperandNum; ++i) {
          combineArgs[i] = reduceOp.getOperands()[i];
          combineArgs[reduceOperandNum + i] =
              mapping.lookup(reduceOp.getOperands()[i]);
        }
        auto currOp = &*rewriter.getInsertionPoint();
        auto insertOp = currOp->getPrevNode();
        rewriter.inlineBlockBefore(&newReduce, currOp, combineArgs);
        ValueRange returnOperands;
        getInlineInfo(unrollOpTree, insertOp, returnOperands);

        auto isReduceOp = [&](OpOperand &operand) {
          return reduceOp == operand.getOwner();
        };
        llvm::SmallVector<Value> iterArgs(reduceOperandNum);
        for (auto [i, returnOperand] : llvm::enumerate(returnOperands)) {
          iterArgs[i] = reduceOp.getOperands()[i];
          reduceOp.getOperands()[i].replaceUsesWithIf(returnOperand,
                                                      isReduceOp);
        }
        // Find ptr chain of discrete for moving to loop body
        SetVector<Operation *> newUnrollOpTree(unrollOpTree);
        findDiscretePtrChain(unrollOpTree, newUnrollOpTree);
        // 3. Create Loop for ReduceWithinCore
        ValueRange iterArgsRange(iterArgs);
        unrollControlReduce(context, newUnrollOpTree, reduceOp, iterArgsRange,
                            returnOperands);
        // 4. For Vectorize: triton.addf->triton_xpu.vvaddf
        processOpVecTy(m);
      }
    });
  }

  void reductionUnrollControl(ModuleOp &m, MLIRContext *context) {
    // 1. Unroll Control for Reduce For
    forUnrollControl(m, context);
    // 2. Create For for ReduceWithinCore
    createReduceWithinCore(m, context);
    // 3. Calculate discrete offset in the runtime
    createDiscreteOffset(m);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    bool isReduce = false;
    m.walk([&](triton::xpu::ReduceOp redOp) { isReduce = true; });
    if (isReduce) {
      reductionUnrollControl(m, context);
    } else {
      pointwiseUnrollControl(m, context);
    }
  }

private:
  int64_t numUnrollPerCore = 2;
};

} // namespace xpu
} // namespace triton
} // namespace mlir
