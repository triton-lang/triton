//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// TODO[dyq]: Pass Description
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/IRMapping.h"

#define DEBUG_TYPE "tritonxpu-legalize"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPULEGALIZE
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPULegalizePass
    : public impl::TritonXPULegalizeBase<TritonXPULegalizePass> {

  using impl::TritonXPULegalizeBase<
      TritonXPULegalizePass>::TritonXPULegalizeBase;

  TritonXPULegalizePass() = default;
  TritonXPULegalizePass(unsigned bufferSize, unsigned coreNum) {
    this->bufferSize = bufferSize;
    this->coreNum = coreNum;
  }

  mlir::Operation *findRootOp(mlir::Operation *op) {
    mlir::Operation *rootOp = op;
    while (rootOp->getParentOp()) {
      rootOp = rootOp->getParentOp();
      if (rootOp->getParentOp() && isa<triton::FuncOp>(rootOp->getParentOp())) {
        return rootOp;
      }
    }
    return op;
  }

  void getGroupInfo(llvm::SetVector<Operation *>(opTree), size_t &ngroup,
                    size_t &groupsize, size_t &rowspercore) {
    bool isFirst = true;

    auto _getGroupInfo = [&](triton::xpu::ClusterLayoutAttr &encoding) {
      int _ngroup = product(encoding.getGroupsPerCluster());
      int _groupsize = product(encoding.getCoresPerGroup());
      int _rowspercore = encoding.getSizePerCore()[0];
      if (isFirst) {
        ngroup = _ngroup;
        groupsize = _groupsize;
        rowspercore = _rowspercore;
      } else {
        assert(ngroup == _ngroup && "reduction ngroup is not consistent");
        assert(groupsize == _groupsize &&
               "reduction groupsize is not consistent");
        assert(rowspercore == _rowspercore &&
               "reduction rowspercore is not consistent");
      }
      isFirst = false;
    };
    for (auto op : opTree) {
      op->walk([&](triton::ReduceOp reduceOp) {
        auto defOp = reduceOp.getSrcs()[0].getDefiningOp();
        if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(defOp)) {
          if (auto reshapeResTy =
                  dyn_cast<RankedTensorType>(reshapeOp.getResult().getType())) {
            if (reshapeResTy.getShape().size() == 1) {
              auto reshapeSrcTy =
                  cast<RankedTensorType>(reshapeOp.getOperand().getType());
              if (auto globalEncoding =
                      mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(
                          reshapeSrcTy.getEncoding())) {
                _getGroupInfo(globalEncoding);
              }
            }
          }
        } else {
          if (auto tensorType = mlir::dyn_cast<RankedTensorType>(
                  reduceOp.getOperandTypes()[0])) {
            if (auto globalEncoding =
                    mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(
                        tensorType.getEncoding())) {
              _getGroupInfo(globalEncoding);
            }
          }
        }
      });
    }
    return;
  }

  size_t previousPowerOf2(size_t n) {
    size_t exp = std::log2(n);
    return std::pow(2, exp);
  }

  size_t getSizePerCluster(Type &type, bool unrollOpt) {
    if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) {
      if (auto global = mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(
              tensorType.getEncoding())) {
        size_t sizePerCluster = 1;
        auto tensorShape = tensorType.getShape();
        auto groupsPerCluster = global.getCoresPerGroup();
        auto coresPerGroup = global.getGroupsPerCluster();
        auto sizePerCore = global.getSizePerCore();

        auto rank = tensorShape.size();
        assert(rank == groupsPerCluster.size());
        for (auto i = 0; i < rank; ++i) {
          sizePerCluster *=
              unrollOpt ? groupsPerCluster[i] * coresPerGroup[i]
                        : std::min(sizePerCore[i], (unsigned)tensorShape[i]) *
                              groupsPerCluster[i] * coresPerGroup[i];
        }
        return sizePerCluster;
      } else {
        llvm_unreachable("Only Support ClusterEncodingAttr");
      }
    }
    return 1;
  }

  llvm::SmallVector<int64_t> getSlicedShape(const std::vector<int64_t> &shape,
                                            size_t spaceSize,
                                            bool isReduceMultiGroup) {
    size_t dimSize = 1u;
    size_t rank = shape.size();
    llvm::SmallVector<int64_t> slicedShape(rank, 1u);
    // think about col mask, slice multi-dim tensor to 1-dim tensor
    assert(rank <= 2 && "only 1-dim or 2-dim tensor is supported");

    if (!isReduceMultiGroup) // no opt
      spaceSize = std::min(static_cast<int64_t>(spaceSize), shape[rank - 1]);

    for (int i = rank - 1; i >= 0; --i) {
      dimSize *= shape[i];
      const double sliceNum = static_cast<double>(dimSize) / spaceSize;
      if (sliceNum > 1) {
        slicedShape[i] = std::ceil(shape[i] / sliceNum);
        break;
      }
      slicedShape[i] = shape[i];
    }
    return slicedShape;
  }

  // get greatest common divisor
  size_t gcd(size_t a, size_t b) {
    if (b == 0)
      return a;
    else
      return gcd(b, a % b);
  }

  size_t getLCM(llvm::SmallVector<size_t> &datas) {
    for (int i = 1; i < datas.size(); ++i) {
      datas[i] = datas[i - 1] / gcd(datas[i - 1], datas[i]) * datas[i];
    }
    return datas.back();
  }

  llvm::SmallVector<size_t> getIterationCount(llvm::SmallVector<Type> &types,
                                              bool isReduceMultiGroup,
                                              bool unrollOpt) {
    // bytesPerCluster = bytesPerCore * coresPerGroup * groupsPerCluster
    // 8KB local memory per core, reserve 2KB for parameters
    const size_t bytesPerCluster = 4 * 16 * (6 << 10);
    size_t typesBytes = 0;
    size_t tensorDim = 1;
    for (size_t i = 0; i < types.size(); ++i) {
      Type type = types[i];
      Type valueElemType = getElementTypeOrSelf(type);
      size_t valueElemBytes =
          std::max<int>(valueElemType.getIntOrFloatBitWidth(), 8) / 8u;
      typesBytes += valueElemBytes;
      if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) {
        tensorDim = std::max(tensorDim, tensorType.getShape().size());
      }
    }
    llvm::SmallVector<llvm::SmallVector<size_t>> iterCounts(
        tensorDim, llvm::SmallVector<size_t>(types.size(), 1u));
    for (size_t i = 0; i < types.size(); ++i) {
      if (auto tensorType = mlir::dyn_cast<RankedTensorType>(types[i])) {
        size_t spaceSize =
            std::min(previousPowerOf2(bytesPerCluster / typesBytes),
                     getSizePerCluster(types[i], unrollOpt));
        auto tensorShape = tensorType.getShape();

        if (tensorShape.size() > 2) {
          llvm_unreachable("3D Shape Unsupported.");
        } else if (tensorShape.size() == 2 &&
                   tensorShape[1] > coreNum * bufferSize) {
          LLVM_DEBUG(llvm::dbgs() << "2D Shape[-1] = " << tensorShape[1]
                                  << "); coreNum * bufferSize = "
                                  << coreNum * bufferSize << "\n");
          llvm_unreachable("2D Shape[-1] <= core_num * buffer_size Limit. "
                           "Please Adjust COL_BLOCK_SIZE");
        }

        llvm::SmallVector<int64_t> slicedShape =
            getSlicedShape(tensorShape, spaceSize, isReduceMultiGroup);
        for (size_t j = 0; j < tensorShape.size(); ++j) {
          iterCounts[j][i] =
              std::ceil(static_cast<double>(tensorShape[j]) / slicedShape[j]);
        }
      }
    }

    llvm::SmallVector<size_t> lcmIterCount(tensorDim, 1u);
    for (size_t j = 0; j < iterCounts.size(); ++j) {
      lcmIterCount[j] = getLCM(iterCounts[j]);
    }
    return lcmIterCount;
  }

  Type getSlicedType(const Type &type, llvm::SmallVector<size_t> iterCount,
                     bool isInner, bool needNewEncoding = false) {
    // Slice tensor according iteration count
    if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) {
      auto tensorShape = tensorType.getShape();
      llvm::SmallVector<int64_t> slicedShape(tensorShape.size(), 1u);
      if (tensorShape.size() == iterCount.size()) {
        for (int i = 0; i < tensorShape.size(); ++i) {
          slicedShape[i] = std::max(tensorShape[i] / iterCount[i], size_t(1));
        }
      } else if (tensorShape.size() < iterCount.size()) {
        assert(tensorShape.size() == 1 && iterCount.size() == 2);
        size_t count = isInner ? iterCount[1] : iterCount[0];
        slicedShape[0] = std::max(tensorShape[0] / count, size_t(1));
      } else {
        llvm_unreachable(
            "tensorShape.size() is not more than iterCount.size()");
      }
      ArrayRef<int64_t> sliceTensorShape(slicedShape);
      Attribute encoding;
      if (needNewEncoding) {
        int rank = sliceTensorShape.size();
        llvm::SmallVector<unsigned> order(rank);
        std::iota(order.begin(), order.end(), 0);
        encoding = triton::xpu::ClusterLayoutAttr::get(
            &getContext(), sliceTensorShape, order, 128, 64);
      } else {
        encoding = tensorType.getEncoding();
      }
      return RankedTensorType::get(sliceTensorShape,
                                   tensorType.getElementType(), encoding);
    }
    return type;
  }

  void
  getChains(const llvm::SmallVector<llvm::SetVector<Operation *>> &allOpTrees,
            llvm::SmallVector<llvm::SetVector<Operation *>> &innerChains,
            llvm::SmallVector<llvm::SetVector<Operation *>> &outerChains) {
    for (auto allOpTree : allOpTrees) {
      llvm::SetVector<Operation *> innerChain;
      llvm::SetVector<Operation *> outerChain;
      for (auto op : allOpTree) {
        if (auto rangeOp = dyn_cast<triton::MakeRangeOp>(op)) {
          for (auto user : rangeOp.getResult().getUsers()) {
            if (auto userOp = findUserOp<triton::ExpandDimsOp>(user)) {
              auto expandDimOp = cast<triton::ExpandDimsOp>(userOp);
              if (expandDimOp.getAxis() == 1) {
                outerChain.insert(rangeOp);
              }
            }
          }
        }
        if (auto expandDimOp = dyn_cast<triton::ExpandDimsOp>(op)) {
          auto src = expandDimOp.getSrc();
          auto result = expandDimOp.getResult();
          if (auto srcTy = mlir::dyn_cast<RankedTensorType>(src.getType())) {
            if (auto resTy =
                    mlir::dyn_cast<RankedTensorType>(result.getType())) {
              if (expandDimOp.getAxis() == 0) {
                getOpChainBwd(innerChain, expandDimOp);
                innerChain.remove(expandDimOp);
              }
            }
          }
        }
        if (auto broadcastOp = dyn_cast<triton::xpu::BroadcastOp>(op)) {
          auto src = broadcastOp.getSrc();
          auto result = broadcastOp.getResult();
          if (auto srcTy = mlir::dyn_cast<RankedTensorType>(src.getType())) {
            if (auto resTy =
                    mlir::dyn_cast<RankedTensorType>(result.getType())) {
              auto srcShape = srcTy.getShape();
              auto resShape = resTy.getShape();
              if (srcShape[0] != resShape[0]) { // unequal dim 0 shape means
                                                // in the inner axis op chain
                getOpChainBwd(innerChain, broadcastOp);
                innerChain.remove(broadcastOp);
              }
            }
          }
        }
        if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
          if (reduceOp.getAxis() == 0) {
            getOpChainFwd(innerChain, reduceOp);
          }
        }
      }
      outerChains.emplace_back(outerChain);
      innerChains.emplace_back(innerChain);
    }
  }

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();
    llvm::SmallVector<llvm::SetVector<Operation *>> sortedOpTrees;
    llvm::SetVector<Operation *> visitedOps;
    llvm::SmallVector<llvm::SetVector<Operation *>> allOpTrees;
    llvm::SetVector<Operation *> visitedAllOps;
    llvm::SmallVector<llvm::SmallVector<size_t>> iterCounts;
    llvm::SmallVector<llvm::SmallVector<Type>> allTensorTypes;
    llvm::SetVector<Operation *> storeOps;
    SmallVector<size_t> reduceNGroups;
    SmallVector<size_t> reduceGroupSizes;
    SmallVector<size_t> reduceRowsPerCores;

    // Find SM2GM ptr op chain
    llvm::SetVector<Operation *> sm2gmPtrLenOpChain;
    m.walk([&](triton::xpu::SM2GMOp sm2gmOp) {
      sm2gmPtrLenOpChain.insert(sm2gmOp);
      getOpChainBwd(sm2gmPtrLenOpChain, sm2gmOp.getPtr().getDefiningOp());
      if (sm2gmOp.getLen()) {
        getOpChainBwd(sm2gmPtrLenOpChain, sm2gmOp.getLen().getDefiningOp());
      }
    });

    llvm::SetVector<Operation *> endOps;
    m.walk([&](triton::xpu::LM2GMOp lm2gmOp) { endOps.insert(lm2gmOp); });
    m.walk([&](triton::xpu::SM2GMOp sm2gmOp) { endOps.insert(sm2gmOp); });

    for (auto currStoreOp : endOps) {
      if (!visitedOps.contains(currStoreOp) &&
          !inSameSCFIfBlock(storeOps, currStoreOp)) {
        storeOps.insert(currStoreOp);

        // Get the opTree on the storeOp path
        auto currStoreRootOp = findRootOp(currStoreOp);
        auto currStoreRootBlock = currStoreRootOp->getBlock();
        llvm::SetVector<Operation *> opTree;
        getOpTreeBwd(opTree, visitedOps, currStoreRootOp, currStoreRootBlock);
        llvm::SetVector<Operation *> sortedOpTree = sortOpTreeBwd(opTree);
        sortedOpTrees.emplace_back(sortedOpTree);

        llvm::SetVector<Operation *> allOpTree;
        getOpTreeBwd(allOpTree, visitedAllOps, currStoreOp);
        allOpTrees.emplace_back(allOpTree);

        // Get all tensors types of loadOp or storeOp
        llvm::SmallVector<Type> tensorTypes;
        for (auto op : allOpTree) {
          if (auto loadOp = dyn_cast<triton::xpu::LoadOp>(op)) {
            auto loadResType = loadOp.getResult().getType();
            tensorTypes.emplace_back(loadResType);
          }
          if (auto storeOp = dyn_cast<triton::xpu::LM2GMOp>(op)) {
            auto storeValType = storeOp.getValue().getType();
            tensorTypes.emplace_back(storeValType);
          }
        }
        // Get the iteration count
        allTensorTypes.emplace_back(tensorTypes);
      }
    }

    assert(allTensorTypes.size() == sortedOpTrees.size() &&
           "iteration count != the number of opTrees");

    // 0. Get reduceId/reduceNum for shared memory init
    unsigned reduceId = 0;
    unsigned reduceNum = 0;
    for (auto sortedOpTree : sortedOpTrees) {
      size_t ngroup = 1;
      size_t groupsize = 64;
      size_t rowspercore = 1;
      getGroupInfo(sortedOpTree, ngroup, groupsize, rowspercore);
      reduceNGroups.emplace_back(ngroup);
      reduceGroupSizes.emplace_back(groupsize);
      reduceRowsPerCores.emplace_back(rowspercore);
      for (auto op : sortedOpTree) {
        if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
          reduceNum++;
        }
      }
    }

    for (int i = 0; i < allTensorTypes.size(); ++i) {
      auto tensorTypes = allTensorTypes[i];
      auto opTree = allOpTrees[i];
      // unrollOpt only for 1D tensor, fixed stride and
      // OffsetState::Unknown(resnet max pool infer)
      bool unrollOpt = false;
      // TODO[dyq]: choose [renge]for1 control rather than unrollOpt
      //   for (auto op : opTree) {
      //     if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMOp>(op)) {
      //       auto tensorType = gm2lmOp.getPtr().getType();
      //       auto rank =
      //           mlir::isa<RankedTensorType>(tensorType)
      //               ?
      //               mlir::cast<RankedTensorType>(tensorType).getShape().size()
      //               : 1;
      //       auto cond_1 = rank == 1 ? true : false;
      //       auto cond_2 = gm2lmOp.getFixedStride() == -1 ? true : false;
      //       auto cond_3 = static_cast<mlir::OffsetState>(
      //                         gm2lmOp.getOffsetState()) ==
      //                         OffsetState::Unknown ? true : false;
      //       if (cond_1 && cond_2 && cond_3) {
      //         unrollOpt = true;
      //       } else {
      //         unrollOpt = false;
      //         break;
      //       }
      //     }
      //   }

      bool atomicSim = false;
      size_t simIterCount;
      for (auto tensorTy : tensorTypes) {
        if (auto rankTensorTy = mlir::dyn_cast<RankedTensorType>(tensorTy)) {
          auto gEncoding = mlir::cast<triton::xpu::ClusterLayoutAttr>(
              rankTensorTy.getEncoding());
          auto coresPerGroup = gEncoding.getCoresPerGroup();
          auto groupsPerCluster = gEncoding.getGroupsPerCluster();

          auto oneCoreAct =
              (llvm::find_if(coresPerGroup,
                             [](unsigned int num) { return num != 1; }) ==
               coresPerGroup.end()) &&
              (llvm::find_if(groupsPerCluster, [](unsigned int num) {
                 return num != 1;
               }) == groupsPerCluster.end());

          if (oneCoreAct) {
            atomicSim = true;
            auto shape = rankTensorTy.getShape();
            simIterCount = product(shape);
          } else {
            atomicSim = false;
            break;
          }
        }
      }
      bool isReduceMultiGroup = reduceNGroups[i] > 1 ? true : false;
      llvm::SmallVector<size_t> iterCount =
          getIterationCount(tensorTypes, isReduceMultiGroup, unrollOpt);

      m.walk([&](triton::xpu::GM2LMOp gm2lmOp) {
        if (findUserOp<triton::ReduceOp>(gm2lmOp) ||
            findUserOp<triton::xpu::ReduceOp>(gm2lmOp)) {
          atomicSim = false;
        }
      });

      if (atomicSim) {
        int32_t lrie = 1;
        m.walk([&](triton::xpu::GM2LMOp gm2lmOp) { lrie = gm2lmOp.getLrie(); });
        iterCount.assign(iterCount.size(),
                         mlir::ceil<size_t>(simIterCount, lrie));
      }

      iterCounts.emplace_back(iterCount);
    }

    // For reduce2d, inner axis tensor is not sliced
    llvm::SmallVector<llvm::SetVector<Operation *>> innerChains;
    llvm::SmallVector<llvm::SetVector<Operation *>> outerChains;
    getChains(allOpTrees, innerChains, outerChains);

    // Duplicate the MakeRangeOp to avoid conflict when innerChains and
    // outerChains all include it(bilibli_mul_reducesum dyanmic 2x14x3x256)
    for (int i = 0; i < innerChains.size(); ++i) {
      llvm::SetVector<Operation *> innerChain = innerChains[i];
      llvm::SetVector<Operation *> outerChain = outerChains[i];

      for (auto it = outerChain.begin(); it != outerChain.end(); ++it) {
        Operation *outerOp = *it;
        if (inOpChain(innerChain, outerOp)) { // Common MROp
          if (auto rangeOp = dyn_cast<triton::MakeRangeOp>(outerOp)) {
            // Find MROp's Whose User is ExpandDimsOp(dim=0)
            for (auto user : rangeOp->getUsers()) {
              if (auto op = findUserOp<triton::ExpandDimsOp>(user)) {
                auto expandDimsOp = cast<triton::ExpandDimsOp>(op);
                if (expandDimsOp.getAxis() == 0) {
                  // Recover MakeRangeOp
                  OpBuilder builder(rangeOp);
                  auto loc = builder.getUnknownLoc();
                  auto newMakeRangeOp = builder.create<triton::MakeRangeOp>(
                      loc, rangeOp.getType(), rangeOp.getStart(),
                      rangeOp.getEnd());

                  // Link To InnerChain
                  auto operands = user->getOperands();
                  for (auto _it = operands.begin(); _it != operands.end();
                       ++_it) {
                    auto operand = *_it;
                    if (operand == rangeOp) {
                      user->setOperand(std::distance(operands.begin(), _it),
                                       newMakeRangeOp);
                    }
                  }

                  // Now the old common mrOp is only used by outerChain
                  innerChains[i].insert(newMakeRangeOp);
                  innerChains[i].remove(rangeOp);
                  sortedOpTrees[i].insert(newMakeRangeOp);
                  sortedOpTrees[i] = sortOpTreeBwd(sortedOpTrees[i]);
                }
              }
            }
          }
        }
      }
    }

    // for (auto [i, opTree] : llvm::enumerate(allOpTrees)) {
    //   LLVM_DEBUG(llvm::dbgs() << "\nDump OpTree-" << i << ":\n");
    //   for (auto op : opTree) {
    //     op->dump();
    //   }

    //   LLVM_DEBUG(llvm::dbgs() << "\nDump outerChain-" << i << ":\n");
    //   for (auto op : outerChains[i]) {
    //     op->dump();
    //   }

    //   LLVM_DEBUG(llvm::dbgs() << "\nDump innerChain-" << i << ":\n");
    //   for (auto op : innerChains[i]) {
    //     op->dump();
    //   }
    // }

    auto getInnerChainInfo = [&](Operation *op) -> std::string {
      for (size_t i = 0; i < innerChains.size(); ++i) {
        if (innerChains[i].count(op)) {
          return "InnerChain";
        }
      }
      return "";
    };

    auto printCSV = [&](mlir::ModuleOp &mod) {
      LLVM_DEBUG(llvm::dbgs() << "{\n");
      LLVM_DEBUG(llvm::dbgs() << "Operation,Chain Info\n");

      // 遍历 mod 中的所有操作
      mod.walk([&](mlir::Operation *op) {
        if (dyn_cast<triton::FuncOp>(op) || dyn_cast<mlir::ModuleOp>(op))
          return;
        // 获取操作的字符串表示，记得处理逗号和换行符
        std::string opStr;
        llvm::raw_string_ostream os(opStr);
        op->print(os);
        // 替换逗号和换行符
        std::replace(opStr.begin(), opStr.end(), ',', ';');
        std::replace(opStr.begin(), opStr.end(), '\n', ' ');

        // 获取 InnerChain 信息
        std::string chainInfo = getInnerChainInfo(op);

        // 输出一行
        LLVM_DEBUG(llvm::dbgs() << opStr << "," << chainInfo << "\n");
      });
      LLVM_DEBUG(llvm::dbgs() << "}\n");
    };

    // printCSV(m);

    // 1. Create loop for GM2LM/LM2GM
    for (size_t i = 0; i < iterCounts.size(); ++i) {
      llvm::SetVector<Operation *> sortedOpTree = sortedOpTrees[i];
      llvm::SmallVector<size_t> iterCount = iterCounts[i];
      size_t outIterCount = iterCount[0];
      size_t reduceNGroup = reduceNGroups[i];
      size_t reduceGroupSize = reduceGroupSizes[i];
      size_t reduceRowsPerCore = reduceRowsPerCores[i];
      bool isReduceMultiGroup = reduceNGroup > 1 ? true : false;
      llvm::SetVector<Operation *> innerChain = innerChains[i];
      llvm::SetVector<Operation *> outerChain = outerChains[i];
      auto endOp = sortedOpTree[0];
      OpBuilder builder(endOp);
      auto loc = builder.getUnknownLoc();

      // Set loop args and create for loop.
      auto low =
          builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(0));
      auto upper = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getIndexAttr(outIterCount));
      auto step =
          builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(1));
      // Control elem_size per Loop To Avoid the Mem Overflow
      auto forLoopOp = builder.create<scf::ForOp>(loc, low, upper, step);
      builder.setInsertionPointToStart(forLoopOp.getBody());

      // Create loop body
      Value idx = builder.create<mlir::arith::IndexCastOp>(
          loc, builder.getI32Type(), forLoopOp.getInductionVar());

      Operation *yieldOp = forLoopOp.getBody()->getTerminator();

      // LLVM_DEBUG(llvm::dbgs() << "\nBefore Loop Move:\n" << m << " \n");

      for (auto op : llvm::reverse(sortedOpTree)) {
        // op->dump();

        bool isInner = inOpChain(innerChain, op);
        // LLVM_DEBUG(llvm::dbgs() << "\nisInner: " << isInner);

        if (!isa<scf::YieldOp>(op)) {
          op->moveBefore(yieldOp);
        }

        auto setSlicedResTy = [&](Operation *op, bool isInner = false,
                                  bool needNewEncoding = false) {
          for (auto [i, resTy] : llvm::enumerate(op->getResultTypes())) {
            // LLVM_DEBUG(llvm::dbgs() << "\nOrigin Type: " << resTy);
            auto slicedResTy =
                getSlicedType(resTy, iterCount, isInner, needNewEncoding);
            // LLVM_DEBUG(llvm::dbgs() << "\nSlicedResType Type: " <<
            // slicedResTy);
            op->getResult(i).setType(slicedResTy);
          }
        };

        if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(op)) {
          auto type = makeRangeOp.getType();
          if (outerChain.count(makeRangeOp)) {
            auto slicedResTy = getSlicedType(type, iterCount, false);
            auto newOutRangeOp = builder.create<triton::xpu::OutRangeOp>(
                loc, slicedResTy, reduceGroupSize, reduceRowsPerCore, idx);
            op->replaceAllUsesWith(newOutRangeOp->getResults());
          } else {
            Value index = isInner ? Value() : idx;
            uint32_t start = makeRangeOp.getStart();
            uint32_t end = makeRangeOp.getEnd();
            uint32_t realSize = end - start;
            auto newMakeRangeOp = builder.create<triton::xpu::MakeRangeOp>(
                loc, type, builder.getI32IntegerAttr(start),
                builder.getI32IntegerAttr(end),
                builder.getI32IntegerAttr(realSize), index, Value());
            setSlicedResTy(newMakeRangeOp, isInner);
            uint32_t newEnd =
                start + product(newMakeRangeOp.getType().getShape());
            newMakeRangeOp.setEnd(newEnd);
            op->replaceAllUsesWith(newMakeRangeOp->getResults());
          }
        } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
          //   LLVM_DEBUG(llvm::dbgs() << "\nbefore modify reduceOp\n" << m);
          auto newReduceIdxOp = builder.create<triton::xpu::ReduceOp>(
              loc, reduceOp->getResultTypes(), reduceOp.getSrcs(),
              reduceOp.getAxis(), idx);
          auto &newCombineOp = newReduceIdxOp.getCombineOp();
          builder.cloneRegionBefore(reduceOp.getCombineOp(), newCombineOp,
                                    newCombineOp.end());
          setSlicedResTy(newReduceIdxOp, isInner);
          op->replaceAllUsesWith(newReduceIdxOp->getResults());
          //   LLVM_DEBUG(llvm::dbgs() << "\nAfter modify reduceOp\n" << m);
          for (auto &opInCombine : newCombineOp.getOps()) {
            if (auto redReturnOp =
                    dyn_cast<mlir::triton::ReduceReturnOp>(opInCombine)) {
              auto oldInsertionPoint = builder.saveInsertionPoint();
              builder.setInsertionPoint(redReturnOp);
              auto newRedReturnOp = builder.create<triton::xpu::ReduceReturnOp>(
                  loc, redReturnOp.getOperands());
              builder.restoreInsertionPoint(oldInsertionPoint);
              redReturnOp->replaceAllUsesWith(newRedReturnOp->getResults());
              opInCombine.erase(); // avoid the HasParent Trait
              break;               // stop the loop in combine region
            }
          }
          op->erase();
        } else if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
          if (auto attr =
                  mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue())) {
            auto slicedResTy =
                getSlicedType(constOp.getType(), iterCount, isInner);
            ShapedType slicedShapedType = mlir::cast<ShapedType>(slicedResTy);
            auto newValue = DenseElementsAttr::getFromRawBuffer(
                slicedShapedType, attr.getRawData());
            auto newConstOp = builder.create<mlir::arith::ConstantOp>(
                loc, slicedResTy, newValue);
            op->replaceAllUsesWith(newConstOp->getResults());
          }
        } else if (auto forOp = dyn_cast<mlir::scf::ForOp>(op)) {

          if (iterCount.back() != 1) {
            if (auto stepOp = dyn_cast<mlir::arith::ConstantOp>(
                    forOp.getStep().getDefiningOp())) {
              setSlicedResTy(stepOp, isInner);
            }
          }

          // Set forOp Result Type
          setSlicedResTy(forOp, isInner);

          // Set forOp Arg Type
          auto forBody = forOp.getBody();
          auto forArgs = forBody->getArguments();
          for (auto forArg : forArgs) {
            bool isInnerArg = inOpChain(innerChain, forArg.getDefiningOp());
            auto slicedArgType =
                getSlicedType(forArg.getType(), iterCount, isInnerArg);
            forArg.setType(slicedArgType);
          }

          // Set forOp's childOp Result Type
          auto &forRegion = forOp.getRegion();
          auto &forBlock = forRegion.front();
          SetVector<Operation *> erasedOps;
          for (auto &inBlockOp : forBlock) {
            bool inBlockIsInner = inOpChain(innerChain, &inBlockOp);
            if (auto reduceOpInFor =
                    mlir::dyn_cast<triton::ReduceOp>(inBlockOp)) {
              OpBuilder builderInFor(reduceOpInFor);
              auto newReduceIdxOp = builderInFor.create<triton::xpu::ReduceOp>(
                  reduceOpInFor->getLoc(), reduceOpInFor->getResultTypes(),
                  reduceOpInFor.getSrcs(), reduceOpInFor.getAxis(), idx);
              auto &newCombineOp = newReduceIdxOp.getCombineOp();
              builderInFor.cloneRegionBefore(reduceOpInFor.getCombineOp(),
                                             newCombineOp, newCombineOp.end());
              setSlicedResTy(newReduceIdxOp, inBlockIsInner);
              reduceOpInFor.replaceAllUsesWith(newReduceIdxOp->getResults());
              erasedOps.insert(reduceOpInFor);
              for (auto &opInCombine : newCombineOp.getOps()) {
                if (auto redReturnOp =
                        dyn_cast<mlir::triton::ReduceReturnOp>(opInCombine)) {
                  auto oldInsertionPoint = builderInFor.saveInsertionPoint();
                  builderInFor.setInsertionPoint(redReturnOp);
                  auto newRedReturnOp =
                      builderInFor.create<triton::xpu::ReduceReturnOp>(
                          redReturnOp.getLoc(), redReturnOp.getOperands());
                  builderInFor.restoreInsertionPoint(oldInsertionPoint);
                  redReturnOp->replaceAllUsesWith(newRedReturnOp->getResults());
                  erasedOps.insert(redReturnOp);
                }
              }
            } else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(inBlockOp)) {
              // Set IfOp's childOp Arg Type(Then)
              auto &thenRegion = ifOp.getThenRegion();
              auto &thenBlock = thenRegion.front();
              for (auto &inBlockOp : thenBlock) {
                bool isInnerOp = inOpChain(innerChain, &inBlockOp);

                auto newInBlockOp = thenBlock.begin();
                auto dist = std::distance(thenBlock.begin(),
                                          Block::iterator(inBlockOp));
                std::advance(newInBlockOp, dist);

                for (auto newInBlockOpRes : newInBlockOp->getResults()) {
                  auto slicedOpType = getSlicedType(newInBlockOpRes.getType(),
                                                    iterCount, isInnerOp);
                  newInBlockOpRes.setType(slicedOpType);
                }
              }

              // Set IfOp's childOp Arg Type(Else)
              auto &elseRegion = ifOp.getElseRegion();
              if (!elseRegion.empty()) {
                auto &elseBlock = elseRegion.front();
                for (auto &inBlockOp0 : elseBlock) {
                  bool isInnerOp = inOpChain(innerChain, &inBlockOp0);
                  setSlicedResTy(&inBlockOp0, isInnerOp);
                }
                for (auto newInBlockOpRes : ifOp->getResults()) {
                  auto slicedOpType =
                      getSlicedType(newInBlockOpRes.getType(), iterCount, true);
                  newInBlockOpRes.setType(slicedOpType);
                }
              }
            } else {
              setSlicedResTy(&inBlockOp, inBlockIsInner);
            }
          }
          for (auto op : erasedOps) {
            if (op->use_empty())
              op->erase();
          }
        } else if (auto ifOp = dyn_cast<mlir::scf::IfOp>(op)) {
          // iterCount.back() != 1 check?
          // Set IfOp Result Type
          setSlicedResTy(ifOp, isInner);

          // Set IfOp Arg Type
          auto newIfBody = ifOp.getBody();
          auto newIfArgs = newIfBody->getArguments();
          for (auto newIfArg : newIfArgs) {
            bool isInnerArg = inOpChain(innerChain, newIfArg.getDefiningOp());
            auto slicedArgType =
                getSlicedType(newIfArg.getType(), iterCount, isInnerArg);
            newIfArg.setType(slicedArgType);
          }

          // Set IfOp's childOp Arg Type(Then)
          auto &newIfThenRegion = ifOp.getThenRegion();
          auto &newIfThenBlock = newIfThenRegion.front();
          auto &oldIfThenRegion = ifOp.getThenRegion();
          auto &oldIfThenBlock = oldIfThenRegion.front();
          for (auto &inBlockOp : oldIfThenBlock) {
            bool isInnerOp = inOpChain(innerChain, &inBlockOp);

            auto newInBlockOp = newIfThenBlock.begin();
            auto dist = std::distance(oldIfThenBlock.begin(),
                                      Block::iterator(inBlockOp));
            std::advance(newInBlockOp, dist);

            for (auto newInBlockOpRes : newInBlockOp->getResults()) {
              auto slicedOpType = getSlicedType(newInBlockOpRes.getType(),
                                                iterCount, isInnerOp);
              newInBlockOpRes.setType(slicedOpType);
            }
          }

          // Set IfOp's childOp Arg Type(Else)
          auto &newIfElseRegion = ifOp.getElseRegion();
          if (!newIfElseRegion.empty()) {
            auto &newIfElseBlock = newIfElseRegion.front();
            auto &oldIfElseRegion = ifOp.getElseRegion();
            auto &oldIfElseBlock = oldIfElseRegion.front();
            for (auto &inBlockOp : oldIfElseBlock) {
              bool isInnerOp = inOpChain(innerChain, &inBlockOp);

              auto newInBlockOp = newIfElseBlock.begin();
              auto dist = std::distance(oldIfElseBlock.begin(),
                                        Block::iterator(inBlockOp));
              std::advance(newInBlockOp, dist);

              for (auto newInBlockOpRes : newInBlockOp->getResults()) {
                auto slicedOpType = getSlicedType(newInBlockOpRes.getType(),
                                                  iterCount, isInnerOp);
                newInBlockOpRes.setType(slicedOpType);
              }
            }
          }

        } else if (auto reshapeOp = dyn_cast<mlir::triton::ReshapeOp>(op)) {
          if (auto reshapeResTy =
                  dyn_cast<RankedTensorType>(reshapeOp.getResult().getType())) {
            auto reshapeResShape = reshapeResTy.getShape();
            if (reshapeResShape.size() == 1) {
              auto reshapeSrcTy =
                  cast<RankedTensorType>(reshapeOp.getOperand().getType());
              auto reshapeSrcShape = reshapeSrcTy.getShape();
              size_t reshapeSrcSize = product(reshapeSrcShape);
              llvm::SmallVector<int64_t> slicedShape(1, 1u);

              slicedShape[0] =
                  std::ceil(static_cast<double>(reshapeSrcSize / iterCount[0]));
              auto slicedReshapeSrcTy = RankedTensorType::get(
                  slicedShape, reshapeSrcTy.getElementType(),
                  reshapeResTy.getEncoding());
              reshapeOp.getResult().setType(slicedReshapeSrcTy);
            }
          }
        } else {
          setSlicedResTy(op, isInner);
        }

        // LLVM_DEBUG(llvm::dbgs() << "After Deal:\n" << m << "\n");
      }
    }

    // Create sm2gmPtrLenOpChain before func.returnOp
    if (!sm2gmPtrLenOpChain.empty()) {
      SmallVector<func::ReturnOp> funcRetures;
      m.walk([&](func::ReturnOp funcReture) {
        funcRetures.push_back(funcReture);
      });
      assert(funcRetures.size() == 1 &&
             "Only one func.return is expected in the module");
      auto sortedSm2gmPtrLenOpChain = sortOpTreeBwd(sm2gmPtrLenOpChain);
      for (int j = sortedSm2gmPtrLenOpChain.size() - 1; j >= 0; --j) {
        auto op = sortedSm2gmPtrLenOpChain[j];
        OpBuilder builder(op);
        op->moveBefore(funcRetures[0]);
        // Set encoding, only core0 sm2gm
        for (auto res : op->getResults()) {
          auto resTy = res.getType();

          if (auto resTensorTy = mlir::dyn_cast<RankedTensorType>(resTy)) {
            auto resShape = resTensorTy.getShape();
            auto elemTy = resTensorTy.getElementType();

            if (auto resEncoding =
                    mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(
                        resTensorTy.getEncoding())) {
              auto sizePerCore = resEncoding.getSizePerCore();
              auto coresPerGroup = resEncoding.getCoresPerGroup();
              auto groupsPerCluster = resEncoding.getGroupsPerCluster();
              auto order = resEncoding.getOrder();
              auto isReduceOpt = resEncoding.getIsReduceOpt();

              SmallVector<unsigned> newCoresPerGroup(coresPerGroup.size(), 1);
              SmallVector<unsigned> newGroupsPerCluster(groupsPerCluster.size(),
                                                        1);
              SmallVector<unsigned> newSizePerCore(resShape.begin(),
                                                   resShape.end());

              auto newEncoding = triton::xpu::ClusterLayoutAttr::get(
                  context, newSizePerCore, newCoresPerGroup,
                  newGroupsPerCluster, order, isReduceOpt);

              auto newResTy =
                  RankedTensorType::get(resShape, elemTy, newEncoding);
              res.setType(newResTy);
              if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
                if (auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(
                        constOp.getValue())) {
                  auto newValue = DenseElementsAttr::getFromRawBuffer(
                      newResTy, attr.getRawData());
                  constOp.setValueAttr(newValue);
                }
              }
            } else if (auto resEncoding =
                           mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(
                               resTensorTy.getEncoding())) {
              auto resGlobalEncoding =
                  mlir::cast<triton::xpu::ClusterLayoutAttr>(
                      resEncoding.getParent());

              auto dim = resEncoding.getDim();
              auto sizePerCore = resGlobalEncoding.getSizePerCore();
              auto coresPerGroup = resGlobalEncoding.getCoresPerGroup();
              auto groupsPerCluster = resGlobalEncoding.getGroupsPerCluster();
              auto order = resGlobalEncoding.getOrder();
              auto isReduceOpt = resGlobalEncoding.getIsReduceOpt();

              SmallVector<unsigned> newCoresPerGroup(coresPerGroup.size(), 1);
              SmallVector<unsigned> newGroupsPerCluster(groupsPerCluster.size(),
                                                        1);
              SmallVector<unsigned> newSizePerCore(sizePerCore.size(), 1);
              assert(sizePerCore.size() < 3 && resShape.size() < 3 &&
                     resShape.size() <= sizePerCore.size());
              if (sizePerCore.size() == 2 && resShape.size() == 1) {
                newSizePerCore[1 - dim] = resShape[0];
              } else {
                for (int i = 0; i < resShape.size(); ++i) {
                  newSizePerCore[i] = resShape[i];
                }
              }

              auto newGlobalEncoding = triton::xpu::ClusterLayoutAttr::get(
                  context, newSizePerCore, newCoresPerGroup,
                  newGroupsPerCluster, order, isReduceOpt);
              auto newEncoding = triton::gpu::SliceEncodingAttr::get(
                  context, resEncoding.getDim(), newGlobalEncoding);

              auto newResTy =
                  RankedTensorType::get(resShape, elemTy, newEncoding);
              res.setType(newResTy);
            } else {
              assert(0 && "Unexpected tensor encoding in SM Optimization");
            }
          }
        }
      }
    }

    // Set ReduceOpHelper
    m.walk([&](triton::xpu::ReduceOp redOp) {
      ReduceOpHelper helper(redOp);
      helper.setReduceId(reduceId);
      helper.setReduceNum(reduceNum);
      reduceId++;
    });

    // MakeRange Replace Protection
    m.walk([&](triton::MakeRangeOp mrOp) {
      OpBuilder builder(mrOp);
      auto loc = mrOp->getLoc();
      uint32_t start = mrOp.getStart();
      uint32_t end = mrOp.getEnd();
      uint32_t realSize = end - start;
      auto newMakeRangeOp = builder.create<triton::xpu::MakeRangeOp>(
          loc, mrOp.getType(), builder.getI32IntegerAttr(start),
          builder.getI32IntegerAttr(end), builder.getI32IntegerAttr(realSize),
          Value(), Value());
      mrOp->replaceAllUsesWith(newMakeRangeOp->getResults());
    });
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
