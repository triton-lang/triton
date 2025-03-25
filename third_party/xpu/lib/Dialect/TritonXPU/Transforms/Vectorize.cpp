//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// TODO: Pass Description
//===----------------------------------------------------------------------===//

// clang-format off
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"
// clang-format on

#define DEBUG_TYPE "tritonxpu-vectorize"

namespace mlir {
namespace triton {
namespace xpu {

enum class ElemState {
  SS = 0, /*00*/
  SV = 1, /*01*/
  VS = 2, /*10*/
  VV = 3  /*11*/
};

using OperationTree = llvm::SetVector<mlir::Operation *>;

#define ARITH_BINARY_FLOAT_OP                                                  \
  arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,                  \
      arith::MaximumFOp, arith::MinimumFOp

#define ARITH_BINARY_INT_OP                                                    \
  arith::SubIOp, arith::AndIOp, arith::OrIOp, arith::MulIOp, arith::AddIOp,    \
      arith::XOrIOp

#define MATH_UNARY_OP                                                          \
  math::ExpOp, math::SqrtOp, math::SinOp, math::CosOp, arith::ExtFOp,          \
      arith::TruncFOp, math::AbsFOp

// TODO: VMin when LLVM can select
#define REDUCE_COMBINE_OP                                                      \
  arith::AddFOp, arith::MulFOp, arith::MaxNumFOp, arith::MinNumFOp,            \
      arith::OrIOp, arith::XOrIOp, arith::AndIOp, triton::xpu::ReduceReturnOp

template <typename OP> struct VOp;

#define VOP(SrcType, DstType)                                                  \
  template <> struct VOp<SrcType> {                                            \
    typedef DstType type;                                                      \
  };

VOP(arith::AddFOp, triton::xpu::VvaddFOp)
VOP(arith::SubFOp, triton::xpu::VvsubFOp)
VOP(arith::MulFOp, triton::xpu::VvmulFOp)
VOP(arith::DivFOp, triton::xpu::VvdivFOp)
VOP(arith::MaximumFOp, triton::xpu::VvmaxFOp)
VOP(arith::MinimumFOp, triton::xpu::VvminFOp)

VOP(arith::AddIOp, triton::xpu::VvaddIOp)
VOP(arith::SubIOp, triton::xpu::VvsubIOp)
VOP(arith::MulIOp, triton::xpu::VvmulIOp)
VOP(arith::AndIOp, triton::xpu::VvandIOp)
VOP(arith::XOrIOp, triton::xpu::VvxorIOp)
VOP(arith::OrIOp, triton::xpu::VvorIOp)

VOP(math::ExpOp, triton::xpu::VExpFOp)
VOP(math::AbsFOp, triton::xpu::VAbsFOp)
VOP(math::LogOp, triton::xpu::VLogFOp)
VOP(math::SqrtOp, triton::xpu::VSqrtFOp)
VOP(math::SinOp, triton::xpu::VSinFOp)
VOP(math::CosOp, triton::xpu::VCosFOp)
VOP(arith::ExtFOp, triton::xpu::VExtFOp)
VOP(arith::TruncFOp, triton::xpu::VTruncFOp)
VOP(arith::SIToFPOp, triton::xpu::VSIToFPOp)

template <typename OP> struct VV2SVOp;

#define VV2SVOp(SrcType, DstType)                                              \
  template <> struct VV2SVOp<SrcType> {                                        \
    typedef DstType type;                                                      \
  };

VV2SVOp(triton::xpu::VvaddFOp, triton::xpu::SvaddFOp);
VV2SVOp(triton::xpu::VvmulFOp, triton::xpu::SvmulFOp);
VV2SVOp(triton::xpu::VvsubFOp, triton::xpu::SvsubFOp);
VV2SVOp(triton::xpu::VvmaxFOp, triton::xpu::SvmaxFOp);

} // namespace xpu
} // namespace triton
} // namespace mlir

namespace mlir {

namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUVECTORIZE
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUVectorizePass
    : public impl::TritonXPUVectorizeBase<TritonXPUVectorizePass> {

  using impl::TritonXPUVectorizeBase<
      TritonXPUVectorizePass>::TritonXPUVectorizeBase;

  template <typename T>
  static decltype(auto) createBinVectorizedOp(T op, Type vectorizedTensorTy) {
    OpBuilder builder(op);
    return builder.create<typename VOp<T>::type>(
        op.getLoc(), vectorizedTensorTy, op.getLhs(), op.getRhs());
  }

  template <typename T>
  static decltype(auto) createUnaryVectorizedOp(T op, Type vectorizedTensorTy) {
    OpBuilder builder(op);
    return builder.create<typename VOp<T>::type>(
        op.getLoc(), vectorizedTensorTy, op.getOperand());
  }

  static decltype(auto) createLibdeviceOp(triton::ExternElementwiseOp &op,
                                          const llvm::StringRef &symbol,
                                          Type vectorizedTensorTy) {
    OpBuilder builder(op);
    return builder.create<triton::ExternElementwiseOp>(
        op.getLoc(), vectorizedTensorTy, op.getOperands(), op.getLibname(),
        op.getLibpath(), symbol, op.getPure());
  }

  // TODO[dyq]: open isMultipleOfBank
  // bool isMultipleOfBank(ModuleOp &mod) {
  //   bool res = false;
  //   mod.walk([&](arith::CmpIOp cmpiOp) {
  //     auto lhs = cmpiOp.getLhs();
  //     auto rhs = cmpiOp.getRhs();

  //     if (cmpiOp.predicate() == arith::CmpIPredicate::slt) {
  //       auto lhsShape = lhs.getType().cast<RankedTensorType>().getShape();

  //       if (lhsShape.size() == 2 && lhsShape[0] == 1) { // inner Cmp
  //       Calculation
  //         if (auto rhsOp =
  //                 rhs.getDefiningOp<arith::ConstantOp>()) { // Static Rnumel
  //           auto denseAttr = rhsOp.getValue().dyn_cast<DenseElementsAttr>();
  //           auto elemPerCore =
  //               *denseAttr.getValues<int>().begin();     // get rnumel int
  //           res = (elemPerCore & (bufferSize - 1)) == 0; // check multiple?
  //         }
  //       }
  //     }
  //   });
  //   return res;
  // }

  Operation *getBlockArgumentOp(Value arg) {
    BlockArgument blockArg = mlir::dyn_cast<BlockArgument>(arg);
    Block *block = blockArg.getOwner();
    unsigned argIndex = blockArg.getArgNumber();

    if (auto forOp = dyn_cast<mlir::scf::ForOp>(block->getParentOp())) {
      // TODO[dyq]: check getIterOperands -> getInitArgs
      Value initValue =
          forOp.getInitArgs()[argIndex - forOp.getNumInductionVars()];
      return initValue.getDefiningOp();
    }
    llvm_unreachable(
        "[Vectorization]: Operand is Not a BlockArgument of scf::for.");
    return nullptr;
  }

  bool binLikeOpVectorize(Value lhs, Value rhs, OperationTree &visited,
                          OperationTree &vectorizedOps) {
    bool isFP32Ty = getElementTypeOrSelf(lhs.getType()).isF32() &&
                    getElementTypeOrSelf(rhs.getType()).isF32();
    bool isFP16Ty = getElementTypeOrSelf(lhs.getType()).isF16() &&
                    getElementTypeOrSelf(rhs.getType()).isF16();
    bool isINT32Ty = getElementTypeOrSelf(lhs.getType()).isInteger(32) &&
                     getElementTypeOrSelf(rhs.getType()).isInteger(32);
    if (!isFP32Ty && !isFP16Ty && !isINT32Ty) {
      return false;
    }

    bool isVectorized = false;

    Operation *lhsOp = lhs.getDefiningOp();
    Operation *rhsOp = rhs.getDefiningOp();

    Operation *lhsLoopInitOp = nullptr;
    Operation *rhsLoopInitOp = nullptr;

    if (mlir::isa<BlockArgument>(lhs)) {
      lhsLoopInitOp = getBlockArgumentOp(lhs);
    }

    if (mlir::isa<BlockArgument>(rhs)) {
      rhsLoopInitOp = getBlockArgumentOp(rhs);
    }

    bool lhsVectorized = lhsOp
                             ? vectorize(lhsOp, visited, vectorizedOps)
                             : vectorize(lhsLoopInitOp, visited, vectorizedOps);
    bool rhsVectorized = rhsOp
                             ? vectorize(rhsOp, visited, vectorizedOps)
                             : vectorize(rhsLoopInitOp, visited, vectorizedOps);

    isVectorized = lhsVectorized && rhsVectorized;
    return isVectorized;
  }

  bool vectorize(Operation *op, OperationTree &visited,
                 OperationTree &vectorizedOps) {
    assert(op && "[Vectorization]: Empty Operation pointer");
    visited.insert(op);

    if (vectorizedOps.contains(op))
      return true;

    bool isVectorized = false;
    TypeSwitch<const Operation *>(op)
        .Case<triton::xpu::GM2LMOp>([&](auto loadOp) { isVectorized = true; })
        .Case<triton::xpu::LM2GMOp>([&](auto loadOp) { isVectorized = true; })
        .Case<triton::xpu::GetCoreIdOp>(
            [&](auto coreIdOp) { isVectorized = true; })
        .Case<triton::GetProgramIdOp>(
            [&](auto programIdOp) { isVectorized = true; })
        .Case<arith::ConstantOp>([&](auto constOp) { isVectorized = true; })
        .Case<arith::IndexCastOp>([&](auto unaryOp) { isVectorized = true; })
        .Case<triton::xpu::LoadOp>([&](auto loadOp) {
          unsigned numElems = getTotalElemsPerThread(loadOp.getType());
          Type elemTy = getElementTypeOrSelf(loadOp.getType());
          auto elemWidth = elemTy.getIntOrFloatBitWidth();
          auto vectorWidth = 512 / elemWidth;
          isVectorized = numElems % vectorWidth == 0 && numElems != 0;
        })
        .Case<triton::xpu::StoreOp>([&](auto storeOp) {
          isVectorized = vectorize(storeOp.getValue().getDefiningOp(), visited,
                                   vectorizedOps);
        })
        .Case<triton::xpu::ReduceOp>([&](auto reduceOp) {
          if (ReduceVec) {
            isVectorized = true;
            for (Block &block : reduceOp.getCombineOp().getBlocks()) {
              for (auto &op : block) {
                if (!isa<REDUCE_COMBINE_OP>(op)) {
                  isVectorized = false;
                  break;
                }
              }
            }
          } else {
            isVectorized = false;
          }
        })
        .Case<triton::xpu::ExtractOp>([&](auto extractOp) {
          isVectorized = vectorize(extractOp.getTensor().getDefiningOp(),
                                   visited, vectorizedOps);
        })
        .Case<triton::SplatOp>([&](auto splatOp) {
          auto defineOp = splatOp.getSrc().getDefiningOp();
          if (!defineOp) { // some splatOp deal in_ptr
            isVectorized = true;
          } else { // some splatOp deal tensor
            auto srcTy = splatOp.getSrc().getType();
            isVectorized = getTotalElemsPerThread(srcTy) == 1;
          }
        })
        .Case<triton::xpu::BroadcastOp>([&](auto broadCastOp) {
          // Some BroadcastOp From ReduceOp
          auto srcTy =
              mlir::dyn_cast<RankedTensorType>(broadCastOp.getSrc().getType());
          auto resTy = mlir::dyn_cast<RankedTensorType>(
              broadCastOp.getResult().getType());

          auto srcShape = srcTy.getShape();
          auto resShape = resTy.getShape();

          auto rank = srcTy.getRank();
          unsigned resNumElems = getTotalElemsPerThread(resTy);

          if (rank == 2 && resNumElems >= 16) {
            // srcShape[0] > 32: Scalar Calculations Perform Better than Vector
            // Calculations When The Data Size is Small.
            if ((srcShape[0] > 32 && srcShape[1] == 1) ||
                (srcShape[0] == 1 && srcShape[1] == resShape[1])) {
              isVectorized = true;
            }
          }
        })
        .Case<triton::ExpandDimsOp>([&](auto expandDimsOp) {
          isVectorized = vectorize(expandDimsOp.getOperand().getDefiningOp(),
                                   visited, vectorizedOps);
        })
        .Case<triton::AddPtrOp>([&](auto addPtrOp) {
          isVectorized = vectorize(addPtrOp.getPtr().getDefiningOp(), visited,
                                   vectorizedOps) &&
                         vectorize(addPtrOp.getOffset().getDefiningOp(),
                                   visited, vectorizedOps);
        })
        .Case<triton::gpu::ConvertLayoutOp>([&](auto cvtOp) {
          isVectorized = vectorize(cvtOp.getOperand().getDefiningOp(), visited,
                                   vectorizedOps);
        })
        .Case<arith::SelectOp>([&](auto selectOp) {
          auto tv = selectOp.getTrueValue();
          auto fv = selectOp.getFalseValue();
          isVectorized = binLikeOpVectorize(tv, fv, visited, vectorizedOps);
        })
        .Case<arith::CmpIOp>([&](auto cmpIOp) {
          isVectorized = false;
          // TODO: Add vCmpIOp Support
          //   auto lhs = cmpIOp.getLhs();
          //   auto rhs = cmpIOp.getRhs();
          //   isVectorized = binLikeOpVectorize(lhs, rhs, visited,
          //   vectorizedOps);
        })
        .Case<arith::CmpFOp>([&](auto cmpFOp) {
          auto lhs = cmpFOp.getLhs();
          auto rhs = cmpFOp.getRhs();
          isVectorized = binLikeOpVectorize(lhs, rhs, visited, vectorizedOps);
        })
        .Case<scf::IfOp>([&](auto ifOp) {
          // For then Region
          Region &thenRegion = ifOp.getThenRegion();
          Block &thenBlock = thenRegion.front();
          Operation *thenTerminator = thenBlock.getTerminator();

          if (auto yieldOp = dyn_cast<scf::YieldOp>(thenTerminator)) {
            if (auto prevOp = yieldOp.getOperands().front().getDefiningOp()) {
              isVectorized = vectorize(prevOp, visited, vectorizedOps);
            }
          }

          // For Else Region
          if (!ifOp.getElseRegion().empty()) {
            Region &elseRegion = ifOp.getElseRegion();
            Block &elseBlock = elseRegion.front();
            Operation *elseTerminator = elseBlock.getTerminator();
            if (auto yieldOp = dyn_cast<scf::YieldOp>(elseTerminator)) {
              if (auto prevOp = yieldOp.getOperands().front().getDefiningOp()) {
                isVectorized &= vectorize(prevOp, visited, vectorizedOps);
              }
            }
          }
        })
        .Case<scf::ForOp>([&](auto forOp) {
          // TODO[dyq]: check getIterOperands -> getInitArgs
          auto iterArgsInitValues = forOp.getInitArgs();
          Region &region = forOp.getRegion();
          Block &block = region.front();
          Operation *terminator = block.getTerminator();

          if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
            if (auto prevOp = yieldOp.getOperands().front().getDefiningOp()) {
              isVectorized = vectorize(prevOp, visited, vectorizedOps) &&
                             iterArgsInitValues.size() == 1;
            }
          }
        })
        .Case<scf::YieldOp>([&](auto yieldOp) {
          if (auto prevOp = yieldOp.getOperands().front().getDefiningOp()) {
            isVectorized = vectorize(prevOp, visited, vectorizedOps);
          }
        })
        .Case<triton::ExternElementwiseOp>([&](auto extElemwiseOp) {
          auto symbol = extElemwiseOp.getSymbol();
          auto prevOp = extElemwiseOp.getOperands().front().getDefiningOp();
          assert(extElemwiseOp.getOperands().size() > 0 &&
                 "Unexcepted ExternElementwiseOp Operand");
          if (symbol == "_ZN3xpu5tanhfEf") {
            isVectorized = false;
            // isVectorized = true;
            // for (auto operand : extElemwiseOp.getOperands()) {
            //   isVectorized =
            //       isVectorized && vectorize(prevOp, visited, vectorizedOps);
            // }
          } else if (symbol == "_ZN3xpu3erfEf") {
            isVectorized = true;
            for (auto operand : extElemwiseOp.getOperands()) {
              isVectorized =
                  isVectorized && vectorize(prevOp, visited, vectorizedOps);
            }
          } else if (symbol == "_ZN3xpu5isinfEf") {
            isVectorized = false;
            // TODO: check visinf logic
            // isVectorized = true;
            // for (auto operand : extElemwiseOp.getOperands()) {
            //   isVectorized =
            //       isVectorized && vectorize(prevOp, visited, vectorizedOps);
            // }
          } else {
            isVectorized = false;
            LLVM_DEBUG(llvm::dbgs()
                       << "[Vectorization]: Unsupported LibDeviceOp Symbol"
                       << symbol << "\n");
          }
        })
        .Case<arith::SIToFPOp>([&](arith::SIToFPOp unaryOp) {
          auto inType = getElementTypeOrSelf(unaryOp.getIn().getType());
          isVectorized = inType.isInteger(32) &&
                         vectorize(unaryOp.getOperand().getDefiningOp(),
                                   visited, vectorizedOps);
        })
        .Case<ARITH_BINARY_FLOAT_OP>([&](auto binOp) {
          auto lhs = binOp.getLhs();
          auto rhs = binOp.getRhs();
          isVectorized = binLikeOpVectorize(lhs, rhs, visited, vectorizedOps);
        })
        .Case<ARITH_BINARY_INT_OP>([&](auto binOp) {
          auto lhs = binOp.getLhs();
          auto rhs = binOp.getRhs();
          isVectorized = binLikeOpVectorize(lhs, rhs, visited, vectorizedOps);
        })
        .Case<MATH_UNARY_OP>([&](auto unaryOp) {
          isVectorized = vectorize(unaryOp.getOperand().getDefiningOp(),
                                   visited, vectorizedOps);
        });

    if (!isVectorized) {
      if (dumpFlag) {
        LLVM_DEBUG({
          op->dump();
          llvm_unreachable("[Vectorization]: Unsupported Operation");
        });
      }
      return false;
    }

    // Dont Need To Vectorize ReduceOp's Result
    if (auto reduceOp = dyn_cast<triton::xpu::ReduceOp>(op))
      return true;

    for (Operation *user : op->getUsers()) {
      if (visited.contains(user))
        continue;

      // FIXME: We've omitted the `other` value of LoadOp when create GM2LMOp in
      // the past. However, `other` value comes back as we are about to separate
      // GM2LMOp and LoadOp, and it will lead to a user LoadOp be in the
      // vectorization path. Actions should be taken to handle this case. Here
      // we workaround to skip LoadOp's `other` value.
      if (auto loadOp = dyn_cast<triton::xpu::LoadOp>(user)) {
        if (op == loadOp.getOther().getDefiningOp()) {
          continue;
        }
      }

      if (!vectorize(user, visited, vectorizedOps))
        return false;
    }

    vectorizedOps.insert(op);
    return true;
  }

  RankedTensorType getVectorType(Type tensorType, unsigned _elemWidth = 0) {
    unsigned numElems = getTotalElemsPerThread(tensorType);
    Type elemTy = getElementTypeOrSelf(tensorType);
    auto elemWidth =
        _elemWidth == 0 ? elemTy.getIntOrFloatBitWidth() : _elemWidth;
    auto vectorWidth = 512 / elemWidth;

    RankedTensorType newTensorTy;

    if (numElems % vectorWidth == 0 &&
        numElems != 0) { // normal vector<16xf32>/vector<32xf16>
      // Step 1. getVectorType
      VectorType newVectorType = mlir::VectorType::get(vectorWidth, elemTy);

      // Step 2. getShape
      RankedTensorType oriTensorTy = mlir::cast<RankedTensorType>(tensorType);
      auto oriShape = oriTensorTy.getShape();
      llvm::SmallVector<int64_t, 4> newShape(oriShape.begin(), oriShape.end());
      auto rank = oriShape.size();
      newShape[rank - 1] /= vectorWidth;

      // Step 3. getEncoding
      auto oriEncoding =
          mlir::cast<triton::xpu::ClusterLayoutAttr>(oriTensorTy.getEncoding());
      auto sizePerCore = oriEncoding.getSizePerCore().vec();
      auto corePerGroup = oriEncoding.getCoresPerGroup().vec();
      auto groupsPerCluster = oriEncoding.getGroupsPerCluster().vec();
      auto order = oriEncoding.getOrder().vec();
      auto isReduceOpt = oriEncoding.getIsReduceOpt();

      sizePerCore[rank - 1] =
          std::max(1, int(sizePerCore[rank - 1] / vectorWidth));

      auto newEncoding = triton::xpu::ClusterLayoutAttr::get(
          tensorType.getContext(), sizePerCore, corePerGroup, groupsPerCluster,
          order, isReduceOpt);

      // Step 4. create RankedTensorType
      newTensorTy = RankedTensorType::get(newShape, newVectorType, newEncoding);
    } else if (numElems == 1) { // special vector<1xf32>
      // Step 1. getVectorType
      VectorType newVectorType = mlir::VectorType::get(1, elemTy);
      // Step 2. getEncoding
      auto newEncoding = triton::xpu::ClusterLayoutAttr::get(
          tensorType.getContext(), {1}, {4}, {16}, {0}, false);
      // Step 3. create RankedTensorType
      newTensorTy = RankedTensorType::get(1, newVectorType, newEncoding);
    } else {
      llvm_unreachable(
          "Only Supported vector<32xTy> or vector<16xTy> or vector<1xTy>");
    }
    return newTensorTy;
  }

  void processOpVecTy(OperationTree &vectorizedOps, ModuleOp &mod) {
    for (auto *op : vectorizedOps) {
      TypeSwitch<Operation *>(op)
          .Case<triton::xpu::LoadOp>([&](auto loadOp) {
            auto newVectorizedTensorTy =
                getVectorType(loadOp.getResult().getType());
            loadOp.getResult().setType(newVectorizedTensorTy);
          })
          .Case<triton::xpu::LM2GMOp>([&](auto lm2gmOp) { (void)lm2gmOp; })
          .Case<triton::xpu::StoreOp>([&](auto storeOp) { (void)storeOp; })
          .Case<ARITH_BINARY_FLOAT_OP>([&](auto binOp) {
            auto newVectorizedTensorTy =
                getVectorType(binOp.getResult().getType());
            auto newBinOp = createBinVectorizedOp(binOp, newVectorizedTensorTy);
            binOp.replaceAllUsesWith(newBinOp.getResult());
            binOp.erase();
          })
          .Case<ARITH_BINARY_INT_OP>([&](auto binOp) {
            auto newVectorizedTensorTy =
                getVectorType(binOp.getResult().getType());
            auto newBinOp = createBinVectorizedOp(binOp, newVectorizedTensorTy);
            binOp.replaceAllUsesWith(newBinOp.getResult());
            binOp.erase();
          })
          .Case<MATH_UNARY_OP>([&](auto unaryOp) {
            auto newVectorizedTensorTy =
                getVectorType(unaryOp.getResult().getType());
            auto newUnaryOp =
                createUnaryVectorizedOp(unaryOp, newVectorizedTensorTy);
            unaryOp.replaceAllUsesWith(newUnaryOp.getResult());
            unaryOp.erase();
          })
          .Case<arith::SIToFPOp>([&](auto unaryOp) {
            auto newVectorizedTensorTy =
                getVectorType(unaryOp.getResult().getType());
            auto newUnaryOp =
                createUnaryVectorizedOp(unaryOp, newVectorizedTensorTy);
            unaryOp.replaceAllUsesWith(newUnaryOp.getResult());
            unaryOp.erase();
          })
          .Case<arith::ConstantOp>([&](auto constOp) {
            auto newVectorizedTensorTy =
                getVectorType(constOp.getResult().getType());
            OpBuilder builder(constOp);
            auto newConstOp = builder.create<triton::xpu::VConstOp>(
                constOp.getLoc(), newVectorizedTensorTy, constOp.getValue());
            constOp.replaceAllUsesWith(newConstOp.getResult());
            constOp.erase();
          })
          .Case<triton::SplatOp>([&](auto splatOp) {
            auto newVectorizedTensorTy =
                getVectorType(splatOp.getResult().getType());
            OpBuilder builder(splatOp);
            auto newSplatOp = builder.create<triton::xpu::VSplatOp>(
                splatOp.getLoc(), newVectorizedTensorTy, splatOp.getOperand());
            splatOp.replaceAllUsesWith(newSplatOp.getResult());
            splatOp.erase();
          })
          .Case<scf::ForOp>([&](auto forOp) {
            auto forBody = forOp.getBody();
            auto forArgs = forBody->getArguments();
            // TODO[dyq]: check getIterOperands -> getInitArgs
            auto iterArgsInitValues = forOp.getInitArgs();
            assert(iterArgsInitValues.size() == 1 &&
                   "[Vectorization]: Only Support ForOp with One Iter Args");
            Value iterArgInitValue = iterArgsInitValues.front();
            auto newVectorizedTensorTy = iterArgInitValue.getType();

            // 1. Change Input Iter Args Type
            forArgs[1].setType(newVectorizedTensorTy);

            // 2. Change Output Type
            forOp.getResult(0).setType(newVectorizedTensorTy);
          })
          .Case<scf::IfOp>([&](auto ifOp) {
            // 1. Get Terminator Type
            Region &thenRegion = ifOp.getThenRegion();
            Block &thenBlock = thenRegion.front();
            Operation *thenTerminator = thenBlock.getTerminator();

            Type resType;
            if (auto yieldOp = dyn_cast<scf::YieldOp>(thenTerminator)) {
              if (auto prevOp = yieldOp.getOperands().front().getDefiningOp()) {
                resType = prevOp->getResult(0).getType();
              }
            } else {
              resType = thenTerminator->getResult(0).getType();
            }

            // 2. Change Output Type
            ifOp.getResult(0).setType(resType);
          })
          .Case<scf::YieldOp>([&](auto yieldOp) { (void)yieldOp; })
          .Case<triton::gpu::ConvertLayoutOp>([&](auto cvtOp) {
            auto newVectorizedTensorTy =
                getVectorType(cvtOp.getResult().getType());
            cvtOp.getResult().setType(newVectorizedTensorTy);
          })
          .Case<arith::SelectOp>([&](auto selectOp) {
            auto newVectorizedTensorTy =
                getVectorType(selectOp.getResult().getType());
            OpBuilder builder(selectOp);
            auto newSelectOp = builder.create<triton::xpu::VSelectOp>(
                selectOp.getLoc(), newVectorizedTensorTy,
                selectOp.getCondition(), selectOp.getTrueValue(),
                selectOp.getFalseValue());
            selectOp.replaceAllUsesWith(newSelectOp.getResult());
            selectOp.erase();
          })
          .Case<arith::CmpFOp>([&](auto cmpFOp) {
            auto rhsTy = cmpFOp.getRhs().getType();
            Type elemTy = getElementTypeOrSelf(getElementTypeOrSelf(rhsTy));
            auto newVectorizedTensorTy = getVectorType(
                cmpFOp.getResult().getType(), elemTy.getIntOrFloatBitWidth());
            OpBuilder builder(cmpFOp);
            auto newCmpFOp = builder.create<triton::xpu::VCmpFOp>(
                cmpFOp.getLoc(), newVectorizedTensorTy, cmpFOp.getPredicate(),
                cmpFOp.getLhs(), cmpFOp.getRhs());
            cmpFOp.replaceAllUsesWith(newCmpFOp.getResult());
            cmpFOp.erase();
          })
          .Case<triton::xpu::BroadcastOp>([&](auto broadCastOp) {
            auto newVectorizedTensorTy =
                getVectorType(broadCastOp.getResult().getType());
            broadCastOp.getResult().setType(newVectorizedTensorTy);
          })
          .Case<triton::ExpandDimsOp>([&](auto expandOp) {
            auto newVectorizedTensorTy =
                getVectorType(expandOp.getResult().getType());
            expandOp.getResult().setType(newVectorizedTensorTy);
          })
          .Case<triton::ExternElementwiseOp>([&](auto extElemwiseOp) {
            auto symbol = extElemwiseOp.getSymbol();
            OpBuilder builder(extElemwiseOp);
            auto newVectorizedTensorTy =
                getVectorType(extElemwiseOp.getResult().getType());
            if (symbol == "_ZN3xpu5tanhfEf") {
              auto newExtElemwiseOp =
                  createLibdeviceOp(extElemwiseOp, "_ZN3xpu6vtanhfEDv16_f",
                                    newVectorizedTensorTy);
              extElemwiseOp.replaceAllUsesWith(newExtElemwiseOp.getResult());
              extElemwiseOp.erase();
            } else if (symbol == "_ZN3xpu3erfEf") {
              auto newExtElemwiseOp = createLibdeviceOp(
                  extElemwiseOp, "_ZN3xpu4verfEDv16_f", newVectorizedTensorTy);
              extElemwiseOp.replaceAllUsesWith(newExtElemwiseOp.getResult());
              extElemwiseOp.erase();
            } else if (symbol == "_ZN3xpu5isinfEf") {
              auto newExtElemwiseOp =
                  createLibdeviceOp(extElemwiseOp, "_ZN3xpu6visinfEDv16_f",
                                    newVectorizedTensorTy);
              extElemwiseOp.replaceAllUsesWith(newExtElemwiseOp.getResult());
              extElemwiseOp.erase();
            } else {
              LLVM_DEBUG(llvm::dbgs()
                         << "[Vectorization]: Can not Convert Symbol " << symbol
                         << " to Vfunc\n");
            }
          })
          .Default([&](auto &op) {
            LLVM_DEBUG(op->dump());
            llvm_unreachable(
                "[Vectorization]: Unsupported Operation Type To VecType");
          });
    }
  }

  bool inline vectorizedTyValid(Type elemTy) {
    if (elemTy.isF16() || elemTy.isF32() || elemTy.isBF16() ||
        elemTy.isInteger(16) || elemTy.isInteger(32))
      return true;
    return false;
  }

  void vectorizeAndProcessOpVecTy(ModuleOp &mod, Operation *rootOp,
                                  Type rootOpTy, std::string logMessage) {
    auto rowsPerCore = 1;
    if (auto rootOpTensorTy = mlir::dyn_cast<RankedTensorType>(rootOpTy)) {
      auto rank = rootOpTensorTy.getShape().size();
      if (rank > 1) {
        rowsPerCore = mlir::cast<triton::xpu::ClusterLayoutAttr>(
                          rootOpTensorTy.getEncoding())
                          .getSizePerCore()[0];
      }
    }

    unsigned numElems = getTotalElemsPerThread(rootOpTy) / rowsPerCore;
    Type vecTy = getElementTypeOrSelf(rootOpTy);
    Type elemTy = getElementTypeOrSelf(vecTy);
    auto elemWidth = elemTy.getIntOrFloatBitWidth();
    auto vectorWidth = 512 / elemWidth;
    if (numElems < vectorWidth || numElems % vectorWidth > 0 ||
        !vectorizedTyValid(elemTy))
      return;

    OperationTree visited;
    OperationTree vectorizedOps;

    if (!vectorize(rootOp, visited, vectorizedOps))
      return;

    LLVM_DEBUG({
      llvm::errs() << logMessage << "\n";
      if (dumpFlag) {
        for (auto vecOp : vectorizedOps)
          vecOp->dump();
      }
    });

    auto encoding = mlir::cast<RankedTensorType>(rootOpTy).getEncoding();

    processOpVecTy(vectorizedOps, mod);
  }

  void maximumFusion(arith::SelectOp selectOp) {
    if (auto orIOp = selectOp.getCondition().getDefiningOp<arith::OrIOp>()) {
      if (orIOp.getResult().hasOneUse()) {
        auto lhs = orIOp.getLhs().getDefiningOp<arith::CmpFOp>();
        auto rhs = orIOp.getRhs().getDefiningOp<arith::CmpFOp>();

        bool isMax = (lhs.getPredicate() == arith::CmpFPredicate::OGT &&
                      rhs.getPredicate() == arith::CmpFPredicate::UNE) ||
                     (lhs.getPredicate() == arith::CmpFPredicate::UNE &&
                      rhs.getPredicate() == arith::CmpFPredicate::OGT);
        bool isMin = (lhs.getPredicate() == arith::CmpFPredicate::OLT &&
                      rhs.getPredicate() == arith::CmpFPredicate::UNE) ||
                     (lhs.getPredicate() == arith::CmpFPredicate::UNE &&
                      rhs.getPredicate() == arith::CmpFPredicate::OLT);

        if (lhs && rhs && lhs.getResult().hasOneUse() &&
            rhs.getResult().hasOneUse()) {
          OpBuilder builder(selectOp);
          if (isMax) {
            auto newMaxFOp = builder.create<arith::MaximumFOp>(
                selectOp.getLoc(), selectOp.getType(), selectOp.getTrueValue(),
                selectOp.getFalseValue());
            selectOp->replaceAllUsesWith(newMaxFOp);
            selectOp->erase();
            orIOp->erase();
            lhs->erase();
            rhs->erase();
            LLVM_DEBUG(llvm::dbgs()
                       << "[Vectorization]: Apply Maximum Fusion Optimization "
                          "For VVMax.\n");
          } else if (isMin) {
            auto newMinFOp = builder.create<arith::MinimumFOp>(
                selectOp.getLoc(), selectOp.getType(), selectOp.getTrueValue(),
                selectOp.getFalseValue());
            selectOp->replaceAllUsesWith(newMinFOp);
            selectOp->erase();
            orIOp->erase();
            lhs->erase();
            rhs->erase();
            LLVM_DEBUG(llvm::dbgs()
                       << "[Vectorization]: Apply Minimum Fusion Optimization "
                          "For VVMin.\n");
          }
        }
      }
    }
  }

  bool isLoadVectorized(triton::xpu::LoadOp loadOp) {
    Type resTy = loadOp.getType();
    Type resElemTy = getElementTypeOrSelf(resTy);
    return mlir::isa<mlir::VectorType>(resElemTy);
  }

  bool SVOptimization_Cond(Operation *op) {
    bool canSVOpt = false;
    // TODO: Check block Argument
    if (!op)
      return canSVOpt;

    TypeSwitch<const Operation *>(op)
        .Case<triton::xpu::LoadOp>([&](auto loadOp) {
          auto gm2lmOp = cast<triton::xpu::GM2LMOp>(loadOp->getPrevNode());
          OffsetState offsetState =
              static_cast<OffsetState>(gm2lmOp.getOffsetState());
          if (offsetState == OffsetState::DiscreteSame &&
              isLoadVectorized(loadOp))
            canSVOpt = true;
        })
        .Case<triton::xpu::BroadcastOp>([&](auto bcOp) {
          auto src = bcOp.getSrc();
          if (auto srcTy = mlir::dyn_cast<RankedTensorType>(src.getType())) {
            auto srcShape = srcTy.getShape();
            if (srcShape.size() == 2 && srcShape[0] == 64 && srcShape[1] == 1) {
              canSVOpt = true;
            }
          }
        })
        .Case<triton::xpu::VConstOp>([&](auto vConstOp) { canSVOpt = true; })
        .Default([&](auto &op) { canSVOpt = false; });

    return canSVOpt;
  }

  bool collectVUser(Operation *op, DenseMap<Operation *, ElemState> &vBinOps) {
    // To check if the collection was successful.
    bool canSVOpt = true;
    for (auto user : op->getUsers()) {
      TypeSwitch<const Operation *>(user)
          .Case<XPU_VVECTORIZED_BINARY_OP>([&](auto vBinOp) {
            auto lDefineOp =
                vBinOp.getLhs().getDefiningOp(); // getLhs define op
            auto rDefineOp = vBinOp.getRhs().getDefiningOp();

            bool lCond = SVOptimization_Cond(lDefineOp);
            bool rCond = SVOptimization_Cond(rDefineOp);

            bool opIsLhs = lDefineOp == op;

            if ((opIsLhs ? lCond : rCond) && (lCond != rCond)) {
              vBinOps[vBinOp] = opIsLhs ? ElemState::SV : ElemState::VS;
            } else {
              canSVOpt = false;
            }
          })
          .Default([&](auto &user) { canSVOpt = false; });

      if (!canSVOpt)
        break;
    }
    return canSVOpt;
  }

  void SVOptimization_Modify(triton::xpu::LoadOp loadOp) {
    // Get Information
    Type tensorType = loadOp.getType();
    Type vecElemTy = getElementTypeOrSelf(tensorType);

    // vecNums / numElems  (all vector<16xTy> use one same Ty)
    unsigned vecNums =
        mlir::cast<RankedTensorType>(tensorType).getNumElements();
    unsigned numElems = getTotalElemsPerThread(tensorType);

    // elem type
    Type elemTy = getElementTypeOrSelf(vecElemTy);

    // encoding
    auto encoding = mlir::cast<triton::xpu::ClusterLayoutAttr>(
        mlir::cast<RankedTensorType>(tensorType).getEncoding());

    std::vector<unsigned> sizePerCore = {1}; // 1 for scalar
    Attribute newEncoding = triton::xpu::ClusterLayoutAttr::get(
        encoding.getContext(), sizePerCore, encoding.getCoresPerGroup(),
        encoding.getGroupsPerCluster(), encoding.getOrder(),
        encoding.getIsReduceOpt());

    Type newTensorType = RankedTensorType::get(
        ceil<unsigned>(vecNums, numElems), elemTy, newEncoding);

    // Replace Origin Op
    OpBuilder builder(loadOp);
    loadOp->setAttr("SVOpt", builder.getBoolAttr(true));
    loadOp->getResult(0).setType(newTensorType);
  }

  // To check if the SVOptimization(Own) was successful.
  void SVOptimization_Modify(triton::xpu::BroadcastOp vBCOp) {
    auto src = vBCOp.getSrc();
    vBCOp.replaceAllUsesWith(src);
    vBCOp.erase();
  }

  // To check if the SVOptimization(Own) was successful.
  void SVOptimization_Modify(triton::xpu::VConstOp vConstOp) {
    auto res = vConstOp.getResult();
    auto resTy = mlir::cast<RankedTensorType>(res.getType());
    triton::xpu::ClusterLayoutAttr vConstOpEncoding =
        mlir::cast<triton::xpu::ClusterLayoutAttr>(resTy.getEncoding());
    unsigned rank = resTy.getRank();

    auto elemTy = getElementTypeOrSelf(vConstOp.getType());
    auto _elemTy = getElementTypeOrSelf(elemTy);
    RankedTensorType newSrcTy;
    if (rank == 1) {
      newSrcTy =
          RankedTensorType::get({/*core_num=*/64}, _elemTy, vConstOpEncoding);
    } else if (rank == 2) {
      newSrcTy = RankedTensorType::get({/*core_num=*/64, 1}, _elemTy,
                                       vConstOpEncoding);
    } else {
      llvm_unreachable("Got Unsupport Rank");
    }

    // TODO[dyq]: dyn_cast -> cast
    auto oriDenseAttr =
        mlir::dyn_cast<mlir::DenseElementsAttr>(vConstOp.getValue());
    auto initValue = DenseElementsAttr::getFromRawBuffer(
        newSrcTy, oriDenseAttr.getRawData());

    OpBuilder builder(vConstOp);
    auto newConstOp = builder.create<arith::ConstantOp>(vConstOp.getLoc(),
                                                        newSrcTy, initValue);
    vConstOp.replaceAllUsesWith(newConstOp.getResult());
    vConstOp.erase();
  }

  template <typename T> void createSVBinOp(T vBinOp, ElemState elemStateInt) {
    if (elemStateInt == ElemState::VS) {
      // SVSUB Has A Strict Order Of Operations.
      // V-S -> -S+V
      if constexpr (std::is_same_v<T, triton::xpu::VvsubFOp>) {
        OpBuilder builder(vBinOp);
        auto negFOp =
            builder.create<arith::NegFOp>(vBinOp.getLoc(), vBinOp.getRhs());
        auto svBinFOp = builder.create<triton::xpu::SvaddFOp>(
            vBinOp.getLoc(), vBinOp.getType(), vBinOp.getLhs(), negFOp,
            static_cast<int32_t>(elemStateInt));
        vBinOp.replaceAllUsesWith(svBinFOp.getResult());
        vBinOp.erase();
        LLVM_DEBUG(llvm::dbgs()
                   << "[Vectorization]: Apply VSSUB -> SVADD Optimization.\n");
        return;
      }
    }

    OpBuilder builder(vBinOp);
    auto svBinFOp = builder.create<typename VV2SVOp<T>::type>(
        vBinOp.getLoc(), vBinOp.getType(), vBinOp.getLhs(), vBinOp.getRhs(),
        static_cast<int32_t>(elemStateInt));
    vBinOp.replaceAllUsesWith(svBinFOp.getResult());
    vBinOp.erase();
  }

  void VvOpToSvOp(DenseMap<Operation *, ElemState> &vBinOps,
                  std::string logMessage) {
    for (auto &pair : vBinOps) {
      auto op = pair.first;
      auto elemStateInt = pair.second;
      TypeSwitch<const Operation *>(op)
          .Case<XPU_VVECTORIZED_BINARY_OP>(
              [&](auto vBinOp) { createSVBinOp(vBinOp, elemStateInt); })
          .Default([&](auto &op) {
            llvm_unreachable(
                "[Vectorization]: Got An Unexpected SV Operation Type");
          });
    }
    LLVM_DEBUG(llvm::dbgs() << logMessage);
  }

  template <typename T> void SVOptimization(T op, std::string logMessage) {
    // Step 1. collect all vUser
    DenseMap<Operation *, ElemState> vBinOps;
    if (!collectVUser(op, vBinOps))
      return;

    // Step 2. Deal Input Op Own Modification
    SVOptimization_Modify(op);

    // Step 3. Deal Input Op's User Modification
    VvOpToSvOp(vBinOps, logMessage);
  }

  // Simpify Mod Graph
  // TODO[dyq]: use canonicalizer
  void cvtOpclean(triton::gpu::ConvertLayoutOp cvtOp) {
    auto src = cvtOp.getSrc();
    auto res = cvtOp.getResult();

    if (src.getType() != res.getType())
      return;

    cvtOp.replaceAllUsesWith(src);
    cvtOp.erase();
  }

  void VvdivToVvmul(triton::xpu::VvdivFOp vvdivOp) {
    // Only can be optimized to vvmul when the denominator is a scalar, it can
    // be further optimized to svmul
    if (auto bcOp =
            vvdivOp.getRhs().getDefiningOp<triton::xpu::BroadcastOp>()) {
      auto src = bcOp.getSrc();
      auto res = bcOp.getResult();

      // Check 1. Src Shape Must Be 64x1xf32
      if (auto srcTy = mlir::dyn_cast<RankedTensorType>(src.getType())) {
        auto srcShape = srcTy.getShape();
        if (srcShape.size() != 2 || !(srcShape[0] == 64 && srcShape[1] == 1)) {
          return;
        } else {
          // Step 2. Create DivOp For Rhs
          OpBuilder builder(bcOp);
          SmallVector<Attribute, 4> intValues(srcShape[1],
                                              builder.getF32FloatAttr(1));
          DenseElementsAttr denseAttr =
              DenseFPElementsAttr::get(srcTy, intValues);
          auto ones =
              builder.create<arith::ConstantOp>(bcOp.getLoc(), denseAttr);

          auto oneDivByRhs =
              builder.create<arith::DivFOp>(bcOp.getLoc(), srcTy, ones, src);

          bcOp->setOperand(0, oneDivByRhs);

          // Step 3. Change vvdiv by vvmul
          OpBuilder builder_tmp(vvdivOp);
          auto vvmulOp = builder_tmp.create<triton::xpu::VvmulFOp>(
              vvdivOp.getLoc(), vvdivOp.getType(), vvdivOp.getLhs(),
              vvdivOp.getRhs());
          vvdivOp.replaceAllUsesWith(vvmulOp->getResult(0));
          vvdivOp.erase();
          LLVM_DEBUG(
              llvm::dbgs()
              << "[Vectorization]: Apply VVDIV -> VVMUL Optimization.\n");
        }
      } else {
        return;
      }
    }
  }

  void VVMacOpFusion(triton::xpu::VvmulFOp mulOp) {
    for (auto nextOp : mulOp->getUsers()) {
      if (auto addOp = dyn_cast<triton::xpu::VvaddFOp>(nextOp)) {
        auto lDefineOp = addOp.getLhs().getDefiningOp(); // getLhs define op
        OpBuilder builder(addOp);
        auto newMacOp = builder.create<triton::xpu::VMacFOp>(
            mulOp.getLoc(), mulOp.getType(), mulOp.getLhs(), mulOp.getRhs(),
            lDefineOp == mulOp ? addOp.getRhs() : addOp.getLhs());

        addOp->replaceAllUsesWith(newMacOp);
        addOp->erase();
        LLVM_DEBUG(llvm::dbgs()
                   << "[Vectorization]: Apply VVMacOp Fusion Optimization.\n");
      }
    }
  }

  void BF16ToFP32VecOptimize(ModuleOp &mod) {
    // bf16Tofp32Unordered could only used in order-independent cases
    bool bf16Tofp32Unordered = true;
    int load_cnt = 0;
    mod.walk([&](triton::xpu::LoadOp loadOp) {
      load_cnt++;
      Type ptrTy = loadOp.getPtr().getType();
      Type ptrElemTy = getElementTypeOrSelf(ptrTy);
      Type ptrDataTy = mlir::cast<PointerType>(ptrElemTy).getPointeeType();
      Type resTy = loadOp.getResult().getType();
      Type resElemTy = getElementTypeOrSelf(resTy);
      Type resScalarTy = getElementTypeOrSelf(resElemTy);

      if (resScalarTy.isF32() && ptrDataTy.isBF16()) {
        auto stride = loadOp.getStride();
        auto tensorColSize = loadOp.getTensorColSize();
        bool isVector = mlir::isa<VectorType>(resElemTy);
        bool isSvOpt = loadOp.getSVOpt();
        bool isDiscreteSame = stride == 0;
        bool isContiguous = stride == 1;
        bool notCoreDealMultiRows = tensorColSize == -1;
        bf16Tofp32Unordered &=
            (isVector && isContiguous && notCoreDealMultiRows) || isSvOpt ||
            isDiscreteSame;
      } else {
        bf16Tofp32Unordered &= false;
      }
    });

    bf16Tofp32Unordered = load_cnt == 0 ? false : bf16Tofp32Unordered;

    mod.walk([&](triton::xpu::StoreOp storeOp) {
      Value val = storeOp.getValue();
      Type valTy = val.getType();
      Type valElemTy = getElementTypeOrSelf(valTy);
      if (bf16Tofp32Unordered && mlir::isa<VectorType>(valElemTy)) {
        if (findDefOpBwd<triton::xpu::MakeRangeOp>(val)) {
          bf16Tofp32Unordered &= false;
        }
      }
    });

    mod.walk([&](triton::xpu::LoadOp loadOp) {
      OpBuilder builder(loadOp);
      loadOp->setAttr("bf16Tofp32Unordered",
                      builder.getBoolAttr(bf16Tofp32Unordered));
    });

    mod.walk([&](triton::xpu::StoreOp storeOp) {
      OpBuilder builder(storeOp);
      storeOp->setAttr("bf16Tofp32Unordered",
                       builder.getBoolAttr(bf16Tofp32Unordered));
    });

    if (bf16Tofp32Unordered) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "[Vectorization]: Apply BF16ToFP32VecUnordered Optimization.\n");
    }
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    // Maximum Fusion Online
    // [cmpf, cmpf, ori, select] -> [fmax]
    if (Maximum_Fusion) {
      mod.walk([&](arith::SelectOp selectOp) { maximumFusion(selectOp); });
    }

    // Eliminate SelectOp For bufferSize X Col Size
    // TODO[dyq]: open isMultipleOfBank
    // if (isMultipleOfBank(mod)) {
    //   mod.walk([&](arith::SelectOp selectOp) {
    //     // Have Only One User(ReduceOp)
    //     if (selectOp.getResult().hasOneUse()) {
    //       auto userOp = *selectOp->user_begin();
    //       if (auto redOp = dyn_cast<triton::xpu::ReduceOp>(userOp)) {
    //         auto trueVal = selectOp.getTrueValue();
    //         auto trueValOp = trueVal.getDefiningOp();

    //         selectOp->replaceAllUsesWith(trueValOp->getResults());
    //         selectOp->erase();
    //         LLVM_DEBUG(llvm::dbgs() << "[Vectorization]: Eliminate SelectOp
    //         For "
    //                         "bufferSize X Col Size.\n");
    //       }
    //     }
    //   });
    // }

    if (ReduceVec) {
      // For [Load -> Reduce] || [Broadcast -> Reduce]
      mod.walk([&](triton::xpu::ReduceOp redOp) {
        for (int i = 0; i < redOp.getOperands().size() - 1; ++i) {
          auto reduceOperand = redOp.getOperands()[i];
          auto reduceOperandOp = reduceOperand.getDefiningOp();
          auto reduceOperandTy = reduceOperand.getType();
          vectorizeAndProcessOpVecTy(mod, reduceOperandOp, reduceOperandTy,
                                     "[Vectorization]: [Load -> "
                                     "Reduce] || [Broadcast -> Reduce] Hit.");
        }

        ReduceOpHelper help(redOp);
        if (help.isVectorized()) {
          // reduceop's correct encoding should be inferd by its input type.
          auto srcLayout = help.getSrcLayout();
          for (Value redRes : redOp.getResults()) {
            if (auto resTy = dyn_cast<RankedTensorType>(redRes.getType())) {
              auto resSliceEncoding =
                  cast<triton::gpu::SliceEncodingAttr>(resTy.getEncoding());
              auto srcClusterEncoding =
                  cast<triton::xpu::ClusterLayoutAttr>(srcLayout);
              auto newEncoding = triton::gpu::SliceEncodingAttr::get(
                  redOp.getContext(), resSliceEncoding.getDim(),
                  srcClusterEncoding);
              auto newResTy = RankedTensorType::get(
                  resTy.getShape(), resTy.getElementType(), newEncoding);
              redRes.setType(newResTy);
            }
          }

          for (Block &block : redOp.getCombineOp().getBlocks()) {
            // Set Arg's Type to VecType
            auto inputTypes = redOp.getInputTypes();
            auto inputSize = inputTypes.size();
            int vecSize = 16;
            for (int i = 0; i < inputSize; ++i) {
              auto vecTy = getElementTypeOrSelf(inputTypes[i]);
              vecSize = cast<VectorType>(vecTy).getNumElements();
              auto arg1 = block.getArguments()[i];
              auto arg2 = block.getArguments()[inputSize + i];
              arg1.setType(vecTy);
              arg2.setType(vecTy);
            }
            // Set CombineOp's Type to VecType
            for (auto &op : block) {
              TypeSwitch<Operation *>(&op)
                  .Case<REDUCE_COMBINE_OP>([&](auto redComOp) {
                    for (auto res : redComOp->getResults()) {
                      auto elemTy = res.getType();
                      VectorType vecType = VectorType::get(vecSize, elemTy);
                      res.setType(vecType);
                    }
                  })
                  .Default([&](auto defaultOp) {
                    LLVM_DEBUG(defaultOp->dump());
                    llvm_unreachable(
                        "[Vectorization]: Unsupported Operation Type "
                        "To VecType in Reduce");
                  });
            }
          }
        }
      });
    }

    // For [Broadcast -> Store]
    mod.walk([&](triton::xpu::StoreOp storeOp) {
      auto storeOpValueTy = storeOp.getValue().getType();
      vectorizeAndProcessOpVecTy(mod, storeOp, storeOpValueTy,
                                 "[Vectorization]: [Broadcast -> Store] Hit.");
    });

    // Eliminate CvtOp in VVOp Path
    if (cvtOp_clean) {
      mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) { cvtOpclean(cvtOp); });
    }

    // Div -> Mul
    if (Div2Mul) {
      mod.walk([&](triton::xpu::VvdivFOp vvdivFOp) { VvdivToVvmul(vvdivFOp); });
    }

    // SV Optimization offline
    if (SV_Fusion) {
      // SVOptimization For LoadOp
      mod.walk([&](triton::xpu::LoadOp vLoadOp) {
        vectorizedLoadOps.insert(vLoadOp);
      });
      for (auto vLoadOp : vectorizedLoadOps) {
        SVOptimization(vLoadOp,
                       "[Vectorization]: Apply SV Optimization For LoadOp.\n");
      }

      // SVOptimization For BroadcastOp
      mod.walk([&](triton::xpu::BroadcastOp vBCOp) {
        vectorizedBcOps.insert(vBCOp);
      });
      for (auto vBCOp : vectorizedBcOps) {
        SVOptimization(
            vBCOp, "[Vectorization]: Apply SV Optimization For BroadcastOp.\n");
      }

      // SVOptimization For ConstOp
      mod.walk([&](triton::xpu::VConstOp vConstOp) {
        vectorizedConstOps.insert(vConstOp);
      });
      for (auto vConstOp : vectorizedConstOps) {
        SVOptimization(
            vConstOp, "[Vectorization]: Apply SV Optimization For VConstOp.\n");
      }
    }

    // MAC Optimization offline
    if (VMAC_Fusion) {
      mod.walk([&](triton::xpu::VvmulFOp vvmulFOp) { // must walk after svOpt
        vvmulFOps.insert(vvmulFOp);
      });

      for (auto vvmulFOp : vvmulFOps) {
        VVMacOpFusion(vvmulFOp);
      }
    }

    // bfloat16 -> float32 Vector Optimization
    if (BF16ToFP32VecOpt) {
      BF16ToFP32VecOptimize(mod);
    }
  }

private:
  llvm::SetVector<triton::xpu::VvmulFOp> vvmulFOps;
  llvm::SetVector<triton::xpu::BroadcastOp> vectorizedBcOps;
  llvm::SetVector<triton::xpu::LoadOp> vectorizedLoadOps;
  llvm::SetVector<triton::xpu::VConstOp> vectorizedConstOps;
  bool Maximum_Fusion = true;
  bool SV_Fusion = true;
  bool VMAC_Fusion = true;
  bool cvtOp_clean = true;
  bool Div2Mul = true;
  bool ReduceVec = true;
  bool BF16ToFP32VecOpt = true;
};

} // namespace xpu
} // namespace triton
} // namespace mlir
