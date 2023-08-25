#include "triton/Conversion/NVGPUToLLVM/NVGPUToLLVMPass.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Conversion/TritonGPUToLLVM/PTXAsmFormat.h"

#include "../lib/Conversion/TritonGPUToLLVM/Utility.h"
using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Conversion/NVGPUToLLVM/Passes.h.inc"

namespace ttn = mlir::triton::nvgpu;
using ::mlir::LLVM::getSRegValue;

namespace {

using Constraint = std::variant<int, std::string>;
using OperandsAndConstraints = std::vector<std::pair<mlir::Value, Constraint>>;
template <typename SourceOp, typename ConcreteT>
class NVGPUOpPatternBase : public mlir::RewritePattern {
public:
  explicit NVGPUOpPatternBase(mlir::MLIRContext *context)
      : mlir::RewritePattern(SourceOp::getOperationName(), 1, context) {}

  mlir::Value convertToType(mlir::Value val, Constraint constraint,
                            Location &loc,
                            mlir::PatternRewriter &rewriter) const {
    if (val.getType().isa<LLVM::LLVMPointerType>()) {
      if (std::holds_alternative<std::string>(constraint)) {
        auto constraintStr = std::get<std::string>(constraint);
        if (constraintStr == "r") {
          return ptrtoint(i32_ty, val);
        } else if (constraintStr == "l") {
          return ptrtoint(i64_ty, val);
        } else {
          return val;
        }
      }
      return val;
    } else {
      if (std::holds_alternative<std::string>(constraint)) {
        auto bitwidth = val.getType().getIntOrFloatBitWidth();
        auto constraintStr = std::get<std::string>(constraint);
        return zext(IntegerType::get(rewriter.getContext(), bitwidth), val);
      }
    }
    return val;
  }
  SmallVector<PTXBuilder::Operand *>
  getPtxOutputs(std::vector<std::string> &outputConstraints,
                PTXBuilder &ptxBuilder) const {
    SmallVector<PTXBuilder::Operand *> ptxOutputs;
    for (unsigned i = 0; i < outputConstraints.size(); i++) {
      auto *ptxOutput = ptxBuilder.newOperand(outputConstraints[i]);
      ptxOutputs.push_back(ptxOutput);
    }
    return ptxOutputs;
  }

  OperandsAndConstraints
  unpackOperands(OperandsAndConstraints &operandsAndConstraints,
                 PTXBuilder &ptxBuilder, Location &loc,
                 mlir::PatternRewriter &rewriter) const {
    OperandsAndConstraints unpackedOperands;
    for (auto &[operand, constraint] : operandsAndConstraints) {
      auto llvmStruct = llvm::dyn_cast<LLVM::LLVMStructType>(operand.getType());
      if (llvmStruct) {
        for (unsigned i = 0; i < llvmStruct.getBody().size(); i++) {
          if (std::holds_alternative<int>(constraint)) {
            auto constraintInt = std::get<int>(constraint);
            unpackedOperands.push_back(
                {extract_val(llvmStruct.getBody()[i], operand, i),
                 constraintInt + i});
          }
        }
      } else {
        unpackedOperands.push_back({operand, constraint});
      }
    }
    return unpackedOperands;
  }

  SmallVector<PTXBuilder::Operand *>
  getPtxOperands(OperandsAndConstraints &operandsAndConstraints,
                 PTXBuilder &ptxBuilder, Location &loc,
                 mlir::PatternRewriter &rewriter) const {
    SmallVector<PTXBuilder::Operand *> ptxOperands;
    auto unpackedOperandsAndConstraints =
        unpackOperands(operandsAndConstraints, ptxBuilder, loc, rewriter);
    for (auto &[operand, constraint] : unpackedOperandsAndConstraints) {
      auto convertedOperand = convertToType(operand, constraint, loc, rewriter);
      if (std::holds_alternative<int>(constraint)) {
        auto *ptxOperand = ptxBuilder.newOperand(
            convertedOperand, std::to_string(std::get<int>(constraint)));
        ptxOperands.push_back(ptxOperand);
      } else {
        auto *ptxOperand = ptxBuilder.newOperand(
            convertedOperand, std::get<std::string>(constraint));
        ptxOperands.push_back(ptxOperand);
      }
    }
    return ptxOperands;
  }

  virtual std::vector<std::string> getOutputConstraints(SourceOp op) const {
    return {};
  }

  virtual OperandsAndConstraints getOperandsAndConstraints(SourceOp op) const {
    return {};
  }

  Type getReturnType(std::vector<std::string> outputConstraints,
                     mlir::PatternRewriter &rewriter) const {
    auto ctx = rewriter.getContext();
    Type resTy;
    if (outputConstraints.empty()) {
      resTy = void_ty(ctx);
    } else {
      SmallVector<Type> retTys;
      for (auto &outputConstraint : outputConstraints) {
        assert(outputConstraint[0] == '=' &&
               "Constraint must be for an output");
        Type retTy;
        switch (outputConstraint[1]) {
        case 'h':
          retTy = IntegerType::get(ctx, 16);
          break;
        case 'r':
          retTy = IntegerType::get(ctx, 32);
          break;
        case 'l':
          retTy = IntegerType::get(ctx, 64);
          break;
        case 'f':
          retTy = FloatType::getF32(ctx);
          break;
        case 'd':
          retTy = FloatType::getF64(ctx);
          break;
        default:
          assert(false && "Unsupported output constraint");
          break;
        }
        retTys.push_back(retTy);
      }
      if (retTys.size() == 1) {
        resTy = retTys[0];
      } else {
        resTy = struct_ty(retTys);
      }
    }
    return resTy;
  }

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto loc = op->getLoc();
    auto sourceOp = llvm::dyn_cast<SourceOp>(op);
    if (!sourceOp)
      return mlir::failure();
    auto concrete = static_cast<const ConcreteT *>(this);
    auto ptxAsm = concrete->getPtxAsm(sourceOp);
    auto hasSideEffects = !isMemoryEffectFree(sourceOp);
    auto operandsAndConstraints = concrete->getOperandsAndConstraints(sourceOp);
    auto outputConstraints = concrete->getOutputConstraints(sourceOp);

    PTXBuilder ptxBuilder;
    auto ptxOutputs = getPtxOutputs(outputConstraints, ptxBuilder);
    auto ptxOperands =
        getPtxOperands(operandsAndConstraints, ptxBuilder, loc, rewriter);
    SmallVector<PTXBuilder::Operand *> outputsAndOperands = ptxOutputs;
    outputsAndOperands.append(ptxOperands.begin(), ptxOperands.end());
    auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);
    ptxInstr(outputsAndOperands, /*onlyAttachMLIRArgs=*/true);
    auto retTy = getReturnType(outputConstraints, rewriter);
    auto res = ptxBuilder.launch(rewriter, loc, retTy,
                                 /*hasSideEffects*/ hasSideEffects);
    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, res);
    }

    return mlir::success();
  }
};

class CGABarrierSyncOpPattern
    : public NVGPUOpPatternBase<ttn::CGABarrierSyncOp,
                                CGABarrierSyncOpPattern> {
public:
  using Base =
      NVGPUOpPatternBase<ttn::CGABarrierSyncOp, CGABarrierSyncOpPattern>;
  using Base::Base;

  std::string getPtxAsm(ttn::CGABarrierSyncOp op) const {
    return "barrier.cluster.sync.aligned;";
  }
};

class FenceAsyncSharedOpPattern
    : public NVGPUOpPatternBase<ttn::FenceAsyncSharedOp,
                                FenceAsyncSharedOpPattern> {
public:
  using Base =
      NVGPUOpPatternBase<ttn::FenceAsyncSharedOp, FenceAsyncSharedOpPattern>;
  using Base::Base;

  std::string getPtxAsm(ttn::FenceAsyncSharedOp op) const {
    auto bCluster = op.getBCluster();
    if (bCluster)
      return "fence.proxy.async.shared::cluster;";
    else
      return "fence.proxy.async.shared::cta;";
  }
};

class WGMMAFenceOpPattern
    : public NVGPUOpPatternBase<ttn::WGMMAFenceOp, WGMMAFenceOpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::WGMMAFenceOp, WGMMAFenceOpPattern>;
  using Base::Base;

  std::string getPtxAsm(ttn::WGMMAFenceOp op) const {
    return "wgmma.fence.sync.aligned;";
  }
};

class WGMMACommitGroupOpPattern
    : public NVGPUOpPatternBase<ttn::WGMMACommitGroupOp,
                                WGMMACommitGroupOpPattern> {
public:
  using Base =
      NVGPUOpPatternBase<ttn::WGMMACommitGroupOp, WGMMACommitGroupOpPattern>;
  using Base::Base;

  std::string getPtxAsm(ttn::WGMMACommitGroupOp op) const {
    return "wgmma.commit_group.sync.aligned;";
  }
};

class WGMMAWaitGroupOpPattern
    : public NVGPUOpPatternBase<ttn::WGMMAWaitGroupOp,
                                WGMMAWaitGroupOpPattern> {
public:
  using Base =
      NVGPUOpPatternBase<ttn::WGMMAWaitGroupOp, WGMMAWaitGroupOpPattern>;
  using Base::Base;

  std::string getPtxAsm(ttn::WGMMAWaitGroupOp op) const {
    auto pendings = op.getPendings();
    return "wgmma.wait_group.sync.aligned " + std::to_string(pendings) + ";";
  }
};

class ClusterArriveOpPattern
    : public NVGPUOpPatternBase<ttn::ClusterArriveOp, ClusterArriveOpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::ClusterArriveOp, ClusterArriveOpPattern>;
  using Base::Base;

  std::string getPtxAsm(ttn::ClusterArriveOp op) const {
    auto relaxed = op.getRelaxed();
    if (relaxed)
      return "barrier.cluster.arrive.relaxed.aligned;";
    else
      return "barrier.cluster.arrive.aligned;";
  }
};

class ClusterWaitOpPattern
    : public NVGPUOpPatternBase<ttn::ClusterWaitOp, ClusterWaitOpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::ClusterWaitOp, ClusterWaitOpPattern>;
  using Base::Base;
  std::string getPtxAsm(ttn::ClusterWaitOp op) const {
    return "barrier.cluster.wait.aligned;";
  }
};

class FenceMBarrierInitOpPattern
    : public NVGPUOpPatternBase<ttn::FenceMBarrierInitOp,
                                FenceMBarrierInitOpPattern> {
public:
  using Base =
      NVGPUOpPatternBase<ttn::FenceMBarrierInitOp, FenceMBarrierInitOpPattern>;
  using Base::Base;

  std::string getPtxAsm(ttn::FenceMBarrierInitOp op) const {
    return "fence.mbarrier_init.release.cluster;";
  }
};

class CGABarrierArriveOpPattern
    : public NVGPUOpPatternBase<ttn::CGABarrierArriveOp,
                                CGABarrierArriveOpPattern> {
public:
  using Base =
      NVGPUOpPatternBase<ttn::CGABarrierArriveOp, CGABarrierArriveOpPattern>;
  using Base::Base;
  std::string getPtxAsm(ttn::CGABarrierArriveOp op) const {
    return "barrier.cluster.arrive;";
  }
};

class CGABarrierWaitOpPattern
    : public NVGPUOpPatternBase<ttn::CGABarrierWaitOp,
                                CGABarrierWaitOpPattern> {
public:
  using Base =
      NVGPUOpPatternBase<ttn::CGABarrierWaitOp, CGABarrierWaitOpPattern>;
  using Base::Base;
  std::string getPtxAsm(ttn::CGABarrierWaitOp op) const {
    return "barrier.cluster.wait;";
  }
};

class RegAllocOpPattern
    : public NVGPUOpPatternBase<ttn::RegAllocOp, RegAllocOpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::RegAllocOp, RegAllocOpPattern>;
  using Base::Base;

  std::string getPtxAsm(ttn::RegAllocOp op) const {
    auto regCount = op.getRegCount();
    return "setmaxnreg.inc.sync.aligned.u32 " + std::to_string(regCount) + ";";
  }
};

class RegDeallocOpPattern
    : public NVGPUOpPatternBase<ttn::RegDeallocOp, RegDeallocOpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::RegDeallocOp, RegDeallocOpPattern>;
  using Base::Base;

  std::string getPtxAsm(ttn::RegDeallocOp op) const {
    auto regCount = op.getRegCount();
    return "setmaxnreg.dec.sync.aligned.u32 " + std::to_string(regCount) + ";";
  }
};

class StoreMatrixOpPattern
    : public NVGPUOpPatternBase<ttn::StoreMatrixOp, StoreMatrixOpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::StoreMatrixOp, StoreMatrixOpPattern>;
  using Base::Base;

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::StoreMatrixOp op) const {
    OperandsAndConstraints operandsAndTypes;
    auto addr = op.getAddr();
    auto datas = op.getDatas();
    operandsAndTypes.push_back({addr, "r"});
    for (unsigned i = 0; i < datas.size(); i++) {
      operandsAndTypes.push_back({datas[i], "r"});
    }
    return operandsAndTypes;
  }

  std::string getPtxAsm(ttn::StoreMatrixOp op) const {
    auto datas = op.getDatas();
    std::string ptxAsm;
    switch (datas.size()) {
    case 1:
      ptxAsm = "stmatrix.sync.aligned.m8n8.x1.shared.b16 [$0], {$1};";
      break;
    case 2:
      ptxAsm = "stmatrix.sync.aligned.m8n8.x2.shared.b16 [$0], {$1, $2};";
      break;
    case 4:
      ptxAsm =
          "stmatrix.sync.aligned.m8n8.x4.shared.b16 [$0], {$1, $2, $3, $4};";
      break;
    default:
      assert(false && "Invalid size");
    }
    return ptxAsm;
  }
};

class MBarrierInitOpPattern
    : public NVGPUOpPatternBase<ttn::MBarrierInitOp, MBarrierInitOpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::MBarrierInitOp, MBarrierInitOpPattern>;
  using Base::Base;

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::MBarrierInitOp op) const {
    OperandsAndConstraints operandsAndTypes;
    Value mbarrier = op.getMbarrier();
    Value pred = op.getPred();
    operandsAndTypes.push_back({mbarrier, "r"});
    operandsAndTypes.push_back({pred, "b"});
    return operandsAndTypes;
  }

  std::string getPtxAsm(ttn::MBarrierInitOp op) const {
    uint32_t count = op.getCount();
    std::string ptxAsm =
        "@$1 mbarrier.init.shared.b64 [$0], " + std::to_string(count) + ";";
    return ptxAsm;
  }
};

class MBarrierArriveOpPattern
    : public NVGPUOpPatternBase<ttn::MBarrierArriveOp,
                                MBarrierArriveOpPattern> {
public:
  using Base =
      NVGPUOpPatternBase<ttn::MBarrierArriveOp, MBarrierArriveOpPattern>;
  using Base::Base;

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::MBarrierArriveOp op) const {
    OperandsAndConstraints operandsAndTypes;
    Value mbarrier = op.getMbarrier();
    Value pred = op.getPred();
    Value ctaId = op.getCtaId();
    auto arriveType = op.getArriveType();

    switch (arriveType) {
    case ttn::MBarriveType::normal:
    case ttn::MBarriveType::cp_async:
    case ttn::MBarriveType::expect_tx:
      operandsAndTypes.push_back({mbarrier, "r"});
      operandsAndTypes.push_back({pred, "b"});
      break;
    case ttn::MBarriveType::remote:
      operandsAndTypes.push_back({mbarrier, "r"});
      operandsAndTypes.push_back({ctaId, "r"});
      operandsAndTypes.push_back({pred, "b"});
      break;
    default:
      llvm::errs() << "Unsupported mbarrier arrive type " << arriveType << "\n";
      llvm_unreachable("");
      break;
    }
    return operandsAndTypes;
  }

  std::string getPtxAsm(ttn::MBarrierArriveOp op) const {
    Value ctaId = op.getCtaId();
    auto arriveType = op.getArriveType();
    uint32_t txCount = op.getTxCount();
    std::string ptxAsm;
    switch (arriveType) {
    case ttn::MBarriveType::normal:
      ptxAsm = "@$1 mbarrier.arrive.shared.b64 _, [$0];";
      break;
    case ttn::MBarriveType::cp_async:
      ptxAsm = "@$1 cp.async.mbarrier.arrive.noinc.shared.b64 [$0];";
      break;
    case ttn::MBarriveType::expect_tx:
      assert(txCount > 0 && "txCount should be valid");
      ptxAsm = "@$1 mbarrier.arrive.expect_tx.shared.b64 _, [$0], " +
               std::to_string(txCount) + ";";
      break;
    case ttn::MBarriveType::remote:
      assert(ctaId && "ctaId should have a valid value");
      ptxAsm =
          " { .reg .b32 remAddr32;                                       \n"
          "  @$2 mapa.shared::cluster.u32  remAddr32, $0, $1;            \n"
          "  @$2 mbarrier.arrive.shared::cluster.b64  _, [remAddr32]; }  \n";
      break;
    default:
      llvm::errs() << "Unsupported mbarrier arrive type " << arriveType << "\n";
      llvm_unreachable("");
      break;
    }
    return ptxAsm;
  }
};

class MBarrierWaitOpPattern
    : public NVGPUOpPatternBase<ttn::MBarrierWaitOp, MBarrierWaitOpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::MBarrierWaitOp, MBarrierWaitOpPattern>;
  using Base::Base;

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::MBarrierWaitOp op) const {
    OperandsAndConstraints operandsAndTypes;
    Value mbarrier = op.getMbarrier();
    Value phase = op.getPhase();
    operandsAndTypes.push_back({mbarrier, "r"});
    operandsAndTypes.push_back({phase, "r"});
    return operandsAndTypes;
  }

  std::string getPtxAsm(ttn::MBarrierWaitOp op) const {
    auto ptxAsm =
        "{                                                           \n"
        ".reg .pred P1;                                              \n"
        "LAB_WAIT:                                                   \n"
        "mbarrier.try_wait.parity.shared.b64 P1, [$0], $1, 0x989680; \n"
        "@P1 bra.uni DONE;                                           \n"
        "bra.uni LAB_WAIT;                                           \n"
        "DONE:                                                       \n"
        "}                                                           \n";
    return ptxAsm;
  }
};

class TMALoadTiledOpPattern
    : public NVGPUOpPatternBase<ttn::TMALoadTiledOp, TMALoadTiledOpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::TMALoadTiledOp, TMALoadTiledOpPattern>;
  using Base::Base;

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::TMALoadTiledOp op) const {
    OperandsAndConstraints operandsAndTypes;
    auto dst = op.getDst();
    auto mbarrier = op.getMbarrier();
    auto tmaDesc = op.getTmaDesc();
    auto l2Desc = op.getL2Desc();
    auto pred = op.getPred();
    auto coords = op.getCoords();
    auto mcastMask = op.getMcastMask();

    auto dimSize = coords.size();
    assert(dimSize == 2 || (dimSize == 4 && mcastMask == nullptr) &&
                               "Does not support TMA configuration");

    operandsAndTypes.push_back({dst, "r"});
    operandsAndTypes.push_back({tmaDesc, "l"});
    for (unsigned i = 0; i < coords.size(); i++) {
      operandsAndTypes.push_back({coords[i], "r"});
    }
    operandsAndTypes.push_back({mbarrier, "l"});
    if (mcastMask) {
      operandsAndTypes.push_back({mcastMask, "h"});
    }
    operandsAndTypes.push_back({l2Desc, "l"});
    operandsAndTypes.push_back({pred, "b"});

    return operandsAndTypes;
  }

  std::string getPtxAsm(ttn::TMALoadTiledOp op) const {
    auto coords = op.getCoords();
    auto mcastMask = op.getMcastMask();
    auto dimSize = coords.size();
    std::string ptxAsm;
    if (dimSize == 2) {
      if (mcastMask == nullptr) {
        ptxAsm = "@$6 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier:"
                 ":complete_tx"
                 "::bytes.L2::cache_hint [$0], [$1, {$2, $3}], [$4], $5;";
      } else {
        ptxAsm = "@$7 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                 "complete_tx::bytes.multicast::cluster.L2::cache_hint"
                 " [$0], [$1, {$2, $3}], [$4], $5, $6;";
      }
    } else if (dimSize == 4) {
      assert(mcastMask == nullptr && "Does not support multicast");
      ptxAsm = "@$8 "
               "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier:"
               ":complete_tx"
               "::bytes.L2::cache_hint [$0], [$1, {$2, $3, $4, $5}], "
               "[$6], $7;";
    } else {
      llvm::errs() << "Unsupported dimSize " << dimSize << "\n";
      llvm_unreachable("");
    }
    return ptxAsm;
  }
};

class TMAStoreTiledOpPattern
    : public NVGPUOpPatternBase<ttn::TMAStoreTiledOp, TMAStoreTiledOpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::TMAStoreTiledOp, TMAStoreTiledOpPattern>;
  using Base::Base;

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::TMAStoreTiledOp op) const {
    OperandsAndConstraints operandsAndTypes;
    auto src = op.getSrc();
    auto tmaDesc = op.getTmaDesc();
    auto pred = op.getPred();
    auto coords = op.getCoords();

    auto dimSize = coords.size();
    if (dimSize != 2 && dimSize != 3 && dimSize != 4) {
      llvm::errs() << "Unsupported dimSize " << dimSize << "\n";
      llvm_unreachable("");
    }
    operandsAndTypes.push_back({tmaDesc, "l"});
    operandsAndTypes.push_back({src, "r"});
    for (unsigned i = 0; i < dimSize; i++) {
      operandsAndTypes.push_back({coords[i], "r"});
    }
    operandsAndTypes.push_back({pred, "b"});

    return operandsAndTypes;
  }

  std::string getPtxAsm(ttn::TMAStoreTiledOp op) const {
    auto coords = op.getCoords();
    auto dimSize = coords.size();
    std::string ptxAsm;
    if (dimSize == 2) {
      ptxAsm = "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
               "[$0, {$2, $3}], [$1];";
    } else if (dimSize == 3) {
      ptxAsm = "@$5 cp.async.bulk.tensor.3d.global.shared::cta.bulk_group"
               "[$0, {$2, $3, $4}], [$1];";
    } else if (dimSize == 4) {
      ptxAsm = "@$6 cp.async.bulk.tensor.4d.global.shared::cta.bulk_group"
               "[$0, {$2, $3, $4, $5}], [$1];";
    } else {
      llvm::errs() << "Unsupported dimSize " << dimSize << "\n";
      llvm_unreachable("");
    }
    return ptxAsm;
  }
};
class NamedBarrierArriveOpPattern
    : public NVGPUOpPatternBase<ttn::NamedBarrierArriveOp,
                                NamedBarrierArriveOpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::NamedBarrierArriveOp,
                                  NamedBarrierArriveOpPattern>;
  using Base::Base;

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::NamedBarrierArriveOp op) const {
    OperandsAndConstraints operandsAndTypes;
    auto bar = op.getBar();
    auto numThreads = op.getNumThreads();
    operandsAndTypes.push_back({bar, "r"});
    operandsAndTypes.push_back({numThreads, "r"});
    return operandsAndTypes;
  }

  std::string getPtxAsm(ttn::NamedBarrierArriveOp op) const {
    return "bar.arrive $0, $1;";
  }
};

class NamedBarrierWaitOpPattern
    : public NVGPUOpPatternBase<ttn::NamedBarrierWaitOp,
                                NamedBarrierWaitOpPattern> {
public:
  using Base =
      NVGPUOpPatternBase<ttn::NamedBarrierWaitOp, NamedBarrierWaitOpPattern>;
  using Base::Base;

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::NamedBarrierWaitOp op) const {
    OperandsAndConstraints operandsAndTypes;
    auto bar = op.getBar();
    auto numThreads = op.getNumThreads();
    operandsAndTypes.push_back({bar, "r"});
    operandsAndTypes.push_back({numThreads, "r"});
    return operandsAndTypes;
  }

  std::string getPtxAsm(ttn::NamedBarrierWaitOp op) const {
    return "bar.sync $0, $1;";
  }
};

class StoreDSmemOpPattern
    : public NVGPUOpPatternBase<ttn::StoreDSmemOp, StoreDSmemOpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::StoreDSmemOp, StoreDSmemOpPattern>;
  using Base::Base;

  OperandsAndConstraints getOperandsAndConstraints(ttn::StoreDSmemOp op) const {
    OperandsAndConstraints operandsAndTypes;
    auto addr = op.getAddr();
    auto ctaId = op.getCtaId();
    auto values = op.getValues();
    auto pred = op.getPred();
    auto bitwidth = op.getBitwidth();
    operandsAndTypes.push_back({addr, "r"});
    operandsAndTypes.push_back({ctaId, "r"});
    operandsAndTypes.push_back({pred, "b"});
    std::string c = bitwidth == 16 ? "h" : (bitwidth == 32 ? "r" : "l");
    for (unsigned i = 0; i < values.size(); i++) {
      operandsAndTypes.push_back({values[i], c});
    }
    return operandsAndTypes;
  }

  std::string getPtxAsm(ttn::StoreDSmemOp op) const {
    auto bitwidth = op.getBitwidth();
    auto vec = op.getVec();
    auto values = op.getValues();
    assert(
        (bitwidth == 8 || bitwidth == 16 || bitwidth == 32 || bitwidth == 64) &&
        "invalid bitwidth");
    assert((vec == 1 || vec == 2 || vec == 4) && vec == values.size() &&
           "invalid vec size");
    std::string ptxAsm;
    if (vec == 1) {
      ptxAsm = "{                                           \n"
               ".reg .u32 remoteAddr;                       \n"
               "mapa.shared::cluster.u32 remoteAddr, $0, $1;\n"
               ".reg .pred p;                               \n"
               "mov.pred p, $2;                             \n"
               "@p st.shared::cluster.u" +
               std::to_string(bitwidth) +
               " [remoteAddr], $3; \n"
               "}\n";
    }
    if (vec == 2) {
      ptxAsm = "{                                           \n"
               ".reg .u32 remoteAddr;                       \n"
               "mapa.shared::cluster.u32 remoteAddr, $0, $1;\n"
               ".reg .pred p;                               \n"
               "mov.pred p, $2;                             \n"
               "@p st.shared::cluster.v.u" +
               std::to_string(bitwidth) +
               " [remoteAddr], {$3, $4}; \n"
               "}\n";
    }
    if (vec == 4) {
      ptxAsm = "{                                           \n"
               ".reg .u32 remoteAddr;                       \n"
               "mapa.shared::cluster.u32 remoteAddr, $0, $1;\n"
               ".reg .pred p;                               \n"
               "mov.pred p, $2;                             \n"
               "@p st.shared::cluster.v.u" +
               std::to_string(bitwidth) +
               " [remoteAddr], {$3, $4, $5, $6}; \n"
               "}\n";
    }
    return ptxAsm;
  }
};

class Sts64OpPattern : public NVGPUOpPatternBase<ttn::Sts64Op, Sts64OpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::Sts64Op, Sts64OpPattern>;
  using Base::Base;

  OperandsAndConstraints getOperandsAndConstraints(ttn::Sts64Op op) const {
    OperandsAndConstraints operandsAndTypes;
    auto offset = op.getOffset();
    auto d0 = op.getD0();
    auto d1 = op.getD1();
    operandsAndTypes.push_back({offset, "r"});
    operandsAndTypes.push_back({d0, "r"});
    operandsAndTypes.push_back({d1, "r"});
    return operandsAndTypes;
  }

  std::string getPtxAsm(ttn::Sts64Op op) const {
    return "st.shared.v2.b32 [$0], {$1, $2};";
  }
};

class LoadDSmemOpPattern
    : public NVGPUOpPatternBase<ttn::LoadDSmemOp, LoadDSmemOpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::LoadDSmemOp, LoadDSmemOpPattern>;
  using Base::Base;

  std::vector<std::string> getOutputConstraints(ttn::LoadDSmemOp op) const {
    auto bitwidth = op.getBitwidth();
    std::string c = bitwidth == 16 ? "=h" : (bitwidth == 32 ? "=r" : "=l");
    auto vec = op.getVec();
    return std::vector<std::string>(vec, c);
  }
  OperandsAndConstraints getOperandsAndConstraints(ttn::LoadDSmemOp op) const {
    OperandsAndConstraints operandsAndTypes;
    auto addr = op.getAddr();
    auto ctaId = op.getCtaId();

    operandsAndTypes.push_back({addr, "r"});
    operandsAndTypes.push_back({ctaId, "r"});
    return operandsAndTypes;
  }

  std::string getPtxAsm(ttn::LoadDSmemOp op) const {
    auto addr = op.getAddr();
    auto ctaId = op.getCtaId();
    auto bitwidth = op.getBitwidth();
    auto vec = op.getVec();

    assert(
        (bitwidth == 8 || bitwidth == 16 || bitwidth == 32 || bitwidth == 64) &&
        "invalid bitwidth");
    assert((vec == 1 || vec == 2 || vec == 4) && "invalid vec size");

    std::string o1 = vec > 1 ? ".v.u" : ".u";
    std::string vecStr = vec == 1   ? "$0"
                         : vec == 2 ? "{$0, $1}"
                                    : "{$0, $1, $2, $3}";
    unsigned argNum = vec == 1 ? 1 : vec == 2 ? 2 : 4;
    auto ptxAsm = "{\n"
                  ".reg .u32 remoteAddr;\n"
                  "mapa.shared::cluster.u32 remoteAddr, $" +
                  std::to_string(argNum) + " , $" + std::to_string(argNum + 1) +
                  " ; \n"
                  "ld.shared::cluster" +
                  o1 + std::to_string(bitwidth) + " " + vecStr +
                  ", [remoteAddr];\n"
                  "}\n";
    return ptxAsm;
  }
};

class WGMMAOpPattern : public NVGPUOpPatternBase<ttn::WGMMAOp, WGMMAOpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::WGMMAOp, WGMMAOpPattern>;
  using Base::Base;

  std::vector<std::string> getOutputConstraints(ttn::WGMMAOp op) const {
    // TODO (zahi): Return type must always be a struct for wgmma, currently
    // we rely on the size of output constraints vector to determine whether
    // the output is a struct or not. We should find a way to pass this info
    auto opC = op.getOpC();
    auto typeC = opC.getType();

    auto structTypeC = typeC.dyn_cast<LLVM::LLVMStructType>();
    uint32_t numCRegs = structTypeC.getBody().size();
    std::string c = structTypeC.getBody().front().isF32() ? "=f" : "=r";
    return std::vector<std::string>(numCRegs, c);
  }

  OperandsAndConstraints getOperandsAndConstraints(ttn::WGMMAOp op) const {
    OperandsAndConstraints operandsAndConstraints;
    auto opA = op.getOpA();
    auto opB = op.getOpB();
    auto opC = op.getOpC();
    auto typeA = opA.getType();

    auto structTypeA = typeA.dyn_cast<LLVM::LLVMStructType>();

    // TODO (zahi): is this the best way to tie inputs/outputs ?
    operandsAndConstraints.push_back({opC, 0});

    if (structTypeA) {
      operandsAndConstraints.push_back({opA, "f"});
    } else {
      operandsAndConstraints.push_back({opA, "l"});
    }

    // Operand B (must be `desc`)
    operandsAndConstraints.push_back({opB, "l"});
    return operandsAndConstraints;
  }

  std::string getPtxAsm(ttn::WGMMAOp op) const {
    using namespace ttn;
    auto opA = op.getOpA();
    auto opB = op.getOpB();
    auto opC = op.getOpC();
    auto m = op.getM();
    auto n = op.getN();
    auto k = op.getK();
    auto eltTypeC = op.getEltTypeC();
    auto eltTypeA = op.getEltTypeA();
    auto eltTypeB = op.getEltTypeB();
    auto layoutA = op.getLayoutA();
    auto layoutB = op.getLayoutB();

    // Register checks
    auto typeA = opA.getType();
    auto typeB = opB.getType();
    auto typeC = opC.getType();
    auto structTypeA = typeA.dyn_cast<LLVM::LLVMStructType>();
    auto structTypeB = typeB.dyn_cast<LLVM::LLVMStructType>();
    auto structTypeC = typeC.dyn_cast<LLVM::LLVMStructType>();
    assert(!structTypeB && "Operand B can not be registers");
    assert(structTypeC && "Operand C must be registers");

    // Element type, MNK shape and transposing support check
    // Reference:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma
    bool transA = layoutA == WGMMALayout::col;
    bool transB = layoutB == WGMMALayout::row;
    bool supported = false, needTransArgs = false, floatTypeWGMMA = false;
    assert(m % 8 == 0 && n % 8 == 0 && k % 8 == 0);
    // Below instructions do support transposing, must pass `trans` arguments
    supported |=
        (eltTypeA == WGMMAEltType::f16) && (eltTypeB == WGMMAEltType::f16) &&
        (eltTypeC == WGMMAEltType::f16 || eltTypeC == WGMMAEltType::f32) &&
        (m == 64 && 8 <= n && n <= 256 && k == 16);
    supported |= (eltTypeA == WGMMAEltType::bf16) &&
                 (eltTypeB == WGMMAEltType::bf16) &&
                 (eltTypeC == WGMMAEltType::f32) &&
                 (m == 64 && 8 <= n && n <= 256 && k == 16);
    needTransArgs = supported;
    floatTypeWGMMA = supported;
    // Below instructions do not support transposing
    if (!supported && !transA && !transB) {
      supported |= (eltTypeA == WGMMAEltType::tf32) &&
                   (eltTypeB == WGMMAEltType::tf32) &&
                   (eltTypeC == WGMMAEltType::f32) &&
                   (m == 64 && 8 <= n && n <= 256 && k == 8);
      supported |=
          (eltTypeA == WGMMAEltType::e4m3 || eltTypeA == WGMMAEltType::e5m2) &&
          (eltTypeB == WGMMAEltType::e4m3 || eltTypeB == WGMMAEltType::e5m2) &&
          (eltTypeC == WGMMAEltType::f16 || eltTypeC == WGMMAEltType::f32) &&
          (m == 64 && 8 <= n && n <= 256 && k == 32);
      floatTypeWGMMA = supported;
      // Below instructions are integer-based
      supported |= (eltTypeA == WGMMAEltType::s8) &&
                   (eltTypeB == WGMMAEltType::s8) &&
                   (eltTypeC == WGMMAEltType::s32) &&
                   (m == 64 && 8 <= n && n <= 224 && k == 32);
    }
    assert(supported && "WGMMA type or shape is not supported");

    // Operands
    uint32_t asmOpIdx = 0;

    // Operand C
    uint32_t numCRegs = structTypeC.getBody().size();

    std::string args = "";
    args += "{";
    for (uint32_t i = 0; i < numCRegs; ++i) {
      args += "$" + std::to_string(asmOpIdx++) + (i == numCRegs - 1 ? "" : ",");
    }
    args += "}, ";

    asmOpIdx += numCRegs;
    // Operand A
    if (structTypeA) {
      uint32_t numARegs = m * k / 128;
      assert(numARegs == structTypeA.getBody().size());
      args += "{";
      for (uint32_t i = 0; i < numARegs; ++i) {
        args +=
            "$" + std::to_string(asmOpIdx++) + (i == numARegs - 1 ? "" : ",");
      }
      args += "}, ";
    } else {
      args += "$" + std::to_string(asmOpIdx++) + ", ";
    }

    // Operand B (must be `desc`)
    args += "$" + std::to_string(asmOpIdx++) + ", ";

    // `scale-d` is 1 by default
    args += "1";

    // `imm-scale-a`, and `imm-scale-b` are 1 by default only for float-based
    // WGMMA
    if (floatTypeWGMMA)
      args += ", 1, 1";

    // Push `trans-a` and `trans-b` args if needed (determined as constant)
    if (needTransArgs)
      args += ", " + std::to_string(transA) + ", " + std::to_string(transB);

    auto ptxAsm = "wgmma.mma_async.sync.aligned"
                  ".m" +
                  std::to_string(m) + "n" + std::to_string(n) + "k" +
                  std::to_string(k) + "." + stringifyEnum(eltTypeC).str() +
                  "." + stringifyEnum(eltTypeA).str() + "." +
                  stringifyEnum(eltTypeB).str() + " " + args + ";";
    return ptxAsm;
  }
};

class ClusterCTAIdOpPattern
    : public NVGPUOpPatternBase<ttn::ClusterCTAIdOp, ClusterCTAIdOpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::ClusterCTAIdOp, ClusterCTAIdOpPattern>;
  using Base::Base;

  std::vector<std::string> getOutputConstraints(ttn::ClusterCTAIdOp op) const {
    return {"=r"};
  }

  std::string getPtxAsm(ttn::ClusterCTAIdOp op) const {
    auto ptxAsm = "{\n"
                  ".reg .u32 a<5>;              \n"
                  "mov.u32 a0, %cluster_ctaid.x;\n"  // x
                  "mov.u32 a1, %cluster_ctaid.y;\n"  // y
                  "mov.u32 a2, %cluster_ctaid.z;\n"  // z
                  "mov.u32 a3, %cluster_nctaid.x;\n" // nx
                  "mov.u32 a4, %cluster_nctaid.y;\n" // ny
                  "mad.lo.u32 a1, a2, a4, a1;     \n"
                  "mad.lo.u32 $0, a1, a3, a0;     \n"
                  "}";
    return ptxAsm;
  }
};

class WGMMADescCreateOpPattern
    : public NVGPUOpPatternBase<ttn::WGMMADescCreateOp,
                                WGMMADescCreateOpPattern> {
public:
  using Base =
      NVGPUOpPatternBase<ttn::WGMMADescCreateOp, WGMMADescCreateOpPattern>;
  using Base::Base;

  std::vector<std::string>
  getOutputConstraints(ttn::WGMMADescCreateOp op) const {
    return {"=l"};
  }

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::WGMMADescCreateOp op) const {
    OperandsAndConstraints operandsAndConstraints;
    auto buffer = op.getBuffer();
    auto height = op.getHeight();

    operandsAndConstraints.push_back({buffer, "l"});
    operandsAndConstraints.push_back({height, "l"});

    return operandsAndConstraints;
  }

  std::string getPtxAsm(ttn::WGMMADescCreateOp op) const {
    uint32_t mode = static_cast<uint32_t>(op.getMode());
    uint64_t swizzling = (mode == 1 ? 128 : mode == 2 ? 64 : 32);
    auto ptxAsm = "{\n"
                  ".reg .u64 a<5>;                              \n"
                  "mov.u64 a0, " +
                  std::to_string(swizzling) +
                  ";\n"
                  "shl.b64 a1, a0, 3;\n" // stride dimension
                  "shr.b64 a1, a1, 4;\n" // stride dimension
                  "mul.lo.u64 a2, $2, " +
                  std::to_string(swizzling) +
                  ";\n"                    // leadingDimension
                  "shr.b64 a2, a2, 4;\n"   // leadingDimension
                  "shl.b64 a3, $1, 46; \n" // startAddr
                  "shr.b64 a3, a3, 50; \n" // startAddr
                  "mov.u64 a4, " +
                  std::to_string(mode) +
                  "; \n" // mode
                  "shl.b64 a4, a4, 62; \n"
                  "shl.b64 a1, a1, 32; \n"
                  "or.b64 a1, a4, a1; \n"
                  "shl.b64 a2, a2, 16; \n"
                  "or.b64 a1, a1, a2; \n"
                  "or.b64 $0, a1, a3; \n"
                  "}";
    return ptxAsm;
  }
};

class OffsetOfSts64OpPattern
    : public NVGPUOpPatternBase<ttn::OffsetOfSts64Op, OffsetOfSts64OpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::OffsetOfSts64Op, OffsetOfSts64OpPattern>;
  using Base::Base;

  std::vector<std::string> getOutputConstraints(ttn::OffsetOfSts64Op op) const {
    return {"=r"};
  }

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::OffsetOfSts64Op op) const {
    OperandsAndConstraints operandsAndConstraints;
    auto threadId = op.getThreadId();
    auto rowOfWarp = op.getRowOfWarp();
    auto elemIdx = op.getElemIdx();

    operandsAndConstraints.push_back({threadId, "r"});
    operandsAndConstraints.push_back({elemIdx, "r"});
    operandsAndConstraints.push_back({rowOfWarp, "r"});

    return operandsAndConstraints;
  }

  std::string getPtxAsm(ttn::OffsetOfSts64Op op) const {
    auto rowStride = op.getRowStride();
    auto swizzleEnabled = op.getSwizzleEnabled();

    if (swizzleEnabled) {
      assert((rowStride == 32 || rowStride == 64 || rowStride == 128) &&
             "wrong rowString for swizzleEnabled");
    }

    uint32_t perPhase = 0;
    uint32_t maxPhase = 0;
    if (rowStride == 128) {
      perPhase = 1;
      maxPhase = 8;
    } else if (rowStride == 64) {
      perPhase = 2;
      maxPhase = 4;
    } else if (rowStride == 32) {
      perPhase = 4;
      maxPhase = 2;
    }

    auto ptxAsm = "{\n"
                  ".reg .u32 a<9>;      \n"
                  "and.b32 a0, $1, 0x1f;\n" // laneid
                  "shr.b32 a1, $2, 4; \n"
                  "and.b32 a1, a1, 0x1; \n"
                  "div.u32 a2, a0, 4; \n"
                  "mad.lo.u32 a2, a1, 8, a2; \n" // myRow
                  "div.u32 a3, $2, 4; \n"
                  "rem.u32 a4, a0, 4; \n"
                  "mul.lo.u32 a4, a4, 2; \n"
                  "mad.lo.u32 a4, a3, 8, a4; \n" // myCol
                  "add.u32 a2, a2, $3; \n"       // myRow = myRow + rowOfWarp
                  "div.u32 a3, a2, " +
                  std::to_string(perPhase) +
                  "; \n"
                  "rem.u32 a3, a3, " +
                  std::to_string(maxPhase) +
                  "; \n" // phase
                  "rem.u32 a5, a2, " +
                  std::to_string(perPhase) +
                  "; \n" // lineOffset
                  "mul.lo.u32 a5, a5, " +
                  std::to_string(rowStride) +
                  "; \n"
                  "mad.lo.u32 a5, a4, 4, a5; \n" // lineOffset
                  "div.u32 a6, a5, 16; \n"
                  "xor.b32 a6, a6, a3; \n" // colOffset
                  "rem.u32 a7, a5, 16; \n"
                  "mad.lo.u32 a7, a6, 16, a7; \n" // colOffset
                  "div.u32 a8, a2, " +
                  std::to_string(perPhase) +
                  "; \n"
                  "mad.lo.u32 $0, a8, 128, a7; \n" // offset
                  "}";
    return ptxAsm;
  }
};

class OffsetOfStmatrixV4OpPattern
    : public NVGPUOpPatternBase<ttn::OffsetOfStmatrixV4Op,
                                OffsetOfStmatrixV4OpPattern> {
public:
  using Base = NVGPUOpPatternBase<ttn::OffsetOfStmatrixV4Op,
                                  OffsetOfStmatrixV4OpPattern>;
  using Base::Base;

  std::vector<std::string>
  getOutputConstraints(ttn::OffsetOfStmatrixV4Op op) const {
    return {"=r"};
  }

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::OffsetOfStmatrixV4Op op) const {
    OperandsAndConstraints operandsAndConstraints;
    auto threadId = op.getThreadId();
    auto rowOfWarp = op.getRowOfWarp();
    auto elemIdx = op.getElemIdx();

    operandsAndConstraints.push_back({threadId, "r"});
    operandsAndConstraints.push_back({elemIdx, "r"});
    operandsAndConstraints.push_back({rowOfWarp, "r"});

    return operandsAndConstraints;
  }

  std::string getPtxAsm(ttn::OffsetOfStmatrixV4Op op) const {
    auto leadingDimOffset = op.getLeadingDimOffset();
    auto rowStride = op.getRowStride();
    auto swizzleEnabled = op.getSwizzleEnabled();

    std::string ptxAsm;
    if (swizzleEnabled) {
      uint32_t perPhase = 0;
      uint32_t maxPhase = 0;
      if (rowStride == 64) {
        perPhase = 1;
        maxPhase = 8;
      } else if (rowStride == 32) {
        perPhase = 2;
        maxPhase = 4;
      } else if (rowStride == 16) {
        perPhase = 4;
        maxPhase = 2;
      }

      ptxAsm =
          "{\n"
          ".reg .u32 a<10>;      \n"
          "div.u32 a0, $2, 8; \n"    // iterOfCol = udiv(elemIdx, i32_val(8))
          "and.b32 a1, $1, 0xf; \n"  // myRow = and_(threadId, i32_val(0xf))
          "add.u32 a1, a1, $3; \n"   // myRow = myRow + rowOfWarp
          "shr.b32 a2, $1, 4; \n"    // myCol = lshr(threadId, i32_val(4))
          "and.b32 a2, a2, 0x1; \n"  // myCol = and_(myCol, i32_val(0x1))
          "mul.lo.u32 a2, a2, 8; \n" // myCol = mul(myCol, i32_val(8))
          "mad.lo.u32 a2, a0, 16, a2; \n" // myCol = add(myCol,
                                          // mul(iterOfCol, i32_val(16)))
          "div.u32 a3, a2, " +
          std::to_string(rowStride) +
          "; \n" // offset0 = udiv(myCol, i32_val(rowStride))
          "mul.lo.u32 a3, a3, " +
          std::to_string(leadingDimOffset) +
          "; \n" // offset0 = mul(offset0, i32_val(leadingDimOffset))
          "rem.u32 a2, a2, " +
          std::to_string(rowStride) +
          "; \n" // myCol = myCol % rowStride
          "div.u32 a4, a1, " +
          std::to_string(perPhase) +
          "; \n" // phase =  myrow // perPhase
          "rem.u32 a4, a4, " +
          std::to_string(maxPhase) +
          "; \n" // phase = phase % maxPhase
          "rem.u32 a5, a1, " +
          std::to_string(perPhase) +
          "; \n" // lineOffset = urem(myRow, i32_val(perPhase))
          "mad.lo.u32 a5, a5, " +
          std::to_string(rowStride) +
          ", a2; \n" // lineOffset = add(mul(lineOffset, rowStride), myCol)
          "div.u32 a6, a5, 8; \n"  // colOffset = udiv(lineOffset, i32_val(8)
          "xor.b32 a6, a6, a4; \n" // colOffset = xor_(colOffset, phase)
          "rem.u32 a7, a5, 8; \n"  // temp = urem(lineOffset, i32_val(8)
          "mad.lo.u32 a7, a6, 8, a7; \n" // colOffset = add(mul(colOffset,
                                         // i32_val(8)), temp)
          "div.u32 a8, a1, " +
          std::to_string(perPhase) +
          "; \n" // offset1 = udiv(myRow, i32_val(perPhase))
          "mad.lo.u32 a9, a8, 64, a7; \n" // offset1 = add(mul(offset1,
                                          // i32_val(64)), colOffset)
          "add.u32 $0, a9, a3; \n"        // result = add(offset1, offset0)
          "}";
    } else {
      ptxAsm = "{\n"
               ".reg .u64 a<5>;      \n"
               "div.u32 a0, $2, 4; \n"    // iterOfCol = udiv(elemIdx,
                                          // i32_val(4))
               "and.b32 a1, $1, 0xf; \n"  // myRow = and_(threadId,
                                          // i32_val(0xf))
               "add.u32 a1, a1, $3; \n"   // myRow = myRow + rowOfWarp
               "shr.b32 a2, $1, 4; \n"    // myCol = lshr(threadId,
                                          // i32_val(4))
               "and.b32 a2, a2, 0x1; \n"  // myCol = and_(myCol,
                                          // i32_val(0x1))
               "mul.lo.u32 a2, a2, 8; \n" // myCol = mul(myCol,
                                          // i32_val(8))
               "mul.u32 a3, a1, " +
               std::to_string(rowStride) +
               "; \n"                         // offset = myRow * rowStride
               "mad.lo.u32 $0, a2, 2, a3; \n" // result = add(mul(myCol,
                                              // i32_val(2)), offset)
               "}\n";
    }

    return ptxAsm;
  }
};

class ConvertNVGPUToLLVM : public ConvertNVGPUToLLVMBase<ConvertNVGPUToLLVM> {

public:
  explicit ConvertNVGPUToLLVM() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    RewritePatternSet patterns(context);

    patterns.add<CGABarrierSyncOpPattern>(context);
    patterns.add<FenceAsyncSharedOpPattern>(context);
    patterns.add<WGMMAFenceOpPattern>(context);
    patterns.add<WGMMACommitGroupOpPattern>(context);
    patterns.add<WGMMAWaitGroupOpPattern>(context);
    patterns.add<StoreMatrixOpPattern>(context);
    patterns.add<OffsetOfStmatrixV4OpPattern>(context);
    patterns.add<WGMMADescCreateOpPattern>(context);
    patterns.add<MBarrierInitOpPattern>(context);
    patterns.add<MBarrierArriveOpPattern>(context);
    patterns.add<MBarrierWaitOpPattern>(context);
    patterns.add<ClusterArriveOpPattern>(context);
    patterns.add<ClusterWaitOpPattern>(context);
    patterns.add<TMALoadTiledOpPattern>(context);
    patterns.add<TMAStoreTiledOpPattern>(context);
    patterns.add<LoadDSmemOpPattern>(context);
    patterns.add<ClusterCTAIdOpPattern>(context);
    patterns.add<RegAllocOpPattern>(context);
    patterns.add<RegDeallocOpPattern>(context);
    patterns.add<WGMMAOpPattern>(context);
    patterns.add<NamedBarrierWaitOpPattern>(context);
    patterns.add<NamedBarrierArriveOpPattern>(context);

    patterns.add<FenceMBarrierInitOpPattern>(context);
    patterns.add<StoreDSmemOpPattern>(context);
    patterns.add<Sts64OpPattern>(context);
    patterns.add<OffsetOfSts64OpPattern>(context);
    patterns.add<CGABarrierWaitOpPattern>(context);
    patterns.add<CGABarrierArriveOpPattern>(context);
    if (applyPatternsAndFoldGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertNVGPUToLLVMPass() {
  return std::make_unique<::ConvertNVGPUToLLVM>();
}

} // namespace triton
} // namespace mlir
