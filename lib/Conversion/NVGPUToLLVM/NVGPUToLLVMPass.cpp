#include "triton/Conversion/NVGPUToLLVM/NVGPUToLLVMPass.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"

#include "triton/Conversion/TritonGPUToLLVM/PTXAsmFormat.h"

#include "../lib/Conversion/TritonGPUToLLVM/Utility.h"
using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Conversion/NVGPUToLLVM/Passes.h.inc"

namespace ttn = mlir::triton::nvgpu;
using ::mlir::LLVM::getSRegValue;

namespace {

using OperandsAndConstraints = std::vector<std::pair<mlir::Value, std::string>>;
typedef std::vector<std::string> Constraints;

const std::string Reg_Alloc_Op = "setmaxnreg.inc.sync.aligned.u32 #regCount;";
const std::string Wgmma_Fence_Op = "wgmma.fence.sync.aligned;";
const std::string Cga_Barrier_Sync_op = "barrier.cluster.sync.aligned;";
const std::string Wgmma_Commit_Group_Op = "wgmma.commit_group.sync.aligned;";
const std::string Cluster_Wait_Op = "barrier.cluster.wait.aligned;";
const std::string Fence_Mbarrier_Init_Op =
    "fence.mbarrier_init.release.cluster;";
const std::string Cga_Barrier_Arrive_Op = "barrier.cluster.arrive;";
const std::string Cga_Barrier_Wait_Op = "barrier.cluster.wait;";
const std::string Reg_Dealloc_Op = "setmaxnreg.dec.sync.aligned.u32 #regCount;";

const std::string Mbarrier_Init_Op =
    "@$1 mbarrier.init.shared.b64 [$0], #count;";
const std::string Mbarrier_Wait_Op =
    "{                                                           \n"
    ".reg .pred P1;                                              \n"
    "LAB_WAIT:                                                   \n"
    "mbarrier.try_wait.parity.shared.b64 P1, [$0], $1, 0x989680; \n"
    "@P1 bra.uni DONE;                                           \n"
    "bra.uni LAB_WAIT;                                           \n"
    "DONE:                                                       \n"
    "}                                                           \n";
const std::string Named_Barrier_Arrive_Op = "bar.arrive $0, $1;";
const std::string Named_Barrier_Wait_Op = "bar.sync $0, $1;";
const std::string Sts64_Op = "st.shared.v2.b32 [$0], {$1, $2};";
const std::string Cluster_Cta_Id_Op = "{\n"
                                      ".reg .u32 a<5>;              \n"
                                      "mov.u32 a0, %cluster_ctaid.x;\n"  // x
                                      "mov.u32 a1, %cluster_ctaid.y;\n"  // y
                                      "mov.u32 a2, %cluster_ctaid.z;\n"  // z
                                      "mov.u32 a3, %cluster_nctaid.x;\n" // nx
                                      "mov.u32 a4, %cluster_nctaid.y;\n" // ny
                                      "mad.lo.u32 a1, a2, a4, a1;     \n"
                                      "mad.lo.u32 $0, a1, a3, a0;     \n"
                                      "}";
const std::string Canonical_Warp_Id_Op =
    "{\n"
    ".reg .u32 a<5>;              \n"
    "mov.u32 a0, %tid.x;          \n" // x
    "mov.u32 a1, %tid.y;          \n" // y
    "mov.u32 a2, %tid.z;          \n" // z
    "mov.u32 a3, %ntid.x;         \n" // nx
    "mov.u32 a4, %ntid.y;         \n" // ny
    "mad.lo.u32 a1, a2, a4, a1;   \n"
    "mad.lo.u32 a0, a1, a3, a0;   \n"
    "shr.u32 a0, a0, 5;           \n"
    ".reg .b32         %tmp<3>;   \n"
    "mov.u32   %tmp0, -1;         \n"
    "mov.u32   %tmp1, 31;         \n"
    "mov.u32   %tmp2, 0;          \n"
    "shfl.sync.idx.b32         $0, a0, %tmp2, %tmp1, %tmp0;           \n"
    "}";

bool isNumber(const std::string &s) {
  return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) {
                         return !std::isdigit(c);
                       }) == s.end();
}

Type getTypeFromConstraint(char constraint, mlir::PatternRewriter &rewriter) {
  Type ty;
  if (constraint == 'b')
    ty = IntegerType::get(rewriter.getContext(), 1);
  else if (constraint == 'h')
    ty = IntegerType::get(rewriter.getContext(), 16);
  else if (constraint == 'r')
    ty = IntegerType::get(rewriter.getContext(), 32);
  else if (constraint == 'l')
    ty = IntegerType::get(rewriter.getContext(), 64);
  else if (constraint == 'f')
    ty = FloatType::getF32(rewriter.getContext());
  else if (constraint == 'd')
    ty = FloatType::getF64(rewriter.getContext());
  else {
    assert(false && "Unsupported constraint");
  }
  return ty;
}

template <typename SourceOp, typename ConcreteT>
class NVGPUOpPatternBase : public mlir::RewritePattern {
public:
  explicit NVGPUOpPatternBase(mlir::MLIRContext *context)
      : mlir::RewritePattern(SourceOp::getOperationName(), 1, context) {}

  // Converts the given value to the type represented by the constraint
  // E.g. if val is of type llvmptr and constraint is 'r', then we convert
  // val to i32 using ptrtoint(i32_ty, val)
  mlir::Value convertToType(mlir::Value val, std::string constraint,
                            Location &loc,
                            mlir::PatternRewriter &rewriter) const {
    auto isConstraintNumber = isNumber(constraint);
    if (!isConstraintNumber) {
      auto ty = getTypeFromConstraint(constraint[0], rewriter);
      if (val.getType().isa<LLVM::LLVMPointerType>()) {
        return ptrtoint(ty, val);
      } else {
        assert(val.getType().getIntOrFloatBitWidth() <=
                   ty.getIntOrFloatBitWidth() &&
               "Cannot convert to a smaller type");
        if (val.getType().getIntOrFloatBitWidth() < ty.getIntOrFloatBitWidth())
          return zext(ty, val);
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
      // if a constraint is a number, then we are doing input/output tying
      // if the operand is a struct, then we need to unpack it, and
      // add the constraint to each of the unpacked operands uses the constraint
      // as an offset
      auto isConstraintNumber = isNumber(constraint);
      if (llvmStruct) {
        for (unsigned i = 0; i < llvmStruct.getBody().size(); i++) {
          if (isConstraintNumber) {
            auto constraintInt = std::stoi(constraint) + i;
            unpackedOperands.push_back(
                {extract_val(llvmStruct.getBody()[i], operand, i),
                 std::to_string(constraintInt)});
          } else {
            unpackedOperands.push_back(
                {extract_val(llvmStruct.getBody()[i], operand, i), constraint});
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
      auto *ptxOperand = ptxBuilder.newOperand(convertedOperand, constraint);
      ptxOperands.push_back(ptxOperand);
    }
    return ptxOperands;
  }

  virtual std::vector<std::string> getOutputConstraints(SourceOp op) const {
    return {};
  }

  virtual OperandsAndConstraints getOperandsAndConstraints(SourceOp op) const {
    return {};
  }

  std::string patchPtxAsm(mlir::Operation *op, std::string ptxAsm) const {
    std::vector<std::pair<int, int>> patchLocations;
    std::vector<std::string> patchValues;
    auto start = ptxAsm.find("#", 0);
    while (start != std::string::npos) {
      auto endIterator =
          std::find_if(ptxAsm.begin() + start + 1, ptxAsm.end(),
                       [](unsigned char c) { return !std::isalnum(c); });

      assert(endIterator != ptxAsm.end() && "unexpected asm format");

      auto end = std::distance(ptxAsm.begin(), endIterator);
      auto patchLocation = std::make_pair(start, end);
      patchLocations.push_back(patchLocation);
      auto patchValue = ptxAsm.substr(start + 1, end - start - 1);
      patchValues.push_back(patchValue);
      start = ptxAsm.find("#", end);
    }
    assert(patchLocations.size() == patchValues.size() &&
           "patchLocations and patchValues should have the same size");
    if (patchLocations.size() == 0) {
      return ptxAsm;
    }
    std::string res = "";
    size_t prevStart = 0;
    unsigned i = 0;
    for (auto &[start, end] : patchLocations) {
      res += ptxAsm.substr(prevStart, start - prevStart);
      auto integerAttr = op->getAttrOfType<IntegerAttr>(patchValues[i]);
      auto attr = integerAttr.getInt();
      res += std::to_string(attr);
      prevStart = end;
      i++;
    }
    if (prevStart < ptxAsm.size())
      res += ptxAsm.substr(prevStart, ptxAsm.size() - prevStart);
    return res;
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
    auto ptxAsmPatched = patchPtxAsm(sourceOp, ptxAsm);
    auto hasSideEffects = !isMemoryEffectFree(sourceOp);
    auto operandsAndConstraints = concrete->getOperandsAndConstraints(sourceOp);
    auto outputConstraints = concrete->getOutputConstraints(sourceOp);

    PTXBuilder ptxBuilder;
    auto ptxOutputs = getPtxOutputs(outputConstraints, ptxBuilder);
    auto ptxOperands =
        getPtxOperands(operandsAndConstraints, ptxBuilder, loc, rewriter);
    SmallVector<PTXBuilder::Operand *> outputsAndOperands = ptxOutputs;
    outputsAndOperands.append(ptxOperands.begin(), ptxOperands.end());
    auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsmPatched);
    ptxInstr(outputsAndOperands, /*onlyAttachMLIRArgs=*/true);
    auto retTy =
        op->getNumResults() == 0 ? void_ty(ctx) : op->getResult(0).getType();
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

template <typename SourceOp>
class NVGPUOpGenericPattern
    : public NVGPUOpPatternBase<SourceOp, NVGPUOpGenericPattern<SourceOp>> {
public:
  explicit NVGPUOpGenericPattern(mlir::MLIRContext *context, std::string ptxAsm,
                                 std::vector<std::string> outputConstraints,
                                 std::vector<std::string> inputConstraints)
      : NVGPUOpPatternBase<SourceOp, NVGPUOpGenericPattern<SourceOp>>(context),
        ptxAsm(ptxAsm), outputConstraints(outputConstraints),
        inputConstraints(inputConstraints) {}

  std::vector<std::string> getOutputConstraints(SourceOp op) const {
    return outputConstraints;
  }
  OperandsAndConstraints getOperandsAndConstraints(SourceOp op) const {
    OperandsAndConstraints operandsAndConstraints;
    for (unsigned i = 0; i < inputConstraints.size(); i++) {
      operandsAndConstraints.push_back(
          {op->getOperand(i), inputConstraints[i]});
    }
    return operandsAndConstraints;
  }
  std::string getPtxAsm(SourceOp op) const { return ptxAsm; }

private:
  std::string ptxAsm;
  std::vector<std::string> outputConstraints;
  std::vector<std::string> inputConstraints;
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
      ptxAsm = "@$4 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
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
               "@p st.shared::cluster.u#bitwidth [remoteAddr], $3; \n"
               "}\n";
    }
    if (vec == 2) {
      ptxAsm = "{                                           \n"
               ".reg .u32 remoteAddr;                       \n"
               "mapa.shared::cluster.u32 remoteAddr, $0, $1;\n"
               ".reg .pred p;                               \n"
               "mov.pred p, $2;                             \n"
               "@p st.shared::cluster.v.u#bitwidth [remoteAddr], {$3, $4}; \n"
               "}\n";
    }
    if (vec == 4) {
      ptxAsm = "{                                           \n"
               ".reg .u32 remoteAddr;                       \n"
               "mapa.shared::cluster.u32 remoteAddr, $0, $1;\n"
               ".reg .pred p;                               \n"
               "mov.pred p, $2;                             \n"
               "@p st.shared::cluster.v.u#bitwidth [remoteAddr], {$3, $4, $5, "
               "$6}; \n"
               "}\n";
    }
    return ptxAsm;
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

class WGMMAWaitGroupOpPattern
    : public NVGPUOpPatternBase<ttn::WGMMAWaitGroupOp,
                                WGMMAWaitGroupOpPattern> {
public:
  using Base =
      NVGPUOpPatternBase<ttn::WGMMAWaitGroupOp, WGMMAWaitGroupOpPattern>;
  using Base::Base;

  std::vector<std::string>
  getOutputConstraints(ttn::WGMMAWaitGroupOp op) const {
    auto outputStructType = op.getType().cast<LLVM::LLVMStructType>();
    uint32_t numOutputRegs = outputStructType.getBody().size();
    std::string output =
        outputStructType.getBody().front().isF32() ? "=f" : "=r";
    return std::vector<std::string>(numOutputRegs, output);
  }

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::WGMMAWaitGroupOp op) const {
    OperandsAndConstraints operandsAndConstraints;
    auto input = op.getInput();
    operandsAndConstraints.push_back({input, "0"});
    return operandsAndConstraints;
  }

  std::string getPtxAsm(ttn::WGMMAWaitGroupOp op) const {
    auto outputStructType = op.getType().dyn_cast<LLVM::LLVMStructType>();
    uint32_t numCRegs = outputStructType.getBody().size();
    std::string args = "";
    uint32_t asmOpIdx = 0;
    for (uint32_t i = 0; i < numCRegs; ++i) {
      args += "$" + std::to_string(asmOpIdx++) + (i == numCRegs - 1 ? "" : ",");
    }
    auto ptxAsm = "// wait for regs: " + args + "\n\t" +
                  "wgmma.wait_group.sync.aligned #pendings;";
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
    auto resultType = op.getType();

    auto outputStructType = resultType.dyn_cast<LLVM::LLVMStructType>();
    uint32_t numOutputRegs = outputStructType.getBody().size();
    std::string output =
        outputStructType.getBody().front().isF32() ? "=f" : "=r";
    return std::vector<std::string>(numOutputRegs, output);
  }

  OperandsAndConstraints getOperandsAndConstraints(ttn::WGMMAOp op) const {
    OperandsAndConstraints operandsAndConstraints;
    auto opA = op.getOpA();
    auto opB = op.getOpB();
    auto opC = op.getOpC();
    auto typeA = opA.getType();

    auto structTypeA = typeA.dyn_cast<LLVM::LLVMStructType>();

    // TODO (zahi): is this the best way to tie inputs/outputs ?
    if (opC)
      operandsAndConstraints.push_back({opC, "0"});

    if (structTypeA) {
      operandsAndConstraints.push_back({opA, "r"});
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
    auto typeOutput = op.getType();
    auto structTypeA = typeA.dyn_cast<LLVM::LLVMStructType>();
    auto structTypeB = typeB.dyn_cast<LLVM::LLVMStructType>();
    auto structTypeOutput = typeOutput.dyn_cast<LLVM::LLVMStructType>();
    assert(!structTypeB && "Operand B can not be registers");
    assert(structTypeOutput && "Output and C operand must be registers");

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
    std::string args = "";

    // Output and operand C
    uint32_t numCRegs = structTypeOutput.getBody().size();

    args += "{";
    for (uint32_t i = 0; i < numCRegs; ++i) {
      args += "$" + std::to_string(asmOpIdx++) + (i == numCRegs - 1 ? "" : ",");
    }
    args += "}, ";

    if (op.getOpC())
      asmOpIdx += numCRegs;

    // Operand A
    if (structTypeA) {
      uint32_t numARegs = structTypeA.getBody().size();
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

    // `scale-d` is 1 if we have a C operand.
    args += op.getOpC() ? "1" : "0";

    // `imm-scale-a`, and `imm-scale-b` are 1 by default only for float-based
    // WGMMA
    if (floatTypeWGMMA)
      args += ", 1, 1";

    // Push `trans-a` and `trans-b` args if needed (determined as constant)
    if (needTransArgs) {
      if (!structTypeA)
        args += ", " + std::to_string(transA);
      args += ", " + std::to_string(transB);
    }

    auto ptxAsm = "wgmma.mma_async.sync.aligned"
                  ".m" +
                  std::to_string(m) + "n" + std::to_string(n) + "k" +
                  std::to_string(k) + "." + stringifyEnum(eltTypeC).str() +
                  "." + stringifyEnum(eltTypeA).str() + "." +
                  stringifyEnum(eltTypeB).str() + " " + args + ";";
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
    } else {
      assert(false && "Unsupported rowStride");
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
                  "mul.lo.u32 a5, a5, #rowStride; \n"
                  "mad.lo.u32 a5, a4, 4, a5; \n" // lineOffset
                  "div.u32 a6, a5, 16; \n"
                  "xor.b32 a6, a6, a3; \n" // colOffset
                  "rem.u32 a7, a5, 16; \n"
                  "mad.lo.u32 a7, a6, 16, a7; \n" // colOffset
                  "div.u32 a8, a2, #perPhase; \n"
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
      } else {
        assert(false && "Unsupported rowStride");
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
          "mad.lo.u32 a2, a0, 16, a2; \n"  // myCol = add(myCol,
                                           // mul(iterOfCol, i32_val(16)))
          "div.u32 a3, a2, #rowStride; \n" // offset0 = udiv(myCol,
                                           // i32_val(rowStride))
          "mul.lo.u32 a3, a3, #leadingDimOffset; \n" // offset0 = mul(offset0,
                                                     // i32_val(leadingDimOffset))
          "rem.u32 a2, a2, #rowStride; \n" // myCol = myCol % rowStride
          "div.u32 a4, a1, " +
          std::to_string(perPhase) +
          "; \n" // phase =  myrow // perPhase
          "rem.u32 a4, a4, " +
          std::to_string(maxPhase) +
          "; \n" // phase = phase % maxPhase
          "rem.u32 a5, a1, " +
          std::to_string(perPhase) +
          "; \n" // lineOffset = urem(myRow, i32_val(perPhase))
          "mad.lo.u32 a5, a5, #rowStride, a2; \n" // lineOffset =
                                                  // add(mul(lineOffset,
                                                  // rowStride), myCol)
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
               "div.u32 a0, $2, 4; \n"          // iterOfCol = udiv(elemIdx,
                                                // i32_val(4))
               "and.b32 a1, $1, 0xf; \n"        // myRow = and_(threadId,
                                                // i32_val(0xf))
               "add.u32 a1, a1, $3; \n"         // myRow = myRow + rowOfWarp
               "shr.b32 a2, $1, 4; \n"          // myCol = lshr(threadId,
                                                // i32_val(4))
               "and.b32 a2, a2, 0x1; \n"        // myCol = and_(myCol,
                                                // i32_val(0x1))
               "mul.lo.u32 a2, a2, 8; \n"       // myCol = mul(myCol,
                                                // i32_val(8))
               "mul.u32 a3, a1, #rowStride; \n" // offset = myRow * rowStride
               "mad.lo.u32 $0, a2, 2, a3; \n"   // result = add(mul(myCol,
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

#define POPULATE_NVGPU_OP(SRC_OP, ASM)                                         \
  patterns.add<NVGPUOpGenericPattern<SRC_OP>>(context, ASM, Constraints(),     \
                                              Constraints());
    POPULATE_NVGPU_OP(ttn::RegAllocOp, Reg_Alloc_Op)
    POPULATE_NVGPU_OP(ttn::WGMMAFenceOp, Wgmma_Fence_Op)
    POPULATE_NVGPU_OP(ttn::CGABarrierSyncOp, Cga_Barrier_Sync_op)
    POPULATE_NVGPU_OP(ttn::WGMMACommitGroupOp, Wgmma_Commit_Group_Op)
    POPULATE_NVGPU_OP(ttn::ClusterWaitOp, Cluster_Wait_Op)
    POPULATE_NVGPU_OP(ttn::FenceMBarrierInitOp, Fence_Mbarrier_Init_Op)
    POPULATE_NVGPU_OP(ttn::CGABarrierArriveOp, Cga_Barrier_Arrive_Op)
    POPULATE_NVGPU_OP(ttn::CGABarrierWaitOp, Cga_Barrier_Wait_Op)
    POPULATE_NVGPU_OP(ttn::RegDeallocOp, Reg_Dealloc_Op)
#undef POPULATE_NVGPU_OP
    patterns.add<NVGPUOpGenericPattern<ttn::MBarrierInitOp>>(
        context, Mbarrier_Init_Op, Constraints(), Constraints({"r", "b"}));
    patterns.add<NVGPUOpGenericPattern<ttn::MBarrierWaitOp>>(
        context, Mbarrier_Wait_Op, Constraints(), Constraints({"r", "r"}));
    patterns.add<NVGPUOpGenericPattern<ttn::NamedBarrierArriveOp>>(
        context, Named_Barrier_Arrive_Op, Constraints(),
        Constraints({"r", "r"}));
    patterns.add<NVGPUOpGenericPattern<ttn::NamedBarrierWaitOp>>(
        context, Named_Barrier_Wait_Op, Constraints(), Constraints({"r", "r"}));
    patterns.add<NVGPUOpGenericPattern<ttn::Sts64Op>>(
        context, Sts64_Op, Constraints(), Constraints({"r", "r", "r"}));
    patterns.add<NVGPUOpGenericPattern<ttn::ClusterCTAIdOp>>(
        context, Cluster_Cta_Id_Op, Constraints({"=r"}), Constraints());
    patterns.add<NVGPUOpGenericPattern<ttn::CanonicalWarpIdOp>>(
        context, Canonical_Warp_Id_Op, Constraints({"=r"}), Constraints());

    patterns.add<FenceAsyncSharedOpPattern, StoreMatrixOpPattern,
                 OffsetOfStmatrixV4OpPattern, MBarrierArriveOpPattern,
                 ClusterArriveOpPattern, TMALoadTiledOpPattern,
                 TMAStoreTiledOpPattern, LoadDSmemOpPattern, WGMMAOpPattern,
                 WGMMAWaitGroupOpPattern, StoreDSmemOpPattern,
                 OffsetOfSts64OpPattern>(context);

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
