#include "triton/Conversion/NVGPUToLLVM/NVGPUToLLVMPass.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"

#include "triton/Conversion/TritonGPUToLLVM/PTXAsmFormat.h"

#include "../lib/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Conversion/NVGPUToLLVM/Passes.h.inc"

namespace {
class CGABarrierSyncOpPattern : public mlir::RewritePattern {
public:
  CGABarrierSyncOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::CGABarrierSyncOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto cgaBarrierSyncOp =
        llvm::dyn_cast<mlir::triton::nvgpu::CGABarrierSyncOp>(op);
    if (!cgaBarrierSyncOp)
      return mlir::failure();
    auto loc = op->getLoc();
    PTXBuilder ptxBuilder;
    auto &ptxInstr =
        *ptxBuilder.create<PTXInstr>("barrier.cluster.sync.aligned");
    ptxInstr();
    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class FenceAsyncSharedOpPattern : public mlir::RewritePattern {
public:
  FenceAsyncSharedOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::FenceAsyncSharedOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto fenceAsyncSharedOp =
        llvm::dyn_cast<mlir::triton::nvgpu::FenceAsyncSharedOp>(op);
    if (!fenceAsyncSharedOp)
      return mlir::failure();
    auto loc = op->getLoc();
    auto bCluster = fenceAsyncSharedOp.getBCluster();
    PTXBuilder ptxBuilder;
    if (bCluster) {
      auto &ptxInstr =
          *ptxBuilder.create<PTXInstr>("fence.proxy.async.shared::cluster");
      ptxInstr();
    } else {
      auto &ptxInstr =
          *ptxBuilder.create<PTXInstr>("fence.proxy.async.shared::cta");
      ptxInstr();
    }
    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class WGMMAFenceOpPattern : public mlir::RewritePattern {
public:
  WGMMAFenceOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::WGMMAFenceOp::getOperationName(), 1, context) {
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto wgmmaFenceOp = llvm::dyn_cast<mlir::triton::nvgpu::WGMMAFenceOp>(op);
    if (!wgmmaFenceOp)
      return mlir::failure();
    auto loc = op->getLoc();
    PTXBuilder ptxBuilder;

    auto &ptxInstr = *ptxBuilder.create<PTXInstr>("wgmma.fence.sync.aligned");
    ptxInstr();

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy, /*hasSideEffect*/ true);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class WGMMACommitGroupOpPattern : public mlir::RewritePattern {
public:
  WGMMACommitGroupOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::WGMMACommitGroupOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto wgmmaCommitGroupOp =
        llvm::dyn_cast<mlir::triton::nvgpu::WGMMACommitGroupOp>(op);
    if (!wgmmaCommitGroupOp)
      return mlir::failure();
    auto loc = op->getLoc();
    PTXBuilder ptxBuilder;

    auto &ptxInstr =
        *ptxBuilder.create<PTXInstr>("wgmma.commit_group.sync.aligned");
    ptxInstr();

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy, /*hasSideEffect*/ true);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class WGMMAWaitGroupOpPattern : public mlir::RewritePattern {
public:
  WGMMAWaitGroupOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::WGMMAWaitGroupOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto wgmmaWaitGroupOp =
        llvm::dyn_cast<mlir::triton::nvgpu::WGMMAWaitGroupOp>(op);
    if (!wgmmaWaitGroupOp)
      return mlir::failure();
    auto loc = op->getLoc();
    auto pendings = wgmmaWaitGroupOp.getPendings();
    PTXBuilder ptxBuilder;

    auto &ptxInstr =
        *ptxBuilder.create<PTXInstr>("wgmma.wait_group.sync.aligned");
    ptxInstr(ptxBuilder.newConstantOperand(pendings));

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy, /*hasSideEffect*/ true);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class StoreMatrixOpPattern : public mlir::RewritePattern {
public:
  StoreMatrixOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::StoreMatrixOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto storeMatrixOp = llvm::dyn_cast<mlir::triton::nvgpu::StoreMatrixOp>(op);
    if (!storeMatrixOp)
      return mlir::failure();
    auto loc = op->getLoc();
    auto addr = storeMatrixOp.getAddr();
    auto datas = storeMatrixOp.getDatas();

    assert(datas.size() == 1 || datas.size() == 2 ||
           datas.size() == 4 && "Invalid size for StoreMatrixOp");
    PTXBuilder ptxBuilder;
    auto &ptxInstr = *ptxBuilder.create<PTXInstr>(
        "stmatrix.sync.aligned.m8n8.x" + std::to_string(datas.size()) +
        ".shared.b16");
    auto *addrOpr = ptxBuilder.newAddrOperand(addr, "r", 0);

    SmallVector<std::pair<Value, std::string>> args;
    for (unsigned i = 0; i < datas.size(); ++i) {
      args.push_back({datas[i], "r"});
    }
    auto *operands = ptxBuilder.newListOperand(args);

    ptxInstr(addrOpr, operands);

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CvtPackOpPattern : public mlir::RewritePattern {
public:
  CvtPackOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::triton::nvgpu::CvtPackOp::getOperationName(),
                             1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto cvtPackOp = llvm::dyn_cast<mlir::triton::nvgpu::CvtPackOp>(op);
    if (!cvtPackOp)
      return mlir::failure();
    auto loc = op->getLoc();
    auto d0 = cvtPackOp.getD0();
    auto d1 = cvtPackOp.getD1();

    PTXBuilder ptxBuilder;
    auto &ptxInstr = *ptxBuilder.create<PTXInstr>("cvt.pack.sat.u16.s32");
    auto *ret = ptxBuilder.newOperand("=r");
    auto *d0Opr = ptxBuilder.newOperand(d0, "r");
    auto *d1Opr = ptxBuilder.newOperand(d1, "r");

    ptxInstr(ret, d0Opr, d1Opr);

    auto asmReturnTy = rewriter.getIntegerType(32);
    auto res = ptxBuilder.launch(rewriter, loc, asmReturnTy);
    rewriter.replaceOp(op, {res});
    return mlir::success();
  }
};

class OffsetOfStmatrixV4OpPattern : public mlir::RewritePattern {
public:
  OffsetOfStmatrixV4OpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::OffsetOfStmatrixV4Op::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto offsetOfStmatrixV4Op =
        llvm::dyn_cast<mlir::triton::nvgpu::OffsetOfStmatrixV4Op>(op);
    if (!offsetOfStmatrixV4Op)
      return mlir::failure();
    auto loc = op->getLoc();
    auto threadId = offsetOfStmatrixV4Op.getThreadId();
    auto rowOfWarp = offsetOfStmatrixV4Op.getRowOfWarp();
    auto elemIdx = offsetOfStmatrixV4Op.getElemIdx();
    auto leadingDimOffset = offsetOfStmatrixV4Op.getLeadingDimOffset();
    auto rowStride = offsetOfStmatrixV4Op.getRowStride();
    auto swizzleEnabled = offsetOfStmatrixV4Op.getSwizzleEnabled();

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

      Value iterOfCol = udiv(elemIdx, i32_val(8));
      Value myRow = add(rowOfWarp, and_(threadId, i32_val(0xf)));
      Value myCol =
          mul(and_(lshr(threadId, i32_val(4)), i32_val(0x1)), i32_val(8));
      myCol = add(myCol, mul(iterOfCol, i32_val(16)));

      Value offset0 =
          mul(udiv(myCol, i32_val(rowStride)), i32_val(leadingDimOffset));
      myCol = urem(myCol, i32_val(rowStride));

      Value phase = urem(udiv(myRow, i32_val(perPhase)), i32_val(maxPhase));

      Value lineOffset =
          add(mul(urem(myRow, i32_val(perPhase)), i32_val(rowStride)), myCol);
      Value colOffset =
          add(mul(xor_(udiv(lineOffset, i32_val(8)), phase), i32_val(8)),
              urem(lineOffset, i32_val(8)));
      Value offset1 =
          add(mul(udiv(myRow, i32_val(perPhase)), i32_val(64)), colOffset);

      Value res = add(offset1, offset0);

      rewriter.replaceOp(op, {res});
    } else {
      Value iterOfCol = udiv(elemIdx, i32_val(4));
      Value myRow = add(rowOfWarp, and_(threadId, i32_val(0xf)));
      Value myCol =
          mul(and_(lshr(threadId, i32_val(4)), i32_val(0x1)), i32_val(8));
      myCol = add(myCol, mul(iterOfCol, i32_val(16)));

      Value offset =
          add(mul(myRow, i32_val(rowStride)), mul(myCol, i32_val(2)));
      rewriter.replaceOp(op, {offset});
    }
    return mlir::success();
  }
};

class WGMMADescCreateOpPattern : public mlir::RewritePattern {
public:
  WGMMADescCreateOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::WGMMADescCreateOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto wgmmaDescCreateOp =
        llvm::dyn_cast<mlir::triton::nvgpu::WGMMADescCreateOp>(op);
    if (!wgmmaDescCreateOp)
      return mlir::failure();
    auto loc = op->getLoc();
    auto buffer = wgmmaDescCreateOp.getBuffer();
    auto height = wgmmaDescCreateOp.getHeight();
    uint32_t mode = static_cast<uint32_t>(wgmmaDescCreateOp.getMode());

    auto smem_nvvm_pointer = ptrtoint(i64_ty, buffer);

    Value desc = int_val(64, 0);
    uint64_t swizzling = (mode == 1 ? 128 : mode == 2 ? 64 : 32);
    Value swizzling_ = int_val(64, swizzling);
    Value smem_address_bit = smem_nvvm_pointer;

    Value strideDimension =
        lshr(shl(swizzling_, int_val(64, 3)), int_val(64, 4));
    Value height64 = zext(i64_ty, height);
    Value leadingDimension = lshr(mul(height64, swizzling_), int_val(64, 4));

    // Value baseOffset = int_val(64, 0);
    Value startAddr =
        lshr(shl(smem_address_bit, int_val(64, 46)), int_val(64, 50));

    Value mode_ = int_val(64, mode);
    desc = or_(desc, shl(mode_, int_val(64, 62)));
    desc = or_(desc, shl(strideDimension, int_val(64, 32)));
    desc = or_(desc, shl(leadingDimension, int_val(64, 16)));
    // desc = or_(desc, shl(baseOffset, int_val(64, 49)));
    desc = or_(desc, startAddr);

    rewriter.replaceOp(op, {desc});
    return mlir::success();
  }
};

class MBarrierInitOpPattern : public mlir::RewritePattern {
public:
  MBarrierInitOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::MBarrierInitOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto mBarrierInitOp =
        llvm::dyn_cast<mlir::triton::nvgpu::MBarrierInitOp>(op);
    if (!mBarrierInitOp)
      return mlir::failure();
    auto loc = op->getLoc();
    Value mbarrier = mBarrierInitOp.getMbarrier();
    Value pred = mBarrierInitOp.getPred();
    uint32_t count = mBarrierInitOp.getCount();
    PTXBuilder ptxBuilder;

    auto &ptxInstr = *ptxBuilder.create<PTXInstr>("mbarrier.init.shared.b64");
    auto *barOpr =
        ptxBuilder.newAddrOperand(ptrtoint(i32_ty, mbarrier), "r", 0);
    auto *expectedOpr = ptxBuilder.newConstantOperand(count);

    ptxInstr(barOpr, expectedOpr).predicate(pred, "b");

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class MBarrierArriveOpPattern : public mlir::RewritePattern {
public:
  MBarrierArriveOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::MBarrierArriveOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto mbarrierArriveOp =
        llvm::dyn_cast<mlir::triton::nvgpu::MBarrierArriveOp>(op);
    if (!mbarrierArriveOp)
      return mlir::failure();
    auto loc = op->getLoc();
    Value mbarrier = mbarrierArriveOp.getMbarrier();
    Value pred = mbarrierArriveOp.getPred();
    Value ctaId = mbarrierArriveOp.getCtaId();
    auto arriveType = mbarrierArriveOp.getArriveType();
    uint32_t txCount = mbarrierArriveOp.getTxCount();

    PTXBuilder ptxBuilder;
    if (arriveType == mlir::triton::nvgpu::MBarriveType::normal) {
      auto &ptxInstr =
          *ptxBuilder.create<PTXInstr>("mbarrier.arrive.shared.b64 _,");
      auto *barOpr =
          ptxBuilder.newAddrOperand(ptrtoint(i32_ty, mbarrier), "r", 0);

      ptxInstr(barOpr).predicate(pred, "b");
    } else if (arriveType == mlir::triton::nvgpu::MBarriveType::cp_async) {
      assert(txCount > 0 && "txCount should be valid");
      auto &ptxInstr = *ptxBuilder.create<PTXInstr>(
          "cp.async.mbarrier.arrive.noinc.shared.b64");
      auto *barOpr =
          ptxBuilder.newAddrOperand(ptrtoint(i32_ty, mbarrier), "r", 0);

      ptxInstr(barOpr).predicate(pred, "b");
    } else if (arriveType == mlir::triton::nvgpu::MBarriveType::expect_tx) {
      auto &ptxInstr = *ptxBuilder.create<PTXInstr>(
          "mbarrier.arrive.expect_tx.shared.b64 _,");
      auto *barOpr =
          ptxBuilder.newAddrOperand(ptrtoint(i32_ty, mbarrier), "r", 0);
      auto *expectedOpr = ptxBuilder.newConstantOperand(txCount);

      ptxInstr(barOpr, expectedOpr).predicate(pred, "b");
    } else if (arriveType == mlir::triton::nvgpu::MBarriveType::remote) {
      auto ptxAsm =
          " { .reg .b32 remAddr32;                                       \n"
          "  @$2 mapa.shared::cluster.u32  remAddr32, $0, $1;            \n"
          "  @$2 mbarrier.arrive.shared::cluster.b64  _, [remAddr32]; }  \n";
      auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);
      auto *barOpr =
          ptxBuilder.newAddrOperand(ptrtoint(i32_ty, mbarrier), "r", 0);
      auto *ctaIdOpr = ptxBuilder.newOperand(ctaId, "r");
      auto *predOpr = ptxBuilder.newOperand(pred, "b");

      ptxInstr({barOpr, ctaIdOpr, predOpr}, /*onlyAttachMLIRArgs=*/true);
    } else {
      assert(false &&
             "Unsupported mbarrier arrive type"); // TODO: is this the right way
                                                  // to assert in LLVM pass ?
    }
    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};
class MBarrierWaitOpPattern : public mlir::RewritePattern {
public:
  MBarrierWaitOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::MBarrierWaitOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto mBarrierWaitOp =
        llvm::dyn_cast<mlir::triton::nvgpu::MBarrierWaitOp>(op);
    if (!mBarrierWaitOp)
      return mlir::failure();
    auto loc = op->getLoc();
    Value mbarrier = mBarrierWaitOp.getMbarrier();
    Value phase = mBarrierWaitOp.getPhase();
    Value largeVal = i32_val(0x989680);
    PTXBuilder ptxBuilder;

    auto ptxAsm = "{\n"
                  ".reg .pred                P1; \n"
                  "LAB_WAIT: \n"
                  "mbarrier.try_wait.parity.shared.b64 P1, [$0], $1, $2; \n"
                  "@P1                       bra.uni DONE; \n"
                  "bra.uni                   LAB_WAIT; \n"
                  "DONE: \n"
                  "}";
    auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);
    auto *barOpr =
        ptxBuilder.newAddrOperand(ptrtoint(i32_ty, mbarrier), "r", 0);
    auto *phaseOpr = ptxBuilder.newOperand(zext(i32_ty, phase), "r");
    auto *largeValOpr = ptxBuilder.newOperand(largeVal, "r");

    ptxInstr({barOpr, phaseOpr, largeValOpr},
             /*onlyAttachMLIRArgs=*/true);

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ClusterArriveOpPattern : public mlir::RewritePattern {
public:
  ClusterArriveOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::ClusterArriveOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto clusterArriveOp =
        llvm::dyn_cast<mlir::triton::nvgpu::ClusterArriveOp>(op);
    if (!clusterArriveOp)
      return mlir::failure();
    auto loc = op->getLoc();
    auto relaxed = clusterArriveOp.getRelaxed();

    PTXBuilder ptxBuilder;
    std::string ptxAsm;
    if (relaxed) {
      ptxAsm = "barrier.cluster.arrive.relaxed.aligned";
    } else {
      ptxAsm = "barrier.cluster.arrive.aligned";
    }

    auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);
    ptxInstr();

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ClusterWaitOpPattern : public mlir::RewritePattern {
public:
  ClusterWaitOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::ClusterWaitOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto clusterWaitOp = llvm::dyn_cast<mlir::triton::nvgpu::ClusterWaitOp>(op);
    if (!clusterWaitOp)
      return mlir::failure();
    auto loc = op->getLoc();

    PTXBuilder ptxBuilder;
    auto ptxAsm = "barrier.cluster.wait.aligned";

    auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);
    ptxInstr();

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class TMALoadTiledOpPattern : public mlir::RewritePattern {
public:
  TMALoadTiledOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::TMALoadTiledOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto tmaLoadTiledOp =
        llvm::dyn_cast<mlir::triton::nvgpu::TMALoadTiledOp>(op);
    if (!tmaLoadTiledOp)
      return mlir::failure();
    auto loc = op->getLoc();
    auto dst = tmaLoadTiledOp.getDst();
    auto mbarrier = tmaLoadTiledOp.getMbarrier();
    auto tmaDesc = tmaLoadTiledOp.getTmaDesc();
    auto l2Desc = tmaLoadTiledOp.getL2Desc();
    auto pred = tmaLoadTiledOp.getPred();
    auto coords = tmaLoadTiledOp.getCoords();
    auto mcastMask = tmaLoadTiledOp.getMcastMask();

    auto dimSize = coords.size();

    PTXBuilder ptxBuilder;
    if (dimSize == 2) {
      if (mcastMask == nullptr) {
        auto ptxAsm =
            "@$6 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier:"
            ":complete_tx"
            "::bytes.L2::cache_hint [$0], [$1, {$2, $3}], "
            "[$4], $5;";
        auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);
        auto *dstOpr = ptxBuilder.newOperand(ptrtoint(i32_ty, dst), "r");
        auto *descOpr = ptxBuilder.newOperand(ptrtoint(i64_ty, tmaDesc), "l");
        auto *c0Opr = ptxBuilder.newOperand(coords[0], "r");
        auto *c1Opr = ptxBuilder.newOperand(coords[1], "r");
        auto *barOpr = ptxBuilder.newOperand(ptrtoint(i64_ty, mbarrier), "r");
        auto *l2DescOpr = ptxBuilder.newOperand(l2Desc, "l");
        auto *predOpr = ptxBuilder.newOperand(pred, "b");

        ptxInstr({dstOpr, descOpr, c0Opr, c1Opr, barOpr, l2DescOpr, predOpr},
                 /*onlyAttachMLIRArgs=*/true);
      } else {
        auto ptxAsm =
            "@$7 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
            "complete_tx::bytes.multicast::cluster.L2::cache_hint"
            " [$0], [$1, {$2, $3}], [$4], $5, $6;";
        auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);
        auto *dstOpr = ptxBuilder.newOperand(ptrtoint(i32_ty, dst), "r");
        auto *descOpr = ptxBuilder.newOperand(ptrtoint(i64_ty, tmaDesc), "l");
        auto *c0Opr = ptxBuilder.newOperand(coords[0], "r");
        auto *c1Opr = ptxBuilder.newOperand(coords[1], "r");
        auto *barOpr = ptxBuilder.newOperand(ptrtoint(i64_ty, mbarrier), "r");
        auto *maskOpr = ptxBuilder.newOperand(mcastMask, "h");
        auto *l2DescOpr = ptxBuilder.newOperand(l2Desc, "l");
        auto *predOpr = ptxBuilder.newOperand(pred, "b");
        ptxInstr({dstOpr, descOpr, c0Opr, c1Opr, barOpr, maskOpr, l2DescOpr,
                  predOpr},
                 /*onlyAttachMLIRArgs=*/true);
      }
    } else if (dimSize == 4) {
      assert(mcastMask == nullptr && "Does not support multicast");
      auto ptxAsm =
          "@$8 "
          "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx"
          "::bytes.L2::cache_hint [$0], [$1, {$2, $3, $4, $5}], "
          "[$6], $7;";
      auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);
      auto *dstOpr = ptxBuilder.newOperand(ptrtoint(i32_ty, dst), "r");
      auto *descOpr = ptxBuilder.newOperand(ptrtoint(i64_ty, tmaDesc), "l");
      auto *c0Opr = ptxBuilder.newOperand(coords[0], "r");
      auto *c1Opr = ptxBuilder.newOperand(coords[1], "r");
      auto *c2Opr = ptxBuilder.newOperand(coords[2], "r");
      auto *c3Opr = ptxBuilder.newOperand(coords[3], "r");
      auto *barOpr = ptxBuilder.newOperand(ptrtoint(i64_ty, mbarrier), "r");
      auto *l2DescOpr = ptxBuilder.newOperand(l2Desc, "l");
      auto *predOpr = ptxBuilder.newOperand(pred, "b");
      ptxInstr({dstOpr, descOpr, c0Opr, c1Opr, c2Opr, c3Opr, barOpr, l2DescOpr,
                predOpr},
               /*onlyAttachMLIRArgs=*/true);
    } else {
      assert(false && "invalid dim size");
    }

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy, /*hasSideEffect*/ true);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class TMAStoreTiledOpPattern : public mlir::RewritePattern {
public:
  TMAStoreTiledOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::TMAStoreTiledOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto tmaStoreTiledOp =
        llvm::dyn_cast<mlir::triton::nvgpu::TMAStoreTiledOp>(op);
    if (!tmaStoreTiledOp)
      return mlir::failure();
    auto loc = op->getLoc();
    auto src = tmaStoreTiledOp.getSrc();
    auto tmaDesc = tmaStoreTiledOp.getTmaDesc();
    auto pred = tmaStoreTiledOp.getPred();
    auto coords = tmaStoreTiledOp.getCoords();

    auto dimSize = coords.size();

    PTXBuilder ptxBuilder;
    if (dimSize == 2) {
      auto ptxAsm = "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
                    "[$0, {$2, $3}], [$1];";
      auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);

      auto *descOpr = ptxBuilder.newOperand(ptrtoint(i64_ty, tmaDesc), "l");
      auto *srcOpr = ptxBuilder.newOperand(ptrtoint(i32_ty, src), "r");
      auto *c0Opr = ptxBuilder.newOperand(coords[0], "r");
      auto *c1Opr = ptxBuilder.newOperand(coords[1], "r");
      auto *predOpr = ptxBuilder.newOperand(pred, "b");
      ptxInstr({descOpr, srcOpr, c0Opr, c1Opr, predOpr},
               /*onlyAttachMLIRArgs=*/true);
    } else if (dimSize == 3) {
      auto ptxAsm = "@$5 cp.async.bulk.tensor.3d.global.shared::cta.bulk_group"
                    "[$0, {$2, $3, $4}], [$1];";
      auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);

      auto *descOpr = ptxBuilder.newOperand(ptrtoint(i64_ty, tmaDesc), "l");
      auto *srcOpr = ptxBuilder.newOperand(ptrtoint(i32_ty, src), "r");
      auto *c0Opr = ptxBuilder.newOperand(coords[0], "r");
      auto *c1Opr = ptxBuilder.newOperand(coords[1], "r");
      auto *c2Opr = ptxBuilder.newOperand(coords[2], "r");
      auto *predOpr = ptxBuilder.newOperand(pred, "b");
      ptxInstr({descOpr, srcOpr, c0Opr, c1Opr, c2Opr, predOpr},
               /*onlyAttachMLIRArgs=*/true);
    } else if (dimSize == 4) {
      auto ptxAsm = "@$6 cp.async.bulk.tensor.4d.global.shared::cta.bulk_group"
                    "[$0, {$2, $3, $4, $5}], [$1];";
      auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);
      auto *descOpr = ptxBuilder.newOperand(ptrtoint(i64_ty, tmaDesc), "l");
      auto *srcOpr = ptxBuilder.newOperand(ptrtoint(i32_ty, src), "r");
      auto *c0Opr = ptxBuilder.newOperand(coords[0], "r");
      auto *c1Opr = ptxBuilder.newOperand(coords[1], "r");
      auto *c2Opr = ptxBuilder.newOperand(coords[2], "r");
      auto *c3Opr = ptxBuilder.newOperand(coords[3], "r");
      auto *predOpr = ptxBuilder.newOperand(pred, "b");
      ptxInstr({descOpr, srcOpr, c0Opr, c1Opr, c2Opr, c3Opr, predOpr},
               /*onlyAttachMLIRArgs=*/true);
    } else {
      assert(false && "invalid dim size");
    }

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy, /*hasSideEffect*/ true);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class LoadDSmemOpPattern : public mlir::RewritePattern {
public:
  LoadDSmemOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::LoadDSmemOp::getOperationName(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto loadDSmemOp = llvm::dyn_cast<mlir::triton::nvgpu::LoadDSmemOp>(op);
    if (!loadDSmemOp)
      return mlir::failure();
    auto loc = op->getLoc();
    auto addr = loadDSmemOp.getAddr();
    auto ctaId = loadDSmemOp.getCtaId();
    auto bitwidth = loadDSmemOp.getBitwidth();
    auto vec = loadDSmemOp.getVec();

    assert(
        (bitwidth == 8 || bitwidth == 16 || bitwidth == 32 || bitwidth == 64) &&
        "invalid bitwidth");
    assert((vec == 1 || vec == 2 || vec == 4) && "invalid vec size");
    PTXBuilder ptxBuilder;

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

    auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);
    std::string c = bitwidth == 16 ? "=h" : (bitwidth == 32 ? "=r" : "=l");
    SmallVector<PTXBuilder::Operand *> oprs;
    for (unsigned i = 0; i < vec; ++i) {
      auto *ret = ptxBuilder.newOperand(c);
      oprs.push_back(ret);
    }
    auto *addrOpr = ptxBuilder.newOperand(addr, "r");
    auto *ctaIdOpr = ptxBuilder.newOperand(ctaId, "r");
    oprs.push_back(addrOpr);
    oprs.push_back(ctaIdOpr);

    Type retTy = IntegerType::get(rewriter.getContext(), bitwidth);
    SmallVector<Type> retTys(vec, retTy);
    if (vec > 1)
      retTy = struct_ty(retTys);

    ptxInstr(oprs,
             /*onlyAttachMLIRArgs=*/true);

    auto res = ptxBuilder.launch(rewriter, loc, retTy);
    rewriter.replaceOp(op, {res});
    return mlir::success();
  }
};

class ClusterCTAIdOpPattern : public mlir::RewritePattern {
public:
  ClusterCTAIdOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::ClusterCTAIdOp::getOperationName(), 1,
            context) {}

  mlir::Value getSRegValue(std::string name, Location loc,
                           mlir::PatternRewriter &rewriter) const {
    PTXBuilder ptxBuilder;
    std::string ptxAsm = "mov.u32 $0, " + name + ";";
    auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);
    auto *ret = ptxBuilder.newOperand("=r");
    ptxInstr({ret}, /*onlyAttachMLIRArgs=*/true);
    auto res =
        ptxBuilder.launch(rewriter, loc, i32_ty, /*hasSideEffect*/ false);
    return res;
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto clusterCTAIdOp =
        llvm::dyn_cast<mlir::triton::nvgpu::ClusterCTAIdOp>(op);
    if (!clusterCTAIdOp)
      return mlir::failure();
    auto loc = op->getLoc();

    auto x = getSRegValue("%cluster_ctaid.x", loc, rewriter);
    auto y = getSRegValue("%cluster_ctaid.y", loc, rewriter);
    auto z = getSRegValue("%cluster_ctaid.z", loc, rewriter);
    auto nx = getSRegValue("%cluster_nctaid.x", loc, rewriter);
    auto ny = getSRegValue("%cluster_nctaid.y", loc, rewriter);
    auto res = add(x, mul(add(y, mul(z, ny)), nx));
    rewriter.replaceOp(op, {res});
    return mlir::success();
  }
};

class WGMMAOpPattern : public mlir::RewritePattern {
public:
  WGMMAOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::triton::nvgpu::WGMMAOp::getOperationName(),
                             1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    using namespace mlir::triton::nvgpu;
    auto ctx = rewriter.getContext();
    auto wgmmaOp = llvm::dyn_cast<mlir::triton::nvgpu::WGMMAOp>(op);
    if (!wgmmaOp)
      return mlir::failure();
    auto loc = op->getLoc();
    auto opA = wgmmaOp.getOpA();
    auto opB = wgmmaOp.getOpB();
    auto opC = wgmmaOp.getOpC();
    auto m = wgmmaOp.getM();
    auto n = wgmmaOp.getN();
    auto k = wgmmaOp.getK();
    auto eltTypeC = wgmmaOp.getEltTypeC();
    auto eltTypeA = wgmmaOp.getEltTypeA();
    auto eltTypeB = wgmmaOp.getEltTypeB();
    auto layoutA = wgmmaOp.getLayoutA();
    auto layoutB = wgmmaOp.getLayoutB();

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
    PTXBuilder ptxBuilder;
    SmallVector<PTXBuilder::Operand *> oprs;

    // Operands
    uint32_t asmOpIdx = 0;

    // Operand C
    uint32_t numCRegs = m * n / 128;
    assert(numCRegs == structTypeC.getBody().size());
    std::string args = "";

    args += "{";
    for (uint32_t i = 0; i < numCRegs; ++i) {
      args += "$" + std::to_string(asmOpIdx++) + (i == numCRegs - 1 ? "" : ",");
      // LLVM does not support `+` semantic, we must repeat the arguments for
      // both input and outputs
      PTXBuilder::Operand *opr;
      if (structTypeC.getBody().front().isF32())
        opr = ptxBuilder.newOperand(
            extract_val(structTypeC.getBody()[i], opC, i), "=f");
      else
        opr = ptxBuilder.newOperand(
            extract_val(structTypeC.getBody()[i], opC, i), "=r");
      oprs.push_back(opr);
    }
    args += "}, ";

    for (uint32_t i = asmOpIdx - numCRegs; i < asmOpIdx; ++i) {
      auto *opr = ptxBuilder.newOperand(std::to_string(i));
      oprs.push_back(opr);
    }

    // Note that LLVM will not skip the indexed repeating placeholders
    asmOpIdx += numCRegs;
    // Operand A
    if (structTypeA) {
      uint32_t numARegs = m * k / 128;
      assert(numARegs == structTypeA.getBody().size());
      args += "{";
      for (uint32_t i = 0; i < numARegs; ++i) {
        args +=
            "$" + std::to_string(asmOpIdx++) + (i == numARegs - 1 ? "" : ",");
        auto *opr = ptxBuilder.newOperand(
            extract_val(structTypeA.getBody()[i], opA, i), "f");
        oprs.push_back(opr);
      }
      args += "}, ";
    } else {
      args += "$" + std::to_string(asmOpIdx++) + ", ";
      auto *opr = ptxBuilder.newOperand(opA, "l");
      oprs.push_back(opr);
    }

    // Operand B (must be `desc`)
    args += "$" + std::to_string(asmOpIdx++) + ", ";
    auto *opr = ptxBuilder.newOperand(opB, "l");
    oprs.push_back(opr);

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

    auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);
    ptxInstr(oprs,
             /*onlyAttachMLIRArgs=*/true);

    auto res =
        ptxBuilder.launch(rewriter, loc, structTypeC, /*hasSideEffect*/ true);
    rewriter.replaceOp(op, {res});
    return mlir::success();
  }
};

class FenceMBarrierInitOpPattern : public mlir::RewritePattern {
public:
  FenceMBarrierInitOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::FenceMBarrierInitOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto fenceMBarrierInitOp =
        llvm::dyn_cast<mlir::triton::nvgpu::FenceMBarrierInitOp>(op);
    if (!fenceMBarrierInitOp)
      return mlir::failure();
    auto loc = op->getLoc();
    PTXBuilder ptxBuilder;

    auto &ptxInstr =
        *ptxBuilder.create<PTXInstr>("fence.mbarrier_init.release.cluster");

    ptxInstr();

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy, /*hasSideEffect*/ true);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};
class NamedBarrierArriveOpPattern : public mlir::RewritePattern {
public:
  NamedBarrierArriveOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::NamedBarrierArriveOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto namedBarrierArriveOp =
        llvm::dyn_cast<mlir::triton::nvgpu::NamedBarrierArriveOp>(op);
    if (!namedBarrierArriveOp)
      return mlir::failure();
    auto loc = op->getLoc();
    auto bar = namedBarrierArriveOp.getBar();
    auto numThreads = namedBarrierArriveOp.getNumThreads();
    PTXBuilder ptxBuilder;

    auto &ptxInstr = *ptxBuilder.create<PTXInstr>("bar.arrive $0, $1;");
    auto *barOpr = ptxBuilder.newOperand(bar, "r");
    auto *numThreadsOpr = ptxBuilder.newOperand(numThreads, "r");
    ptxInstr({barOpr, numThreadsOpr}, /*onlyAttachMLIRArgs=*/true);

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class NamedBarrierWaitOpPattern : public mlir::RewritePattern {
public:
  NamedBarrierWaitOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::NamedBarrierWaitOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto namedBarrierWaitOp =
        llvm::dyn_cast<mlir::triton::nvgpu::NamedBarrierWaitOp>(op);
    if (!namedBarrierWaitOp)
      return mlir::failure();
    auto loc = op->getLoc();
    auto bar = namedBarrierWaitOp.getBar();
    auto numThreads = namedBarrierWaitOp.getNumThreads();
    PTXBuilder ptxBuilder;

    auto &ptxInstr = *ptxBuilder.create<PTXInstr>("bar.sync $0, $1;");
    auto *barOpr = ptxBuilder.newOperand(bar, "r");
    auto *numThreadsOpr = ptxBuilder.newOperand(numThreads, "r");
    ptxInstr({barOpr, numThreadsOpr}, /*onlyAttachMLIRArgs=*/true);

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CGABarrierArriveOpPattern : public mlir::RewritePattern {
public:
  CGABarrierArriveOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::CGABarrierArriveOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto cgaBarrierArriveOp =
        llvm::dyn_cast<mlir::triton::nvgpu::CGABarrierArriveOp>(op);
    if (!cgaBarrierArriveOp)
      return mlir::failure();
    auto loc = op->getLoc();
    PTXBuilder ptxBuilder;

    auto &ptxInstr = *ptxBuilder.create<PTXInstr>("barrier.cluster.arrive");
    ptxInstr();

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CGABarrierWaitOpPattern : public mlir::RewritePattern {
public:
  CGABarrierWaitOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::CGABarrierWaitOp::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto cgaBarrierWaitOp =
        llvm::dyn_cast<mlir::triton::nvgpu::CGABarrierWaitOp>(op);
    if (!cgaBarrierWaitOp)
      return mlir::failure();
    auto loc = op->getLoc();
    PTXBuilder ptxBuilder;

    auto &ptxInstr = *ptxBuilder.create<PTXInstr>("barrier.cluster.wait");
    ptxInstr();

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class StoreDSmemOpPattern : public mlir::RewritePattern {
public:
  StoreDSmemOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::StoreDSmemOp::getOperationName(), 1, context) {
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto storeDSmemOp = llvm::dyn_cast<mlir::triton::nvgpu::StoreDSmemOp>(op);
    if (!storeDSmemOp)
      return mlir::failure();
    auto loc = op->getLoc();
    auto addr = storeDSmemOp.getAddr();
    auto ctaId = storeDSmemOp.getCtaId();
    auto values = storeDSmemOp.getValues();
    auto pred = storeDSmemOp.getPred();

    auto bitwidth = storeDSmemOp.getBitwidth();
    auto vec = storeDSmemOp.getVec();
    assert(
        (bitwidth == 8 || bitwidth == 16 || bitwidth == 32 || bitwidth == 64) &&
        "invalid bitwidth");
    assert((vec == 1 || vec == 2 || vec == 4) && vec == values.size() &&
           "invalid vec size");

    PTXBuilder ptxBuilder;

    std::string ptxAsm = "{\n\t"
                         ".reg .u32 remoteAddr;\n\t"
                         "mapa.shared::cluster.u32 remoteAddr, $0, $1;\n\t"
                         ".reg .pred p;\n\t"
                         "mov.pred p, $2;\n\t"
                         "@p st.shared::cluster";
    if (vec > 1)
      ptxAsm += ".v" + std::to_string(vec);
    ptxAsm += ".u" + std::to_string(bitwidth) + " [remoteAddr], ";
    if (vec == 1)
      ptxAsm += "$3";
    else if (vec == 2)
      ptxAsm += "{$3, $4}";
    else if (vec == 4)
      ptxAsm += "{$3, $4, $5, $6}";
    ptxAsm += ";\n\t";
    ptxAsm += "}\n";
    auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);

    std::string c = bitwidth == 16 ? "h" : (bitwidth == 32 ? "r" : "l");
    SmallVector<PTXBuilder::Operand *> oprs;
    auto *addrOpr = ptxBuilder.newOperand(addr, "r");
    oprs.push_back(addrOpr);
    auto *ctaIdOpr = ptxBuilder.newOperand(ctaId, "r");
    oprs.push_back(ctaIdOpr);
    auto *predOpr = ptxBuilder.newOperand(pred, "b");
    oprs.push_back(predOpr);
    for (unsigned i = 0; i < values.size(); i++) {
      auto *valueOpr = ptxBuilder.newOperand(values[i], c);
      oprs.push_back(valueOpr);
    }
    ptxInstr(oprs,
             /*onlyAttachMLIRArgs=*/true);

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy, /*hasSideEffect*/ true);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class OffsetOfSts64OpPattern : public mlir::RewritePattern {
public:
  OffsetOfSts64OpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::OffsetOfSts64Op::getOperationName(), 1,
            context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto offsetOfSts64Op =
        llvm::dyn_cast<mlir::triton::nvgpu::OffsetOfSts64Op>(op);
    if (!offsetOfSts64Op)
      return mlir::failure();
    auto loc = op->getLoc();
    auto threadId = offsetOfSts64Op.getThreadId();
    auto rowOfWarp = offsetOfSts64Op.getRowOfWarp();
    auto elemIdx = offsetOfSts64Op.getElemIdx();
    auto leadingDimOffset = offsetOfSts64Op.getLeadingDimOffset();
    auto rowStride = offsetOfSts64Op.getRowStride();
    auto swizzleEnabled = offsetOfSts64Op.getSwizzleEnabled();

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

    auto laneId = and_(threadId, i32_val(0x1f));
    auto myRow =
        add(mul(and_(lshr(elemIdx, i32_val(1)), i32_val(0x1)), i32_val(8)),
            udiv(laneId, i32_val(4)));
    auto myCol = add(mul(udiv(elemIdx, i32_val(4)), i32_val(8)),
                     mul(urem(laneId, i32_val(4)), i32_val(2)));
    myRow = add(myRow, rowOfWarp);
    auto phase = urem(udiv(myRow, i32_val(perPhase)), i32_val(maxPhase));
    auto lineOffset =
        add(mul(urem(myRow, i32_val(perPhase)), i32_val(rowStride)),
            mul(myCol, i32_val(4)));
    auto colOffset =
        add(mul(xor_(udiv(lineOffset, i32_val(16)), phase), i32_val(16)),
            urem(lineOffset, i32_val(16)));
    auto offset =
        add(mul(udiv(myRow, i32_val(perPhase)), i32_val(128)), colOffset);

    rewriter.replaceOp(op, {offset});
    return mlir::success();
  }
};

class Sts64OpPattern : public mlir::RewritePattern {
public:
  Sts64OpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::triton::nvgpu::Sts64Op::getOperationName(),
                             1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto sts64Op = llvm::dyn_cast<mlir::triton::nvgpu::Sts64Op>(op);
    if (!sts64Op)
      return mlir::failure();
    auto loc = op->getLoc();
    auto offset = sts64Op.getOffset();
    auto d0 = sts64Op.getD0();
    auto d1 = sts64Op.getD1();

    PTXBuilder ptxBuilder;

    std::string ptxAsm = "st.shared.v2.b32 [$0], {$1, $2}";
    auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);

    SmallVector<PTXBuilder::Operand *> oprs;
    auto *addrOpr = ptxBuilder.newOperand(offset, "r");
    auto *d0Opr = ptxBuilder.newOperand(d0, "r");
    auto *d1Opr = ptxBuilder.newOperand(d1, "r");

    ptxInstr({addrOpr, d0Opr, d1Opr},
             /*onlyAttachMLIRArgs=*/true);

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};
class RegAllocOpPattern : public mlir::RewritePattern {
public:
  RegAllocOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::RegAllocOp::getOperationName(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto regAllocOp = llvm::dyn_cast<mlir::triton::nvgpu::RegAllocOp>(op);
    if (!regAllocOp)
      return mlir::failure();
    auto loc = op->getLoc();
    auto regCount = regAllocOp.getRegCount();

    PTXBuilder ptxBuilder;

    auto ptxAsm =
        "setmaxnreg.inc.sync.aligned.u32 " + std::to_string(regCount) + ";";
    auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);

    ptxInstr({},
             /*onlyAttachMLIRArgs=*/true);

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy, /*hasSideEffect*/ true);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class RegDeallocOpPattern : public mlir::RewritePattern {
public:
  RegDeallocOpPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(
            mlir::triton::nvgpu::RegDeallocOp::getOperationName(), 1, context) {
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto regDeallocOp = llvm::dyn_cast<mlir::triton::nvgpu::RegDeallocOp>(op);
    if (!regDeallocOp)
      return mlir::failure();
    auto loc = op->getLoc();
    auto regCount = regDeallocOp.getRegCount();

    PTXBuilder ptxBuilder;

    auto ptxAsm =
        "setmaxnreg.dec.sync.aligned.u32 " + std::to_string(regCount) + ";";
    auto &ptxInstr = *ptxBuilder.create<PTXInstr>(ptxAsm);

    ptxInstr({},
             /*onlyAttachMLIRArgs=*/true);

    auto asmReturnTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, asmReturnTy, /*hasSideEffect*/ true);
    rewriter.eraseOp(op);
    return mlir::success();
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
    patterns.add<CvtPackOpPattern>(context);
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
    // patterns.add<StoreDSmemOpPattern>(context);
    // patterns.add<Sts64OpPattern>(context);
    // patterns.add<OffsetOfSts64OpPattern>(context);
    // patterns.add<CGABarrierWaitOpPattern>(context);
    // patterns.add<CGABarrierArriveOpPattern>(context);
    // patterns.add<NamedBarrierWaitOpPattern>(context);
    // patterns.add<NamedBarrierArriveOpPattern>(context);
    // patterns.add<FenceMBarrierInitOpPattern>(context);

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
