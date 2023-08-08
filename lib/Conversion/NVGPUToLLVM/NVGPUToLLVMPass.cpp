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
