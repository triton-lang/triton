#include "TritonAMDGPUToLLVM/Passes.h"

#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTBUILTINFUNCTOLLVM
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

class CallOpConversion : public OpRewritePattern<LLVM::CallOp> {
public:
  CallOpConversion(mlir::MLIRContext *context, bool ftz)
      : OpRewritePattern<LLVM::CallOp>(context, 1), ftz(ftz) {}

  LogicalResult
  matchAndRewrite(LLVM::CallOp callOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (isPredicatedLoad(callOp)) {
      return convertPredicatedLoad(callOp, rewriter);
    } else if (isPredicatedStore(callOp)) {
      return convertPredicatedStore(callOp, rewriter);
    } else if (isWrappedLLVMIntrinsic(callOp)) {
      return convertToLLVMIntrinsic(callOp, rewriter);
    } else {
      return failure();
    }
  }

private:
  bool isPredicatedLoad(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().contains(mlir::LLVM::AMD::predicatedLoad);
  }

  bool isPredicatedStore(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().contains(
        mlir::LLVM::AMD::predicatedStore);
  }

  bool isWrappedLLVMIntrinsic(LLVM::CallOp callOp) const {
    if (std::optional<StringRef> callee = callOp.getCallee()) {
      if (callee.value().starts_with("__triton_hip_")) {
        return true;
      }
    }
    return false;
  }

  LogicalResult convertPredicatedStore(LLVM::CallOp callOp,
                                       mlir::PatternRewriter &rewriter) const {
    auto operands = callOp.getOperands();

    auto loc = callOp.getLoc();
    auto ptr = operands[0];
    auto val = operands[1];
    auto pred = operands[2];

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterStore =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterStore);
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, pred, trueBlock, afterStore);
    rewriter.setInsertionPointToStart(trueBlock);
    //               | vialatile | non-tmp | gcn instr gfx94
    // LLVM::StoreOp | 0         | 0       | (cg) global store
    //               | 0         | 1       | (cs) global store nt
    //               | 1         | 0/1     | (wt) global store sc0 sc1
    auto [volatileFlag, nonTmpFlag] =
        mlir::LLVM::AMD::getCacheModifierFlagsForPredicatedCall(callOp);
    auto storeOp = rewriter.create<LLVM::StoreOp>(
        loc, val, ptr, /*alignment=*/0, volatileFlag, nonTmpFlag);
    rewriter.create<LLVM::BrOp>(loc, afterStore);
    rewriter.setInsertionPointToStart(afterStore);
    rewriter.eraseOp(callOp);
    return mlir::success();
  }

  LogicalResult convertPredicatedLoad(LLVM::CallOp callOp,
                                      mlir::PatternRewriter &rewriter) const {
    auto operands = callOp.getOperands();
    auto result = callOp.getResult();

    auto loc = callOp.getLoc();
    auto elemTy = result.getType();
    auto ptr = operands[0];
    auto pred = operands[1];
    auto falseVal = operands[2];

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterLoad =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    afterLoad->addArgument({elemTy}, {loc});
    Block *trueBlock = rewriter.createBlock(afterLoad);
    Block *falseBlock =
        rewriter.splitBlock(trueBlock, rewriter.getInsertionPoint());
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, pred, trueBlock, falseBlock);
    rewriter.setInsertionPointToStart(trueBlock);
    //              | vialatile | non-tmp | gcn instr gfx94
    // LLVM::LoadOp | 0         | 0       | (ca) global load
    //              | 0/1       | 1       | (cg) global load nt
    //              | 1         | 0       | (cv) flat load sc0 sc1
    auto [volatileFlag, nonTmpFlag] =
        mlir::LLVM::AMD::getCacheModifierFlagsForPredicatedCall(callOp);
    auto loadOp = rewriter.create<LLVM::LoadOp>(
        loc, elemTy, ptr, /*alignment=*/0, volatileFlag, nonTmpFlag);
    rewriter.create<LLVM::BrOp>(loc, loadOp->getResult(0), afterLoad);
    rewriter.setInsertionPointToStart(falseBlock);
    rewriter.create<LLVM::BrOp>(loc, falseVal, afterLoad);
    rewriter.setInsertionPointToStart(afterLoad);
    Value loadVal = afterLoad->getArgument(0);
    rewriter.replaceOp(callOp, loadVal);
    return mlir::success();
  }

  LogicalResult convertToLLVMIntrinsic(LLVM::CallOp callOp,
                                       mlir::PatternRewriter &rewriter) const {
    StringRef calleeName = callOp.getCallee().value();

    auto operands = callOp.getOperands();
    auto result = callOp.getResult();

    LLVM::LLVMFunctionType calleeType = callOp.getCalleeFunctionType();
    Type returnType = calleeType.getReturnType();

    auto loc = callOp.getLoc();

    // clang-format off
    // define internal noundef zeroext i1 @__triton_hip_atom_cas_acqrel_relaxed_system(int*, int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef captures(none) %compare, ptr noundef readonly captures(none) %value) #0 !dbg !48 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !49
    //   %1 = load i32, ptr %compare, align 4, !dbg !50
    //   %2 = cmpxchg ptr %atomic_address, i32 %1, i32 %0 <SYNCGROUP> <SUCCESS_ORDERING> <FAILURE_ORDERING>, align 4, !dbg !50
    //   %3 = extractvalue { i32, i1 } %2, 1, !dbg !50
    //   br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !50
    //
    // cmpxchg.store_expected:
    //   %4 = extractvalue { i32, i1 } %2, 0
    //   store i32 %4, ptr %compare, align 4, !dbg !50
    //   br label %cmpxchg.continue, !dbg !50
    //
    // cmpxchg.continue:
    //   ret i1 %3, !dbg !51
    // }
    // clang-format on
    auto buildAtomicCAS = [&rewriter, &loc](

                              Value atomicAddress, Value compare, Value value,
                              LLVM::AtomicOrdering successOrdering,
                              LLVM::AtomicOrdering failureOrdering,
                              std::optional<StringRef> syncGroup =
                                  std::nullopt) {
      auto val = rewriter.create<LLVM::LoadOp>(
          loc, i32_ty, value, /*alignment=*/4,
          /*isVolatile =*/false, /*isNonTemporal*/ false,
          /*isInvariant =*/false, /*isInvariantGroup=*/false);
      auto cmp = rewriter.create<LLVM::LoadOp>(
          loc, i32_ty, compare, /*alignment=*/4,
          /*isVolatile =*/false, /*isNonTemporal*/ false,
          /*isInvariant =*/false, /*isInvariantGroup=*/false);
      auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
          loc, LLVM::AtomicBinOp::add, atomicAddress, cmp, val, successOrdering,
          failureOrdering, syncGroup.value_or(nullptr),
          /*alignment*/ 4);
      SmallVector<int64_t> position{1};
      auto extractOne =
          rewriter.create<LLVM::ExtractValueOp>(loc, cmpxchg, position);
      auto ifOp = rewriter.create<scf::IfOp>(
          loc, TypeRange{rewriter.getI1Type()},
          /*cond*/ extractOne, /*withElseRegion*/ true);
      {
        OpBuilder::InsertionGuard g(*rewriter);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
        SmallVector<int64_t> position{0};
        auto extractZero =
            rewriter.create<LLVM::ExtractValueOp>(loc, cmpxchg, position);
        (void)rewriter.create<LLVM::StoreOp>(loc, extractZero, compare,
                                             /*alignment=*/4);
        (void)rewriter.create<scf::YieldOp>(loc, extractOne);
      }
      {
        OpBuilder::InsertionGuard g(*rewriter);
        rewriter.setInsertionPointToStart(ifOp.elseBlock());
        (void)rewriter.create<scf::YieldOp>(loc, extractOne);
      }
      return ifOp;
    };

    // clang-format off
    // define internal noundef i64 @__triton_hip_load_acquire_agent(unsigned long*)(ptr noundef readonly captures(none) %input) #0 !dbg !76 {
    // entry:
    //   %0 = load atomic i64, ptr %input <SYNCGROUP> <ORDERING>, align 8, !dbg !77
    //   ret i64 %0, !dbg !78
    // }
    // clang-format on
    auto buildAtomicLoad =
        [&rewriter, &loc](Value ptr, LLVM::AtomicOrdering ordering,
                          std::optional<StringRef> syncGroup = std::nullopt) {
          return rewriter.create<LLVM::LoadOp>(
              loc, i64_ty, ptr, /*alignment=*/8,
              /*isVolatile =*/false, /*isNonTemporal*/ false,
              /*isInvariant =*/false, /*isInvariantGroup=*/false, ordering,
              syncGroup.value_or(nullptr));
        };

    // clang-format off
    // define protected amdgpu_kernel void @__triton_hip_store_relaxed_agent(unsigned long*)(ptr addrspace(1) noundef writeonly captures(none) %input.coerce) local_unnamed_addr #1 !dbg !94 {
    // entry:
    //   store atomic i64 <VAL>, ptr addrspace(1) <PTR> <SYNCGROUP> <ORDERING>, align 8, !dbg !95
    //   ret void, !dbg !96
    // }
    // clang-format on
    auto buildAtomicStore =
        [&rewriter, &loc](Value val, Value ptr, LLVM::AtomicOrdering ordering,
                          std::optional<StringRef> syncGroup = std::nullopt) {
          return rewriter.create<LLVM::StoreOp>(
              loc, val, ptr, /*alignment=*/8,
              /*isVolatile =*/false, /*isNonTemporal*/ false,
              /*isInvariant =*/false, /*isInvariantGroup=*/false,
              LLVM::AtomicOrdering::monotonic, "agent");
        };

    // clang-format off
    // define internal noundef i32 @__triton_hip_red_add_release_agent(int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef readonly captures(none) %value) #0 !dbg !124 {
    // entry:
    //   %0 = load i32, ptr <VALUE>, align 4, !dbg !125
    //   %1 = atomicrmw add ptr <ATOMIC_ADDRESS>, i32 %0 <SYNCGROUP> <ORDERING>, align 4, !dbg !126
    //   ret i32 %1, !dbg !127
    // }
    // clang-format on
    auto buildAtomicRMW = [&rewriter, &loc](Value atomicAddress, Value value,
                                            LLVM::AtomicOrdering ordering,
                                            std::optional<StringRef> syncGroup =
                                                std::nullopt) {
      auto loadOp = rewriter.create<LLVM::LoadOp>(
          loc, i32_ty, value, /*alignment=*/4,
          /*isVolatile =*/false, /*isNonTemporal*/ false,
          /*isInvariant =*/false, /*isInvariantGroup=*/false);
      return rewriter.create<LLVM::AtomicRMWOp>(
          loc, LLVM::AtomicBinOp::add, atomicAddress, loadOp, ordering,
          syncGroup.value_or(nullptr), /*alignment*/ 4);
    };

    Operation *replacementOp = nullptr;
    if (calleeName == "__triton_hip_iabs") {
      assert(operands.size() == 1);
      replacementOp = rewriter.create<LLVM::AbsOp>(loc, returnType, operands[0],
                                                   /*is_int_min_poison=*/false);
    } else if (calleeName == "__triton_hip_fabs") {
      assert(operands.size() == 1);
      replacementOp =
          rewriter.create<LLVM::FAbsOp>(loc, returnType, operands[0]);
    } else if (calleeName == "__triton_hip_llrint") {
      assert(operands.size() == 1);
      // Note, LrintOp and LlrintOp result in a code-gen error
      Operation *op = rewriter.create<LLVM::RintOp>(loc, operands[0].getType(),
                                                    operands[0]);
      replacementOp =
          rewriter.create<LLVM::FPToSIOp>(loc, returnType, op->getResult(0));
    } else if (calleeName == "__triton_hip_fast_fdividef") {
      assert(operands.size() == 2);
      const char *intrinsic = "llvm.amdgcn.rcp.f32";
      auto rcpOp = LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic,
                                                   returnType, operands[1]);

      LLVM::FastmathFlagsAttr defaultFlags{};
      replacementOp = rewriter.create<LLVM::FMulOp>(
          loc, returnType, operands[0], rcpOp->getResult(0), defaultFlags);
    } else if (calleeName == "__triton_hip_fast_expf") {
      assert(operands.size() == 1);
      assert(operands[0].getType().getIntOrFloatBitWidth() == 32);
      const double log2e = 1.4426950408889634;
      LLVM::FastmathFlagsAttr defaultFlags{};
      auto mulOp = rewriter.create<LLVM::FMulOp>(
          loc, rewriter.getF32Type(), operands[0],
          LLVM::createConstantF32(loc, rewriter, log2e), defaultFlags);
      const char *intrinsic = ftz ? "llvm.amdgcn.exp2.f32" : "llvm.exp2.f32";

      replacementOp = LLVM::createLLVMIntrinsicCallOp(
          rewriter, loc, intrinsic, returnType, mulOp->getResult(0));
    }

    // clang-format off
    // define internal noundef i32 @__triton_hip_atom_add_acqrel_agent(int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef readonly captures(none) %value) #0 !dbg !12 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !16
    //   %1 = atomicrmw add ptr %atomic_address, i32 %0 syncscope("agent") acq_rel, align 4, !dbg !21
    //   ret i32 %1, !dbg !23
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_atom_add_acqrel_agent") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicRMW(operands[1], operands[0],
                                     LLVM::AtomicOrdering::acq_rel, "agent");
    }
    // clang-format off
    // define internal noundef i32 @__triton_hip_atom_add_acqrel_system(int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef readonly captures(none) %value) #0 !dbg !24 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !25
    //   %1 = atomicrmw add ptr %atomic_address, i32 %0 acq_rel, align 4, !dbg !26
    //   ret i32 %1, !dbg !27
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_atom_add_acqrel_system") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicRMW(operands[1], operands[0],
                                     LLVM::AtomicOrdering::acq_rel, "system");
    }
    // clang-format off
    // define internal noundef i32 @__triton_hip_atom_add_acquire_agent(int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef readonly captures(none) %value) #0 !dbg !28 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !29
    //   %1 = atomicrmw add ptr %atomic_address, i32 %0 syncscope("agent") acquire, align 4, !dbg !30
    //   ret i32 %1, !dbg !31
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_atom_add_acquire_agent") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicRMW(operands[1], operands[0],
                                     LLVM::AtomicOrdering::acquire, "agent");
    }
    // clang-format off
    // define internal noundef i32 @__triton_hip_atom_add_acquire_system(int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef readonly captures(none) %value) #0 !dbg !32 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !33
    //   %1 = atomicrmw add ptr %atomic_address, i32 %0 acquire, align 4, !dbg !34
    //   ret i32 %1, !dbg !35
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_atom_add_acquire_system") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicRMW(operands[1], operands[0],
                                     LLVM::AtomicOrdering::acquire, "system");
    }
    // clang-format off
    // define internal noundef i32 @__triton_hip_atom_add_relaxed_agent(int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef readonly captures(none) %value) #0 !dbg !36 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !37
    //   %1 = atomicrmw add ptr %atomic_address, i32 %0 syncscope("agent") monotonic, align 4, !dbg !38
    //   ret i32 %1, !dbg !39
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_atom_add_relaxed_agent") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicRMW(operands[1], operands[0],
                                     LLVM::AtomicOrdering::monotonic, "agent");
    }
    // clang-format off
    // define internal noundef i32 @__triton_hip_atom_add_relaxed_system(int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef readonly captures(none) %value) #0 !dbg !40 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !41
    //   %1 = atomicrmw add ptr %atomic_address, i32 %0 monotonic, align 4, !dbg !42
    //   ret i32 %1, !dbg !43
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_atom_add_relaxed_system") {
      assert(operands.size() == 2);
      auto loadOp = rewriter.create<LLVM::LoadOp>(
          loc, i32_ty, operands[1], /*alignment=*/4,
          /*isVolatile =*/false, /*isNonTemporal*/ false,
          /*isInvariant =*/false, /*isInvariantGroup=*/false);
      replacementOp = rewriter.create<LLVM::AtomicRMWOp>(
          loc, LLVM::AtomicBinOp::add, operands[0], loadOp,
          LLVM::AtomicOrdering::monotonic, "system", /*alignment*/ 4);
    }

    // TODO(max): zeroext on fn
    // clang-format off
    // define internal noundef zeroext i1 @__triton_hip_atom_cas_acqrel_relaxed_agent(int*, int*, int*)(
    //  ptr noundef captures(none) %atomic_address,
    //  ptr noundef captures(none) %compare,
    //  ptr noundef readonly captures(none) %value
    // ) #0 !dbg !44 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !45
    //   %1 = load i32, ptr %compare, align 4, !dbg !46
    //   %2 = cmpxchg ptr %atomic_address, i32 %1, i32 %0 syncscope("agent") acq_rel monotonic, align 4, !dbg !46
    //   %3 = extractvalue { i32, i1 } %2, 1, !dbg !46
    //   br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !46
    //
    // cmpxchg.store_expected:
    //   %4 = extractvalue { i32, i1 } %2, 0
    //   store i32 %4, ptr %compare, align 4, !dbg !46
    //   br label %cmpxchg.continue, !dbg !46
    //
    // cmpxchg.continue:
    //   ret i1 %3, !dbg !47
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_atom_cas_acqrel_relaxed_agent") {
      assert(operands.size() == 3);
      auto atomicAddress = operands[0], compare = operands[1],
           value = operands[2];
      replacementOp = buildAtomicCAS(atomicAddress, compare, value,
                                     LLVM::AtomicOrdering::acq_rel,
                                     LLVM::AtomicOrdering::monotonic, "agent");
    }

    // clang-format off
    // define internal noundef zeroext i1 @__triton_hip_atom_cas_acqrel_relaxed_system(int*, int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef captures(none) %compare, ptr noundef readonly captures(none) %value) #0 !dbg !48 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !49
    //   %1 = load i32, ptr %compare, align 4, !dbg !50
    //   %2 = cmpxchg ptr %atomic_address, i32 %1, i32 %0 acq_rel monotonic, align 4, !dbg !50
    //   %3 = extractvalue { i32, i1 } %2, 1, !dbg !50
    //   br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !50
    //
    // cmpxchg.store_expected:
    //   %4 = extractvalue { i32, i1 } %2, 0
    //   store i32 %4, ptr %compare, align 4, !dbg !50
    //   br label %cmpxchg.continue, !dbg !50
    //
    // cmpxchg.continue:
    //   ret i1 %3, !dbg !51
    // }
    else if (calleeName == "__triton_hip_atom_cas_acqrel_relaxed_system") {
      assert(operands.size() == 3);
      auto atomicAddress = operands[0], compare = operands[1],
           value = operands[2];
      replacementOp = buildAtomicCAS(atomicAddress, compare, value,LLVM::AtomicOrdering::acq_rel,
                                     LLVM::AtomicOrdering::monotonic);
    }
    // define internal noundef zeroext i1 @__triton_hip_atom_cas_acquire_relaxed_agent(int*, int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef captures(none) %compare, ptr noundef readonly captures(none) %value) #0 !dbg !52 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !53
    //   %1 = load i32, ptr %compare, align 4, !dbg !54
    //   %2 = cmpxchg ptr %atomic_address, i32 %1, i32 %0 syncscope("agent") acquire monotonic, align 4, !dbg !54
    //   %3 = extractvalue { i32, i1 } %2, 1, !dbg !54
    //   br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !54
    //
    // cmpxchg.store_expected:
    //   %4 = extractvalue { i32, i1 } %2, 0
    //   store i32 %4, ptr %compare, align 4, !dbg !54
    //   br label %cmpxchg.continue, !dbg !54
    //
    // cmpxchg.continue:
    //   ret i1 %3, !dbg !55
    // }
    else if (calleeName == "__triton_hip_atom_cas_acquire_relaxed_agent") {
      assert(operands.size() == 3);
      auto atomicAddress = operands[0], compare = operands[1],
           value = operands[2];
      replacementOp = buildAtomicCAS(atomicAddress, compare, value,LLVM::AtomicOrdering::acquire,
                                     LLVM::AtomicOrdering::monotonic, "agent");
    }
    // define internal noundef zeroext i1 @__triton_hip_atom_cas_acquire_relaxed_system(int*, int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef captures(none) %compare, ptr noundef readonly captures(none) %value) #0 !dbg !56 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !57
    //   %1 = load i32, ptr %compare, align 4, !dbg !58
    //   %2 = cmpxchg ptr %atomic_address, i32 %1, i32 %0 acquire monotonic, align 4, !dbg !58
    //   %3 = extractvalue { i32, i1 } %2, 1, !dbg !58
    //   br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !58
    //
    // cmpxchg.store_expected:
    //   %4 = extractvalue { i32, i1 } %2, 0
    //   store i32 %4, ptr %compare, align 4, !dbg !58
    //   br label %cmpxchg.continue, !dbg !58
    //
    // cmpxchg.continue:
    //   ret i1 %3, !dbg !59
    // }
    else if (calleeName == "__triton_hip_atom_cas_acquire_relaxed_system") {
      assert(operands.size() == 3);
      auto atomicAddress = operands[0], compare = operands[1],
           value = operands[2];
      replacementOp = buildAtomicCAS(atomicAddress, compare, value,LLVM::AtomicOrdering::acquire,
                                     LLVM::AtomicOrdering::monotonic);
    }
    // define internal noundef zeroext i1 @__triton_hip_atom_cas_relaxed_relaxed_agent(int*, int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef captures(none) %compare, ptr noundef readonly captures(none) %value) #0 !dbg !60 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !61
    //   %1 = load i32, ptr %compare, align 4, !dbg !62
    //   %2 = cmpxchg ptr %atomic_address, i32 %1, i32 %0 syncscope("agent") monotonic monotonic, align 4, !dbg !62
    //   %3 = extractvalue { i32, i1 } %2, 1, !dbg !62
    //   br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !62
    //
    // cmpxchg.store_expected:
    //   %4 = extractvalue { i32, i1 } %2, 0
    //   store i32 %4, ptr %compare, align 4, !dbg !62
    //   br label %cmpxchg.continue, !dbg !62
    //
    // cmpxchg.continue:
    //   ret i1 %3, !dbg !63
    // }
    else if (calleeName == "__triton_hip_atom_cas_relaxed_relaxed_agent") {
      assert(operands.size() == 3);
      auto atomicAddress = operands[0], compare = operands[1],
           value = operands[2];
      replacementOp = buildAtomicCAS(atomicAddress, compare, value,LLVM::AtomicOrdering::monotonic,
                                     LLVM::AtomicOrdering::monotonic, "agent");
    }
    // define internal noundef zeroext i1 @__triton_hip_atom_cas_relaxed_relaxed_system(int*, int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef captures(none) %compare, ptr noundef readonly captures(none) %value) #0 !dbg !64 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !65
    //   %1 = load i32, ptr %compare, align 4, !dbg !66
    //   %2 = cmpxchg ptr %atomic_address, i32 %1, i32 %0 monotonic monotonic, align 4, !dbg !66
    //   %3 = extractvalue { i32, i1 } %2, 1, !dbg !66
    //   br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !66
    //
    // cmpxchg.store_expected:
    //   %4 = extractvalue { i32, i1 } %2, 0
    //   store i32 %4, ptr %compare, align 4, !dbg !66
    //   br label %cmpxchg.continue, !dbg !66
    //
    // cmpxchg.continue:
    //   ret i1 %3, !dbg !67
    // }
    else if (calleeName == "__triton_hip_atom_cas_relaxed_relaxed_system") {
      assert(operands.size() == 3);
      auto atomicAddress = operands[0], compare = operands[1],
           value = operands[2];
      replacementOp = buildAtomicCAS(atomicAddress, compare, value,LLVM::AtomicOrdering::monotonic,
                                     LLVM::AtomicOrdering::monotonic);
    }
    // define internal noundef zeroext i1 @__triton_hip_atom_cas_release_relaxed_agent(int*, int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef captures(none) %compare, ptr noundef readonly captures(none) %value) #0 !dbg !68 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !69
    //   %1 = load i32, ptr %compare, align 4, !dbg !70
    //   %2 = cmpxchg ptr %atomic_address, i32 %1, i32 %0 syncscope("agent") release monotonic, align 4, !dbg !70
    //   %3 = extractvalue { i32, i1 } %2, 1, !dbg !70
    //   br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !70
    //
    // cmpxchg.store_expected:
    //   %4 = extractvalue { i32, i1 } %2, 0
    //   store i32 %4, ptr %compare, align 4, !dbg !70
    //   br label %cmpxchg.continue, !dbg !70
    //
    // cmpxchg.continue:
    //   ret i1 %3, !dbg !71
    // }
    else if (calleeName == "__triton_hip_atom_cas_release_relaxed_agent") {
      assert(operands.size() == 3);
      auto atomicAddress = operands[0], compare = operands[1],
           value = operands[2];
      replacementOp = buildAtomicCAS(atomicAddress, compare, value,LLVM::AtomicOrdering::release,
                                     LLVM::AtomicOrdering::monotonic, "agent");
    }
    // define internal noundef zeroext i1 @__triton_hip_atom_cas_release_relaxed_system(int*, int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef captures(none) %compare, ptr noundef readonly captures(none) %value) #0 !dbg !72 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !73
    //   %1 = load i32, ptr %compare, align 4, !dbg !74
    //   %2 = cmpxchg ptr %atomic_address, i32 %1, i32 %0 release monotonic, align 4, !dbg !74
    //   %3 = extractvalue { i32, i1 } %2, 1, !dbg !74
    //   br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !74
    //
    // cmpxchg.store_expected:
    //   %4 = extractvalue { i32, i1 } %2, 0
    //   store i32 %4, ptr %compare, align 4, !dbg !74
    //   br label %cmpxchg.continue, !dbg !74
    //
    // cmpxchg.continue:
    //   ret i1 %3, !dbg !75
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_atom_cas_release_relaxed_system") {
      assert(operands.size() == 3);
      auto atomicAddress = operands[0], compare = operands[1],
           value = operands[2];
      replacementOp = buildAtomicCAS(atomicAddress, compare, value,
                                     LLVM::AtomicOrdering::release,
                                     LLVM::AtomicOrdering::monotonic);
    }
    // clang-format off
    // define internal noundef i64 @__triton_hip_load_acquire_agent(unsigned long*)(ptr noundef readonly captures(none) %input) #0 !dbg !76 {
    // entry:
    //   %0 = load atomic i64, ptr %input syncscope("agent") acquire, align 8, !dbg !77
    //   ret i64 %0, !dbg !78
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_load_acquire_agent") {
      assert(operands.size() == 1);
      replacementOp =
          buildAtomicLoad(operands[0], LLVM::AtomicOrdering::acquire, "agent");
    }
    // clang-format off
    // define internal noundef i64 @__triton_hip_load_acquire_system(unsigned long*)(ptr noundef readonly captures(none) %input) #0 !dbg !79 {
    // entry:
    //   %0 = load atomic i64, ptr %input acquire, align 8, !dbg !80
    //   ret i64 %0, !dbg !81
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_load_acquire_system") {
      assert(operands.size() == 1);
      replacementOp =
          buildAtomicLoad(operands[0], LLVM::AtomicOrdering::acquire);
    }
    // clang-format off
    // define internal noundef i64 @__triton_hip_load_acquire_workgroup(unsigned long*)(ptr noundef readonly captures(none) %input) #0 !dbg !82 {
    // entry:
    //   %0 = load atomic i64, ptr %input syncscope("workgroup") acquire, align 8, !dbg !83
    //   ret i64 %0, !dbg !84
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_load_acquire_workgroup") {
      assert(operands.size() == 1);
      replacementOp = buildAtomicLoad(
          operands[0], LLVM::AtomicOrdering::acquire, "workgroup");
    }
    // clang-format off
    // define internal noundef i64 @__triton_hip_load_relaxed_agent(unsigned long*)(ptr noundef readonly captures(none) %input) #0 !dbg !85 {
    // entry:
    //   %0 = load atomic i64, ptr %input syncscope("agent") monotonic, align 8, !dbg !86
    //   ret i64 %0, !dbg !87
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_load_relaxed_agent") {
      assert(operands.size() == 1);
      replacementOp = buildAtomicLoad(operands[0],
                                      LLVM::AtomicOrdering::monotonic, "agent");
    }
    // clang-format off
    // define internal noundef i64 @__triton_hip_load_relaxed_system(unsigned long*)(ptr noundef readonly captures(none) %input) #0 !dbg !88 {
    // entry:
    //   %0 = load atomic i64, ptr %input monotonic, align 8, !dbg !89
    //   ret i64 %0, !dbg !90
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_load_relaxed_system") {
      assert(operands.size() == 1);
      replacementOp =
          buildAtomicLoad(operands[0], LLVM::AtomicOrdering::monotonic);
    }
    // clang-format off
    // define internal noundef i64 @__triton_hip_load_relaxed_workgroup(unsigned long*)(ptr noundef readonly captures(none) %input) #0 !dbg !91 {
    // entry:
    //   %0 = load atomic i64, ptr %input syncscope("workgroup") monotonic, align 8, !dbg !92
    //   ret i64 %0, !dbg !93
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_load_relaxed_workgroup") {
      assert(operands.size() == 1);
      replacementOp = buildAtomicLoad(
          operands[0], LLVM::AtomicOrdering::monotonic, "workgroup");
    }
    // clang-format off
    // define protected amdgpu_kernel void @__triton_hip_store_relaxed_agent(unsigned long*)(ptr addrspace(1) noundef writeonly captures(none) %input.coerce) local_unnamed_addr #1 !dbg !94 {
    // entry:
    //   store atomic i64 1, ptr addrspace(1) %input.coerce syncscope("agent") monotonic, align 8, !dbg !95
    //   ret void, !dbg !96
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_store_relaxed_agent") {
      assert(operands.size() == 1);
      auto one =
          createScalarOrSplatConstant(rewriter, loc, rewriter.getI64Type(), 1);
      replacementOp = buildAtomicStore(
          one, operands[0], LLVM::AtomicOrdering::monotonic, "agent");
    }
    // clang-format off
    // define protected amdgpu_kernel void @__triton_hip_store_relaxed_system(unsigned long*)(ptr addrspace(1) noundef writeonly captures(none) %input.coerce) local_unnamed_addr #1 !dbg !97 {
    // entry:
    //   store atomic i64 1, ptr addrspace(1) %input.coerce monotonic, align 8, !dbg !98
    //   ret void, !dbg !99
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_store_relaxed_system") {
      assert(operands.size() == 1);
      auto one =
          createScalarOrSplatConstant(rewriter, loc, rewriter.getI64Type(), 1);
      replacementOp =
          buildAtomicStore(one, operands[0], LLVM::AtomicOrdering::monotonic);
    }
    // clang-format off
    // define protected amdgpu_kernel void @__triton_hip_store_relaxed_workgroup(unsigned long*)(ptr addrspace(1) noundef writeonly captures(none) %input.coerce) local_unnamed_addr #1 !dbg !100 {
    // entry:
    //   store atomic i64 1, ptr addrspace(1) %input.coerce syncscope("workgroup") monotonic, align 8, !dbg !101
    //   ret void, !dbg !102
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_store_relaxed_workgroup") {
      assert(operands.size() == 1);
      auto one =
          createScalarOrSplatConstant(rewriter, loc, rewriter.getI64Type(), 1);
      replacementOp = buildAtomicStore(
          one, operands[0], LLVM::AtomicOrdering::monotonic, "workgroup");
    }
    // clang-format off
    // define protected amdgpu_kernel void @__triton_hip_store_release_agent(unsigned long*)(ptr addrspace(1) noundef writeonly captures(none) %input.coerce) local_unnamed_addr #1 !dbg !103 {
    // entry:
    //   store atomic i64 1, ptr addrspace(1) %input.coerce syncscope("agent") release, align 8, !dbg !104
    //   ret void, !dbg !105
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_store_release_agent") {
      assert(operands.size() == 1);
      auto one =
          createScalarOrSplatConstant(rewriter, loc, rewriter.getI64Type(), 1);
      replacementOp = buildAtomicStore(one, operands[0],
                                       LLVM::AtomicOrdering::release, "agent");
    }
    // clang-format off
    // define protected amdgpu_kernel void @__triton_hip_store_release_system(unsigned long*)(ptr addrspace(1) noundef writeonly captures(none) %input.coerce) local_unnamed_addr #1 !dbg !106 {
    // entry:
    //   store atomic i64 1, ptr addrspace(1) %input.coerce release, align 8, !dbg !107
    //   ret void, !dbg !108
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_store_release_system") {
      assert(operands.size() == 1);
      auto one =
          createScalarOrSplatConstant(rewriter, loc, rewriter.getI64Type(), 1);
      replacementOp =
          buildAtomicStore(one, operands[0], LLVM::AtomicOrdering::release);
    }
    // clang-format off
    // define protected amdgpu_kernel void @__triton_hip_store_release_workgroup(unsigned long*)(ptr addrspace(1) noundef writeonly captures(none) %input.coerce) local_unnamed_addr #1 !dbg !109 {
    // entry:
    //   store atomic i64 1, ptr addrspace(1) %input.coerce syncscope("workgroup") release, align 8, !dbg !110
    //   ret void, !dbg !111
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_store_release_workgroup") {
      assert(operands.size() == 1);
      auto one =
          createScalarOrSplatConstant(rewriter, loc, rewriter.getI64Type(), 1);
      replacementOp = buildAtomicStore(
          one, operands[0], LLVM::AtomicOrdering::release, "workgroup");
    }
    // clang-format off
    // define internal noundef i32 @__triton_hip_red_add_release_agent(int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef readonly captures(none) %value) #0 !dbg !124 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !125
    //   %1 = atomicrmw add ptr %atomic_address, i32 %0 syncscope("agent") release, align 4, !dbg !126
    //   ret i32 %1, !dbg !127
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_red_add_release_agent") {
      assert(operands.size() == 2);
      auto loadOp = rewriter.create<LLVM::LoadOp>(
          loc, i32_ty, operands[1], /*alignment=*/4,
          /*isVolatile =*/false, /*isNonTemporal*/ false,
          /*isInvariant =*/false, /*isInvariantGroup=*/false);
      replacementOp = rewriter.create<LLVM::AtomicRMWOp>(
          loc, LLVM::AtomicBinOp::add, operands[0], loadOp,
          LLVM::AtomicOrdering::release, "agent", /*alignment*/ 4);
    }
    // clang-format off
    // define internal noundef i32 @__triton_hip_red_add_release_system(int*, int*)(ptr noundef captures(none) %atomic_address, ptr noundef readonly captures(none) %value) #0 !dbg !128 {
    // entry:
    //   %0 = load i32, ptr %value, align 4, !dbg !129
    //   %1 = atomicrmw add ptr %atomic_address, i32 %0 release, align 4, !dbg !130
    //   ret i32 %1, !dbg !131
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_red_add_release_system") {
      assert(operands.size() == 2);
      auto loadOp = rewriter.create<LLVM::LoadOp>(
          loc, i32_ty, operands[1], /*alignment=*/4,
          /*isVolatile =*/false, /*isNonTemporal*/ false,
          /*isInvariant =*/false, /*isInvariantGroup=*/false);
      replacementOp = rewriter.create<LLVM::AtomicRMWOp>(
          loc, LLVM::AtomicBinOp::add, operands[0], loadOp,
          LLVM::AtomicOrdering::release, /*syncscope*/ nullptr,
          /*alignment*/ 4);
    }
    // clang-format off
    // define protected amdgpu_kernel void @__triton_hip_syncthreads()() local_unnamed_addr #2 !dbg !112 {
    // entry:
    //   fence syncscope("workgroup") release, !dbg !113
    //   tail call void @llvm.amdgcn.s.barrier(), !dbg !121
    //   fence syncscope("workgroup") acquire, !dbg !122
    //   ret void, !dbg !123
    // }
    // clang-format on
    else if (calleeName == "__triton_hip_syncthreads") {
      assert(operands.size() == 0);
      (void)rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::release,
                                           "workgroup");
      SmallVector<NamedAttribute> attrs{
          {"callee", FlatSymbolRefAttr::get(rewriter.getContext(),
                                            "llvm.amdgcn.s.barrier")},
          {"CConv", LLVM::CConvAttr::get(rewriter.getContext(),
                                         LLVM::cconv::CConv::Tail)},
      };
      (void)rewriter.create<LLVM::CallOp>(
          loc, TypeRange{LLVM::LLVMVoidType::get(rewriter.getContext())},
          ValueRange{}, attrs);
      replacementOp = rewriter.create<LLVM::FenceOp>(
          loc, LLVM::AtomicOrdering::acquire, "workgroup");
    }

    if (replacementOp) {
      rewriter.replaceOp(callOp, replacementOp);
      return mlir::success();
    }

    return mlir::failure();
  }

private:
  bool ftz;
};

struct ConvertBuiltinFuncToLLVM
    : public triton::impl::ConvertBuiltinFuncToLLVMBase<
          ConvertBuiltinFuncToLLVM> {
  explicit ConvertBuiltinFuncToLLVM(bool ftz) { this->ftz = ftz; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    GreedyRewriteConfig config;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Aggressive;

    RewritePatternSet patterns(context);
    patterns.add<CallOpConversion>(context, this->ftz);

    if (mlir::applyPatternsGreedily(mod, std::move(patterns), config)
            .failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertBuiltinFuncToLLVMPass(bool ftz) {
  return std::make_unique<ConvertBuiltinFuncToLLVM>(ftz);
}

} // namespace mlir::triton
