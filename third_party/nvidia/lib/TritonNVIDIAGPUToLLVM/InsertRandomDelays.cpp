#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir::triton {
#define GEN_PASS_DEF_INSERTRANDOMDELAYS
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// Insert random delays before these operations.
std::array<const char *, 19> before = {"cp.async.bulk.tensor",
                                       "cp.async.bulk.prefetch.tensor",
                                       "cp.async.bulk.commit_group",
                                       "cp.reduce.async.bulk.tensor",
                                       "fence.proxy",
                                       "wgmma.fence",
                                       "wgmma.commit_group",
                                       "bar.arrive",
                                       "barrier.arrive",
                                       "barrier.cluster.arrive",
                                       "mbarrier.init",
                                       "mbarrier.arrive",
                                       "mbarrier.inval",
                                       "mbarrier.expect_tx",
                                       "fence.mbarrier_init",
                                       "cp.async.mbarrier.arrive",
                                       "tcgen05.commit",
                                       "llvm.nvvm.barrier",
                                       "llvm.nvvm.cluster.barrier"};

// Insert random delays after these operations.
std::array<const char *, 9> after = {"cp.async.bulk.wait_group",
                                     "wgmma.wait_group",
                                     "cp.async.wait",
                                     "bar.sync",
                                     "barrier.sync",
                                     "barrier.cluster.wait",
                                     "mbarrier.test_wait",
                                     "mbarrier.try_wait",
                                     "tcgen05.wait"};

/* Helper to create murmurhash3 insert function.
The sequence of operations to insert into the hash given some part is:
  part *= 0xcc9e2d51;
  part = part << 15 | part >> 17;
  hash ^= part * 0x1b873593;
  hash = hash << 13 | hash >> 19;
  hash *= 5;
  hash += 0xe6546b64;
*/
static LLVM::LLVMFuncOp createHashInsertFn(IRRewriter &rewriter,
                                           ModuleOp module,
                                           FunctionOpInterface funcOp) {
  auto ctx = rewriter.getContext();
  auto i32Type = type::i32Ty(ctx);
  auto loc = funcOp.getLoc();
  StringRef murmurhash3InsertFnName = "murmurhash3_insert";
  TritonLLVMOpBuilder llvmBuilder(loc, rewriter);

  LLVM::LLVMFuncOp murmurhash3InsertFn =
      module.lookupSymbol<LLVM::LLVMFuncOp>(murmurhash3InsertFnName);
  if (!murmurhash3InsertFn) {
    rewriter.setInsertionPoint(funcOp);
    murmurhash3InsertFn = rewriter.create<LLVM::LLVMFuncOp>(
        module.getLoc(), murmurhash3InsertFnName,
        LLVM::LLVMFunctionType::get(i32Type, {i32Type, i32Type}));
    murmurhash3InsertFn.setPrivate();

    Block *funcBlock = murmurhash3InsertFn.addEntryBlock(rewriter);
    ValueRange args = funcBlock->getArguments();
    rewriter.setInsertionPointToEnd(funcBlock);

    auto part = args[1];

    // part *= 0xcc9e2d51
    auto constFactor = llvmBuilder.i32_val(0xcc9e2d51);
    auto partMulFactor = llvmBuilder.mul(part, constFactor);

    // part = part << 15 | part >> 17
    auto const15 = llvmBuilder.i32_val(15);
    auto partShiftLeft15 = llvmBuilder.shl(partMulFactor, const15);
    auto const17 = llvmBuilder.i32_val(17);
    auto partShiftRight17 = llvmBuilder.lshr(partMulFactor, const17);
    auto partOr = llvmBuilder.or_(partShiftLeft15, partShiftRight17);

    // hash ^= part * 0x1b873593
    auto hash = args[0];
    auto constFactor2 = llvmBuilder.i32_val(0x1b873593);
    auto partMulFactor2 = llvmBuilder.mul(partOr, constFactor2);
    auto hashXor = llvmBuilder.xor_(hash, partMulFactor2);

    // hash = hash << 13 | hash >> 19
    auto const13 = llvmBuilder.i32_val(13);
    auto hashShiftLeft13 = llvmBuilder.shl(hashXor, const13);
    auto const19 = llvmBuilder.i32_val(19);
    auto hashShiftRight19 = llvmBuilder.lshr(hashXor, const19);
    auto hashOr = llvmBuilder.or_(hashShiftLeft13, hashShiftRight19);

    // hash *= 5
    auto const5 = llvmBuilder.i32_val(5);
    auto hashMulFactor = llvmBuilder.mul(hashOr, const5);

    // hash += 0xe6546b64
    auto constAdd = llvmBuilder.i32_val(0xe6546b64);
    auto hashAddFactor = llvmBuilder.add(hashMulFactor, constAdd);

    // return hash;
    rewriter.create<LLVM::ReturnOp>(loc, hashAddFactor);
  }
  return murmurhash3InsertFn;
}

/* Helper to create murmurhash3 finish function.
The sequence of operations to finish the hash is:
  hash ^= hash >> 16;
  hash *= 0x85ebca6b;
  hash ^= hash >> 13;
  hash *= 0xc2b2ae35;
  hash ^= hash >> 16;
*/
static LLVM::LLVMFuncOp createHashFinishFn(IRRewriter &rewriter,
                                           ModuleOp module,
                                           FunctionOpInterface funcOp) {
  auto ctx = rewriter.getContext();
  auto i32Type = type::i32Ty(ctx);
  StringRef murmurhash3FinishFnName = "murmurhash3_finish";
  auto loc = funcOp.getLoc();
  TritonLLVMOpBuilder llvmBuilder(loc, rewriter);

  LLVM::LLVMFuncOp murmurhash3FinishFn =
      module.lookupSymbol<LLVM::LLVMFuncOp>(murmurhash3FinishFnName);
  if (!murmurhash3FinishFn) {
    rewriter.setInsertionPoint(funcOp);
    murmurhash3FinishFn = rewriter.create<LLVM::LLVMFuncOp>(
        module.getLoc(), murmurhash3FinishFnName,
        LLVM::LLVMFunctionType::get(i32Type, {i32Type}));
    murmurhash3FinishFn.setPrivate();

    Block *funcBlock = murmurhash3FinishFn.addEntryBlock(rewriter);
    ValueRange args = funcBlock->getArguments();
    rewriter.setInsertionPointToEnd(funcBlock);

    auto hash = args[0];

    // Helper lambda to calculate inputHash ^= inputHash >> shiftValue.
    auto hashXorShiftRight = [&](Value inputHash, int32_t shiftValue) -> Value {
      auto constShiftValue = llvmBuilder.i32_val(shiftValue);
      auto hashShiftRight = llvmBuilder.lshr(inputHash, constShiftValue);
      auto hashXor = llvmBuilder.xor_(inputHash, hashShiftRight);
      return hashXor;
    };

    // hash ^= hash >> 16
    auto hashXor16 = hashXorShiftRight(hash, 16);

    // hash *= 0x85ebca6b
    auto constFactor = llvmBuilder.i32_val(0x85ebca6b);
    auto hashMulFactor = llvmBuilder.mul(hashXor16, constFactor);

    // hash ^= hash >> 13
    auto hashXor13 = hashXorShiftRight(hashMulFactor, 13);

    // hash *= 0xc2b2ae35
    auto constFactor2 = llvmBuilder.i32_val(0xc2b2ae35);
    auto hashMulFactor2 = llvmBuilder.mul(hashXor13, constFactor2);

    // hash ^= hash >> 16
    auto hashXor16_2 = hashXorShiftRight(hashMulFactor2, 16);

    // return hash
    rewriter.create<LLVM::ReturnOp>(loc, hashXor16_2);
  }
  return murmurhash3FinishFn;
}

/* Helper to create state hash function.
The state hash is the murmurhash3 of the blockIdx, threadIdx, clock, and
global_timer values.
*/
static LLVM::LLVMFuncOp createStateHashFn(IRRewriter &rewriter, ModuleOp module,
                                          FunctionOpInterface funcOp) {
  auto ctx = rewriter.getContext();
  auto i32Type = type::i32Ty(ctx);
  StringRef stateHashFnName = "state_hash";
  auto loc = funcOp.getLoc();
  TritonLLVMOpBuilder llvmBuilder(loc, rewriter);

  // Create murmurhash3 insert and finish functions.
  LLVM::LLVMFuncOp murmurhash3InsertFn =
      createHashInsertFn(rewriter, module, funcOp);
  LLVM::LLVMFuncOp murmurhash3FinishFn =
      createHashFinishFn(rewriter, module, funcOp);

  LLVM::LLVMFuncOp stateHashFn =
      module.lookupSymbol<LLVM::LLVMFuncOp>(stateHashFnName);
  if (!stateHashFn) {
    rewriter.setInsertionPoint(funcOp);
    stateHashFn = rewriter.create<LLVM::LLVMFuncOp>(
        module.getLoc(), stateHashFnName,
        LLVM::LLVMFunctionType::get(i32Type, {}));
    stateHashFn.setPrivate();

    Block *funcBlock = stateHashFn.addEntryBlock(rewriter);
    rewriter.setInsertionPointToEnd(funcBlock);
    // ret = 0
    auto constZero = llvmBuilder.i32_val(0);

    // Helper lambda to calculate LHS << 16 | RHS.
    auto LHSShiftLeft16OrRHS = [&](Value LHS, Value RHS) -> Value {
      auto const16 = llvmBuilder.i32_val(16);
      auto LHSShiftLeft16 = llvmBuilder.shl(LHS, const16);
      auto orResult = llvmBuilder.or_(LHSShiftLeft16, RHS);
      return orResult;
    };

    // ret = murmurhash3_insert(ret, blockIdx.x << 16 | threadIdx.x)
    auto blockIdx_x = rewriter.create<NVVM::BlockIdXOp>(loc, i32Type);
    auto threadIdx_x = rewriter.create<NVVM::ThreadIdXOp>(loc, i32Type);
    auto shiftedValue1 = LHSShiftLeft16OrRHS(blockIdx_x, threadIdx_x);
    auto insert1 = rewriter.create<LLVM::CallOp>(
        loc, murmurhash3InsertFn, ValueRange{constZero, shiftedValue1});

    // ret = murmurhash3_insert(ret, blockIdx.y << 16 | threadIdx.y)
    auto blockIdx_y = rewriter.create<NVVM::BlockIdYOp>(loc, i32Type);
    auto threadIdx_y = rewriter.create<NVVM::ThreadIdYOp>(loc, i32Type);
    auto shiftedValue2 = LHSShiftLeft16OrRHS(blockIdx_y, threadIdx_y);
    auto insert2 = rewriter.create<LLVM::CallOp>(
        loc, murmurhash3InsertFn,
        ValueRange{insert1.getResult(), shiftedValue2});

    // ret = murmurhash3_insert(ret, blockIdx.z << 16 | threadIdx.z)
    auto blockIdx_z = rewriter.create<NVVM::BlockIdZOp>(loc, i32Type);
    auto threadIdx_z = rewriter.create<NVVM::ThreadIdZOp>(loc, i32Type);
    auto shiftedValue3 = LHSShiftLeft16OrRHS(blockIdx_z, threadIdx_z);
    auto insert3 = rewriter.create<LLVM::CallOp>(
        loc, murmurhash3InsertFn,
        ValueRange{insert2.getResult(), shiftedValue3});

    // Helper lambda to move register value into $0.
    auto getRegisterASM = [&](std::string registerName) -> LLVM::InlineAsmOp {
      return rewriter.create<LLVM::InlineAsmOp>(
          loc, i32Type, ValueRange{},                            // operands
          llvm::formatv("mov.u32 $0, {0};", registerName).str(), // asm_string
          llvm::formatv("=r").str().data(),                      // constraints
          true,  // has_side_effects
          false, // is_align_stack
          LLVM::AsmDialectAttr::get(ctx,
                                    LLVM::AsmDialect::AD_ATT), // asm_dialect
          ArrayAttr()                                          // operand_attrs
      );
    };

    // ret = murmurhash3_insert(ret, globaltimer_lo())
    auto globaltimerLo = getRegisterASM("%globaltimer_lo");
    auto insert4 = rewriter.create<LLVM::CallOp>(
        loc, murmurhash3InsertFn,
        ValueRange{insert3.getResult(), globaltimerLo.getRes()});

    // ret = murmurhash3_insert(ret, globaltimer_hi())
    auto globaltimerHi = getRegisterASM("%globaltimer_hi");
    auto insert5 = rewriter.create<LLVM::CallOp>(
        loc, murmurhash3InsertFn,
        ValueRange{insert4.getResult(), globaltimerHi.getRes()});

    // ret = murmurhash3_insert(ret, clock_lo())
    auto clockLo = getRegisterASM("%clock");
    auto insert6 = rewriter.create<LLVM::CallOp>(
        loc, murmurhash3InsertFn,
        ValueRange{insert5.getResult(), clockLo.getRes()});

    // ret = murmurhash3_insert(ret, clock_hi())
    auto clockHi = getRegisterASM("%clock_hi");
    auto insert7 = rewriter.create<LLVM::CallOp>(
        loc, murmurhash3InsertFn,
        ValueRange{insert6.getResult(), clockHi.getRes()});

    // ret = murmurhash3_finish(ret)
    auto finish = rewriter.create<LLVM::CallOp>(loc, murmurhash3FinishFn,
                                                insert7.getResult());

    // return ret
    rewriter.create<LLVM::ReturnOp>(loc, finish.getResult());
  }
  return stateHashFn;
}

/* Insert random delay.
The delay is calculated as stateHash >> (32 - 21). The delay is then converted
to a nanoseconds value and passed to __nanosleep.
*/
static void insertRandomDelay(IRRewriter &rewriter, Operation *op) {
  auto loc = op->getLoc();
  TritonLLVMOpBuilder llvmBuilder(loc, rewriter);
  auto ctx = rewriter.getContext();
  auto module = op->getParentOfType<ModuleOp>();
  auto funcOp = op->getParentOfType<FunctionOpInterface>();
  auto i32Type = type::i32Ty(ctx);
  auto i64Type = type::i64Ty(ctx);

  // Create state hash function.
  LLVM::LLVMFuncOp stateHashFn = createStateHashFn(rewriter, module, funcOp);

  // Get state hash.
  rewriter.setInsertionPoint(op);
  auto stateHash =
      rewriter.create<LLVM::CallOp>(loc, stateHashFn, ValueRange{});

  // Calculate delay stateHash >> (32 - 21)
  auto ext64StateHash = llvmBuilder.zext(i64Type, stateHash.getResult());
  auto const32Minus21 = llvmBuilder.i64_val(32 - 21);
  auto shiftRightStateHash = llvmBuilder.lshr(ext64StateHash, const32Minus21);
  auto delay = llvmBuilder.trunc(i32Type, shiftRightStateHash);

  // Sleep for the calculated nanoseconds.
  rewriter.create<LLVM::InlineAsmOp>(
      loc, TypeRange(), ValueRange{delay},      // operands
      llvm::formatv("nanosleep.u32 $0;").str(), // asm_string
      llvm::formatv("r").str().data(),          // constraints
      true,                                     // has_side_effects
      false,                                    // is_align_stack
      LLVM::AsmDialectAttr::get(ctx,
                                LLVM::AsmDialect::AD_ATT), // asm_dialect
      ArrayAttr()                                          // operand_attrs
  );
}

namespace {
struct InsertRandomDelays
    : public mlir::triton::impl::InsertRandomDelaysBase<InsertRandomDelays> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());

    llvm::StringRef asmString("");
    auto hasSubstring = [&asmString](const char *prefix) {
      return asmString.contains(prefix);
    };
    auto parseAsmString = [&](Operation *op) {
      if (llvm::any_of(before, hasSubstring)) {
        rewriter.setInsertionPoint(op);
        insertRandomDelay(rewriter, op);
      } else if (llvm::any_of(after, hasSubstring)) {
        rewriter.setInsertionPointAfter(op);
        insertRandomDelay(rewriter, op);
      }
    };

    mod.walk([&](Operation *op) {
      // Synchronization ops can manifest as NVVM ops, inline PTX, or LLVM
      // intrinsics.
      llvm::TypeSwitch<Operation *>(op)
          .Case<NVVM::CpAsyncBulkCommitGroupOp, NVVM::CpAsyncCommitGroupOp>(
              [&](auto) {
                rewriter.setInsertionPoint(op);
                insertRandomDelay(rewriter, op);
              })
          .Case<NVVM::Barrier0Op, NVVM::CpAsyncBulkWaitGroupOp,
                NVVM::CpAsyncWaitGroupOp>([&](auto) {
            rewriter.setInsertionPointAfter(op);
            insertRandomDelay(rewriter, op);
          })
          .Case<LLVM::InlineAsmOp>([&](auto inlineAsmOp) {
            asmString = inlineAsmOp.getAsmString();
            parseAsmString(op);
          })
          .Case<LLVM::CallIntrinsicOp>([&](auto callIntrinsicOp) {
            asmString = callIntrinsicOp.getIntrin();
            parseAsmString(op);
          });
    });
  }
};
} // namespace
