#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "AtomicRMWOpsEmitter.h"

#include <array>
#include <cstdlib>

using namespace triton::AMD;

namespace {

Value generateI32DppMove(RewriterBase &rewriter, Value val, int dppCtrl,
                         int rowMask = 0b1111,  // enable all rows
                         int bankMask = 0b1111, // enable all banks
                         bool boundCtrl = false) {
  assert(val.getType().isInteger(32));
  auto loc = val.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value old = b.i32_val(0);
  auto dppMovOp = ROCDL::DPPUpdateOp::create(
      rewriter, loc, i32_ty, old, val, dppCtrl, rowMask, bankMask, boundCtrl);
  return dppMovOp.getResult();
}

Value shiftLeftI32ByDpp(RewriterBase &rewriter, Value val) {
  return generateI32DppMove(rewriter, val, 0x101); // shift left
}

Value shiftRightI32ByDpp(RewriterBase &rewriter, Value val) {
  return generateI32DppMove(rewriter, val, 0x111); // shift right 1 lane
}

Value generatePopcount64(RewriterBase &rewriter, Value val) {
  auto loc = val.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value m1 = b.i64_val(0x5555555555555555); // binary: 0101 0101..
  Value m2 = b.i64_val(0x3333333333333333); // binary: 0011 0011..
  Value m4 = b.i64_val(0x0f0f0f0f0f0f0f0f); // binary: 0000 1111..
  // binary: 0000 0001 0000 0001..
  Value h01 = b.i64_val(0x0101010101010101);
  // put count of each 2 bits into those 2 bits
  val = b.sub(val, b.and_(m1, b.lshr(val, b.i64_val(1))));
  // put count of each 4 bits into those 4 bits
  val = b.add(b.and_(val, m2), b.and_(b.lshr(val, b.i64_val(2)), m2));
  // put count of each 8 bits into those 8 bits
  val = b.and_(b.add(val, b.lshr(val, b.i64_val(4))), m4);
  // left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
  return b.lshr(b.mul(val, h01), b.i64_val(56));
}

bool getRowShrDppParams(int shift, int &dppCtrl, int &rowMask, int &bankMask,
                        bool &boundCtrl) {
  rowMask = 0xF;
  bankMask = 0xF;
  boundCtrl = false;
  switch (shift) {
  case 1:
    dppCtrl = 0x111; // row_shr:1
    return true;
  case 2:
    dppCtrl = 0x112; // row_shr:2
    return true;
  case 4:
    dppCtrl = 0x114; // row_shr:4
    bankMask = 0xE;
    boundCtrl = true;
    return true;
  case 8:
    dppCtrl = 0x118; // row_shr:8
    bankMask = 0xC;
    boundCtrl = true;
    return true;
  case 16:
    dppCtrl = 0x142; // row_bcast:15
    rowMask = 0xA;
    bankMask = 0xF;
    boundCtrl = true;
    return true;
  case 32:
    dppCtrl = 0x143; // row_bcast:31
    rowMask = 0xC;
    bankMask = 0xF;
    boundCtrl = true;
    return true;
  default:
    return false;
  }
}

Value genReadFirstLane(RewriterBase &rewriter, Value v) {
  auto loc = v.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  std::string intrinsic = "llvm.amdgcn.readfirstlane";
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, i32_ty, v)
      ->getResult(0);
}

Value genPermute(RewriterBase &rewriter, Value v, Value dst) {
  auto loc = v.getLoc();
  std::string intrinsic = "llvm.amdgcn.ds.permute";
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, i32_ty,
                                         ValueRange{dst, v})
      ->getResult(0);
}

Value genBPermute(RewriterBase &rewriter, Value v, Value dst) {
  auto loc = v.getLoc();
  std::string intrinsic = "llvm.amdgcn.ds.bpermute";
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, i32_ty,
                                         ValueRange{dst, v})
      ->getResult(0);
}

template <typename Generator, typename... Values>
Value genI32TiledOp(RewriterBase &rewriter, Generator genCall, Value argToSplit,
                    Values... args) {
  auto loc = argToSplit.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type ty = argToSplit.getType();
  size_t tySize = ty.getIntOrFloatBitWidth();
  size_t i32Size = i32_ty.getIntOrFloatBitWidth();
  size_t count = tySize / i32Size;
  assert(tySize % i32Size == 0 && count > 0 &&
         "Unalligned types are not supported yet.");
  Type i32VecValTy = vec_ty(i32_ty, count);
  Value vec = b.undef(i32VecValTy);
  Value valCasted = b.bitcast(argToSplit, i32VecValTy);
  for (int i = 0; i < count; i++) {
    Value subVal = b.extract_element(i32_ty, valCasted, b.i32_val(i));
    Value result = genCall(rewriter, subVal, args...);
    vec = b.insert_element(i32VecValTy, vec, result, b.i32_val(i));
  }
  return b.bitcast(vec, ty);
}

Value genPrefixSum(RewriterBase &rewriter, Value v0) {
  auto loc = v0.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  auto v0Ty = v0.getType();
  assert(v0Ty.getIntOrFloatBitWidth() == i32_ty.getIntOrFloatBitWidth());

  Value v1 = v0;
  // v_add_f32 v1, v0, v0 row_shr:1 bound_ctrl:0
  Value tmp = generateI32DppMove(rewriter, v0, 0x111);
  v1 = b.add(v1, tmp);

  // v_add_f32 v1, v0, v1 row_shr:2 bound_ctrl:0
  tmp = generateI32DppMove(rewriter, v0, 0x112);
  v1 = b.add(v1, tmp);

  // v_add_f32 v1, v0, v1 row_shr:3 bound_ctrl:0
  tmp = generateI32DppMove(rewriter, v0, 0x113);
  v1 = b.add(v1, tmp);

  // v_add_f32 v1, v1, v1 row_shr:4 bank_mask:0xe
  tmp = generateI32DppMove(rewriter, v1, 0x114, 0xF, 0xE, true);
  v1 = b.add(v1, tmp);

  // v_add_f32 v1, v1, v1 row_shr:8 bank_mask:0xc
  tmp = generateI32DppMove(rewriter, v1, 0x118, 0xF, 0xC, true);
  v1 = b.add(v1, tmp);

  // v_add_f32 v1, v1, v1 row_bcast:15 row_mask:0xa
  tmp = generateI32DppMove(rewriter, v1, 0x142, 0xA, 0xF, true);
  v1 = b.add(v1, tmp);

  // v_add_f32 v1, v1, v1 row_bcast:31 row_mask:0xc
  tmp = generateI32DppMove(rewriter, v1, 0x143, 0xC, 0xF, true);
  v1 = b.add(v1, tmp);

  return v1;
}
} // namespace

namespace mlir::LLVM::AMD {

Value AtomicRMWEmitter::emitAtomicRMW(RewriterBase &rewriter, Value rmwPtr,
                                      Value valElem, Value rmwMask,
                                      std::optional<Value> sharedMemBase,
                                      bool enableIntraWaveReduce) const {
  auto loc = rmwPtr.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type retType = valElem.getType();
  Value undefVal = b.undef(retType);
  // Build blocks to bypass the atomic instruction for ~rmwMask.
  auto *curBlock = rewriter.getInsertionBlock();
  auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
  auto *atomicBlock = rewriter.createBlock(
      curBlock->getParent(), std::next(Region::iterator(curBlock)));
  endBlock->addArgument({retType}, {loc});

  rewriter.setInsertionPointToEnd(curBlock);

  // intraWave reduce optimization for atomic ops needs all active threads
  // at the beginning of a wave. This is achieved as:
  // 1. Compute the prefix sum of the mask, then each active lane gets a
  //    different value (offset) from its previous lane.
  // 2. Multiply the mask and the offset, so only active lanes have a
  //    non-zero offset, and the offset is different in each active lane
  // 3. Sub 1 from offset to get the idx each active lane is moved to
  // 4. Call ds_permute to move active lanes to the beginning of a wave
  // 5. Update mask of each lane
  if (enableIntraWaveReduce) {
    Value maskI32 = b.zext(i32_ty, rmwMask);
    Value offset = genPrefixSum(rewriter, maskI32);
    offset = b.mul(offset, maskI32);
    Value waveSize =
        b.i32_val(mlir::triton::gpu::lookupThreadsPerWarp(rewriter));
    offset = b.select(b.icmp_eq(offset, b.i32_val(0)), waveSize, offset);
    Value idx = b.sub(offset, b.i32_val(1));
    idx = b.mul(idx, b.i32_val(4));
    valElem = genI32TiledOp(rewriter, genPermute, valElem, idx);
    Value castedAddr = b.ptrtoint(i64_ty, rmwPtr);
    castedAddr = genI32TiledOp(rewriter, genPermute, castedAddr, idx);
    rmwPtr = b.inttoptr(rmwPtr.getType(), castedAddr);

    // update mask
    Value maskFlag = targetInfo.ballot(rewriter, loc, i64_ty, rmwMask);
    Value numActiveLanes =
        b.trunc(i32_ty, generatePopcount64(rewriter, maskFlag));

    Value laneID = b.urem(getThreadId(rewriter, loc), waveSize);
    rmwMask = b.icmp_ult(laneID, numActiveLanes);
  }

  LLVM::CondBrOp::create(rewriter, loc, rmwMask, atomicBlock, endBlock,
                         undefVal);

  rewriter.setInsertionPointToEnd(atomicBlock);
  Value atom =
      enableIntraWaveReduce
          ? atomicIntraWaveReduce(rewriter, rmwPtr, valElem, binOp, memOrder,
                                  scopeStr.c_str())
          : LLVM::AtomicRMWOp::create(rewriter, loc, binOp, rmwPtr, valElem,
                                      memOrder, scopeStr.c_str())
                .getResult();

  if (sharedMemBase.has_value()) {
    Value atomPtr = *sharedMemBase;
    b.store(atom, atomPtr);
  }
  LLVM::BrOp::create(rewriter, loc, atom, endBlock);
  rewriter.setInsertionPointToStart(endBlock);

  return endBlock->getArgument(0);
}

Value AtomicRMWEmitter::emitPairedAtomicForEvenTID(RewriterBase &rewriter,
                                                   Value rmwPtr, Value valElem,
                                                   Value rmwMask) const {
  auto loc = rmwPtr.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value i64Ones = b.i64_val(~uint64_t(0));
  Value isOddI32 = b.urem(getThreadId(rewriter, loc), b.i32_val(2));
  // First check if odd threads hold adjacent ptrs to even ones.
  Value castedAddr = b.ptrtoint(i64_ty, rmwPtr);
  // Set casted addr to all ones if the thread is disabled.
  castedAddr = b.select(rmwMask, castedAddr, i64Ones);

  Type valueElemTy = valElem.getType();
  Type packF16Ty = vec_ty(valueElemTy, 2);

  // Move %val to left neighbour to proceed packed atomic further.
  Value packedVal = b.null(packF16Ty);
  packedVal = b.insert_element(packF16Ty, packedVal, valElem, isOddI32);
  // Pack to i32 type to simplify transaction.
  packedVal = b.bitcast(packedVal, i32_ty);
  // Zero operands for disabled threads to make addition no op.
  packedVal = b.select(rmwMask, packedVal, b.i32_val(0));
  Value dppMoveRes = shiftLeftI32ByDpp(rewriter, packedVal);
  Value operand = b.bitcast(b.or_(packedVal, dppMoveRes), packF16Ty);

  Value rightNeighbourAddr =
      genI32TiledOp(rewriter, shiftLeftI32ByDpp, castedAddr);

  // Packing optimization only supported if following conditions are true:
  // 1. address is aligned by 4 bytes
  // 2. right neighbour has adjacent address
  // 3. both threads are active
  Value isAligned = b.icmp_eq(b.urem(castedAddr, b.i64_val(4)), b.i64_val(0));
  Value neighbourAddrAdjacent = b.icmp_eq(
      rightNeighbourAddr,
      b.add(castedAddr, b.i64_val(valueElemTy.getIntOrFloatBitWidth() / 8)));
  Value neighbourEnabled = b.icmp_ne(i64Ones, rightNeighbourAddr);
  Value bothEnabled = b.and_(neighbourEnabled, rmwMask);
  Value enablePackedOpt =
      b.and_(b.and_(isAligned, bothEnabled), neighbourAddrAdjacent);

  // Enable only the even threads.
  Value anyEnabled = b.or_(neighbourEnabled, rmwMask);
  // If one of the threads is disabled, use the neighbour's addr.
  rightNeighbourAddr =
      b.select(neighbourEnabled, rightNeighbourAddr, castedAddr);
  castedAddr = b.select(rmwMask, castedAddr, rightNeighbourAddr);

  rmwMask = b.and_(anyEnabled, b.icmp_eq(isOddI32, b.i32_val(0)));

  // Unpack results back
  Value rightNeighbourPtr = b.inttoptr(rmwPtr.getType(), rightNeighbourAddr);
  rmwPtr = b.inttoptr(rmwPtr.getType(), castedAddr);

  Value undefVal = b.undef(packF16Ty);
  // Build blocks to bypass the atomic instruction for ~rmwMask.
  auto *curBlock = rewriter.getInsertionBlock();
  auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
  auto *atomicBlock = rewriter.createBlock(
      curBlock->getParent(), std::next(Region::iterator(curBlock)));
  endBlock->addArgument({packF16Ty}, {loc});

  rewriter.setInsertionPointToEnd(curBlock);
  LLVM::CondBrOp::create(rewriter, loc, rmwMask, atomicBlock, endBlock,
                         undefVal);

  rewriter.setInsertionPointToEnd(atomicBlock);

  // Determine on the runtime what atomic intrinsic to execute:
  // packed or regular.
  auto *packedBlock = atomicBlock->splitBlock(rewriter.getInsertionPoint());
  auto *regularBlock = rewriter.createBlock(
      atomicBlock->getParent(), std::next(Region::iterator(atomicBlock)));
  rewriter.setInsertionPointToEnd(atomicBlock);

  // If `checkPairs` was set to `false`, `packedBlock` must be removed by DCE
  LLVM::CondBrOp::create(rewriter, loc, enablePackedOpt, packedBlock,
                         regularBlock);

  // Fill out the regular block, where we issue two atomic ops.
  rewriter.setInsertionPointToEnd(regularBlock);
  Value pairedOperand0 = b.extract_element(valueElemTy, operand, b.i32_val(0));
  Value pairedOperand1 = b.extract_element(valueElemTy, operand, b.i32_val(1));
  Value atomNonVec0 = LLVM::AtomicRMWOp::create(
      rewriter, loc, binOp, rmwPtr, pairedOperand0, memOrder, scopeStr.c_str());
  Value atomNonVec1 =
      LLVM::AtomicRMWOp::create(rewriter, loc, binOp, rightNeighbourPtr,
                                pairedOperand1, memOrder, scopeStr.c_str());
  Value packedRes = b.undef(packF16Ty);
  packedRes = b.insert_element(packF16Ty, packedRes, atomNonVec0, b.i32_val(0));
  packedRes = b.insert_element(packF16Ty, packedRes, atomNonVec1, b.i32_val(1));
  LLVM::BrOp::create(rewriter, loc, packedRes, endBlock);

  // Start to fill out the packed block.
  rewriter.setInsertionPointToEnd(packedBlock);

  Value atom = LLVM::AtomicRMWOp::create(rewriter, loc, binOp, rmwPtr, operand,
                                         memOrder, scopeStr.c_str());

  LLVM::BrOp::create(rewriter, loc, atom, endBlock);

  rewriter.setInsertionPointToStart(endBlock);
  Value atomRes = endBlock->getArgument(0);
  // Return packed to i32 result after atomic operation back from
  // master lane.
  auto packedRet = b.bitcast(atomRes, i32_ty);
  Value dppMovRes = shiftRightI32ByDpp(rewriter, packedRet);
  // Unpack results back
  Value unpackedDppRes = b.bitcast(dppMovRes, packF16Ty);
  atomRes = b.insert_element(
      packF16Ty, atomRes,
      b.extract_element(valueElemTy, unpackedDppRes, b.i32_val(1)),
      b.i32_val(1));
  return b.extract_element(valueElemTy, atomRes,
                           b.urem(getThreadId(rewriter, loc), b.i32_val(2)));
}

Value AtomicRMWEmitter::atomicIntraWaveReduce(RewriterBase &rewriter,
                                              Value rmwPtr, Value operand,
                                              LLVM::AtomicBinOp opKind,
                                              LLVM::AtomicOrdering memOrdering,
                                              StringRef scope) const {
  // This approach minimizes intra-warp thread contention when accessing
  // global memory pointers. It is particularly advantageous for certain ISA
  // families, such as CDNA3. The algorithm follows these steps:
  // 1. Analyze thread groups and their relative positions:
  // 1.1. Consider groups of threads sharing identical pointers using
  //      `readfirstlane` and ballot `intrinsics`.
  // 1.2. Compute parameters to form contiguous groups and further optimize
  //      them.
  // 1.3. Disable threads that have already been processed.
  // 1.4. If thread was not considered, jump to `1.1.`.
  // 2. Form contiguous groups:
  //    Use `permute` instructions to organize threads within the wavefront
  //    into continuous groups.
  // 4. Reduce Groups to Leader threads:
  //    Apply `bpermute` and operation-specific arithmetic based on the
  //    opKind to consolidate group data into leader threads.
  // 5. Perform global atomic operations by leader threads.
  auto loc = operand.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type operandElemType = operand.getType();
  Type origPtrType = rmwPtr.getType();

  rmwPtr = b.ptrtoint(i64_ty, rmwPtr);

  auto *curBlock = rewriter.getInsertionBlock();
  auto *atomicBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
  atomicBlock->addArgument(i64_ty, loc);
  atomicBlock->addArgument(operandElemType, loc);
  auto *initLoop = rewriter.createBlock(curBlock->getParent(),
                                        std::next(Region::iterator(curBlock)));

  rewriter.setInsertionPointToEnd(curBlock);

  LLVM::BrOp::create(rewriter, loc, initLoop);
  rewriter.setInsertionPointToEnd(initLoop);

  auto *afterLoopBlock = initLoop->splitBlock(rewriter.getInsertionPoint());
  afterLoopBlock->addArgument(i32_ty, loc);    // idx
  afterLoopBlock->addArgument(i32_ty, loc);    // cnt
  afterLoopBlock->addArgument(int_ty(1), loc); // isLeader
  afterLoopBlock->addArgument(i32_ty, loc);    // maxCnt

  auto *loopBody = rewriter.createBlock(initLoop->getParent(),
                                        std::next(Region::iterator(initLoop)));
  loopBody->addArgument(i32_ty, loc);
  loopBody->addArgument(i32_ty, loc);

  rewriter.setInsertionPointToEnd(initLoop);
  rewriter.create<LLVM::BrOp>(loc, ValueRange({b.i32_val(0), b.i32_val(1)}),
                              loopBody);

  // Greed search of same addr within wavefront. Also collect auxiliary
  // information about relative position:
  // - idx in a group + base laneId. This param is required to form
  // continuous
  //   groups further;
  // - cnt of remaining threads in a group after current thread;
  // - leadership status of the current thread.
  rewriter.setInsertionPointToEnd(loopBody);
  // `readfirstlane` considers only enabled threads
  Value chosen = genI32TiledOp(rewriter, genReadFirstLane, rmwPtr);
  // this flag is required to disable thread if we have already checked its
  // pointer
  Value done = b.icmp_eq(chosen, rmwPtr);
  Value mask = targetInfo.ballot(rewriter, loc, i64_ty, done);
  Value start = loopBody->getArgument(0);
  Value currentMax = loopBody->getArgument(1);
  Value cnt = b.trunc(i32_ty, generatePopcount64(rewriter, mask));
  Value maskLo = b.trunc(i32_ty, mask);
  Value mbcntLoRes =
      ROCDL::MbcntLoOp::create(rewriter, loc, i32_ty, maskLo, b.i32_val(0),
                               /*arg_attrs=*/{}, /*res_attrs=*/{});
  Value maskHi = b.trunc(i32_ty, b.lshr(mask, b.i64_val(32)));
  Value idx =
      ROCDL::MbcntHiOp::create(rewriter, loc, i32_ty, maskHi, mbcntLoRes,
                               /*arg_attrs=*/{}, /*res_attrs=*/{});
  Value base = b.add(start, cnt);
  Value updatedMax =
      b.select(b.icmp_ugt(cnt, currentMax), cnt, currentMax);
  Value leader = b.icmp_eq(idx, b.i32_val(0));
  cnt = b.sub(cnt, idx);
  idx = b.add(idx, start);
  LLVM::CondBrOp::create(rewriter, loc, done, afterLoopBlock,
                         ValueRange({idx, cnt, leader, updatedMax}), loopBody,
                         ValueRange({base, updatedMax}));

  rewriter.setInsertionPointToEnd(afterLoopBlock);

  Value idxRes = afterLoopBlock->getArgument(0);
  Value cntRes = afterLoopBlock->getArgument(1);
  Value leaderRes = afterLoopBlock->getArgument(2);
  Value maxCntRes = afterLoopBlock->getArgument(3);

  Value leaderMask =
      targetInfo.ballot(rewriter, loc, i64_ty, leaderRes);
  Value uniqueAddrCount =
      b.trunc(i32_ty, generatePopcount64(rewriter, leaderMask));

  int waveSizeInt = mlir::triton::gpu::lookupThreadsPerWarp(rewriter);
  Type scalarTy = operandElemType;
  if (auto vecTy = dyn_cast<VectorType>(operandElemType))
    scalarTy = vecTy.getElementType();
  bool isScalarFloat = mlir::isa<FloatType>(scalarTy);
  bool isScalarInt =
      mlir::isa<IntegerType>(scalarTy) || mlir::isa<IndexType>(scalarTy);
  bool isAddLike = opKind == LLVM::AtomicBinOp::fadd ||
                   (opKind == LLVM::AtomicBinOp::add &&
                    (isScalarFloat || isScalarInt));

  int coopThresholdInt = waveSizeInt >> (isAddLike ? 1 : 2);
  coopThresholdInt = std::max(1, coopThresholdInt);
  if (const char *env = std::getenv("TRITON_ATOMIC_COOP_THRESHOLD")) {
    char *endPtr = nullptr;
    long parsed = std::strtol(env, &endPtr, 10);
    if (endPtr != env && parsed > 0)
      coopThresholdInt = static_cast<int>(parsed);
  }

  Value enableCoop =
      b.icmp_ult(uniqueAddrCount, b.i32_val(coopThresholdInt));

  auto *coopBlock = rewriter.createBlock(
      afterLoopBlock->getParent(),
      std::next(Region::iterator(afterLoopBlock)));
  coopBlock->addArgument(rmwPtr.getType(), loc);
  coopBlock->addArgument(operandElemType, loc);
  coopBlock->addArgument(i32_ty, loc);
  coopBlock->addArgument(i32_ty, loc);
  coopBlock->addArgument(int_ty(1), loc);
  coopBlock->addArgument(i32_ty, loc);

  rewriter.setInsertionPointToEnd(afterLoopBlock);
  rewriter.create<LLVM::CondBrOp>(
      loc, enableCoop, coopBlock,
      ValueRange({rmwPtr, operand, idxRes, cntRes, leaderRes, maxCntRes}),
      atomicBlock, ValueRange({rmwPtr, operand}));

  rewriter.setInsertionPointToStart(coopBlock);
  Value coopRmwPtr = coopBlock->getArgument(0);
  Value coopOperand = coopBlock->getArgument(1);
  Value coopIdxRes = coopBlock->getArgument(2);
  Value coopCntRes = coopBlock->getArgument(3);
  Value coopLeaderRes = coopBlock->getArgument(4);
  Value coopMaxCnt = coopBlock->getArgument(5);
  Value idxScaledBase = b.mul(coopIdxRes, b.i32_val(4));
  Value needFullPermute = b.icmp_ugt(coopMaxCnt, b.i32_val(16));

  auto *afterArrangeBlock = coopBlock->splitBlock(rewriter.getInsertionPoint());
  auto addArrangeArgs = [&](Block *block) {
    block->addArgument(rmwPtr.getType(), loc);
    block->addArgument(operandElemType, loc);
    block->addArgument(i32_ty, loc);
    block->addArgument(i32_ty, loc);
    block->addArgument(int_ty(1), loc);
    block->addArgument(i32_ty, loc);
  };
  addArrangeArgs(afterArrangeBlock);

  auto insertBefore = Region::iterator(afterArrangeBlock);
  auto *fullPermuteBlock =
      rewriter.createBlock(coopBlock->getParent(), insertBefore);
  addArrangeArgs(fullPermuteBlock);
  auto *simplePermuteBlock =
      rewriter.createBlock(coopBlock->getParent(), insertBefore);
  addArrangeArgs(simplePermuteBlock);

  rewriter.setInsertionPointToEnd(coopBlock);
  rewriter.create<LLVM::CondBrOp>(
      loc, needFullPermute, fullPermuteBlock,
      ValueRange({coopRmwPtr, coopOperand, idxScaledBase, coopCntRes,
                  coopLeaderRes, coopMaxCnt}),
      simplePermuteBlock,
      ValueRange({coopRmwPtr, coopOperand, idxScaledBase, coopCntRes,
                  coopLeaderRes, coopMaxCnt}));

  rewriter.setInsertionPointToEnd(simplePermuteBlock);
  rewriter.create<LLVM::BrOp>(
      loc, ValueRange(simplePermuteBlock->getArguments()), afterArrangeBlock);

  rewriter.setInsertionPointToEnd(fullPermuteBlock);
  Value arrangedPtr = fullPermuteBlock->getArgument(0);
  Value arrangedOperand = fullPermuteBlock->getArgument(1);
  Value arrangedIdx = fullPermuteBlock->getArgument(2);
  Value arrangedCnt = fullPermuteBlock->getArgument(3);
  Value arrangedLeader = fullPermuteBlock->getArgument(4);
  Value arrangedMax = fullPermuteBlock->getArgument(5);

  arrangedPtr =
      genI32TiledOp(rewriter, genPermute, arrangedPtr, arrangedIdx);
  arrangedOperand =
      genI32TiledOp(rewriter, genPermute, arrangedOperand, arrangedIdx);
  Value packedRoleInfo = genI32TiledOp(
      rewriter, genPermute,
      b.or_(b.zext(i32_ty, arrangedLeader),
            b.or_(arrangedIdx, b.shl(arrangedCnt, b.i32_val(8)))),
      arrangedIdx);
  Value permutedCnt =
      b.and_(b.lshr(packedRoleInfo, b.i32_val(8)), b.i32_val(0xff));
  Value permutedLeader =
      b.icmp_ne(b.and_(packedRoleInfo, b.i32_val(1)), b.i32_val(0));
  rewriter.create<LLVM::BrOp>(
      loc,
      ValueRange({arrangedPtr, arrangedOperand, packedRoleInfo, permutedCnt,
                  permutedLeader, arrangedMax}),
      afterArrangeBlock);

  rewriter.setInsertionPointToStart(afterArrangeBlock);
  Value coopRmwPtrArranged = afterArrangeBlock->getArgument(0);
  Value coopOperandArranged = afterArrangeBlock->getArgument(1);
  Value idxScaledForPermute = afterArrangeBlock->getArgument(2);
  Value coopCntResArranged = afterArrangeBlock->getArgument(3);
  Value coopLeaderResArranged = afterArrangeBlock->getArgument(4);
  Value coopMaxCntArranged = afterArrangeBlock->getArgument(5);

  coopRmwPtr = coopRmwPtrArranged;
  coopOperand = coopOperandArranged;
  coopCntRes = coopCntResArranged;
  coopLeaderRes = coopLeaderResArranged;
  coopMaxCnt = coopMaxCntArranged;

  auto *afterRedBlock =
      rewriter.splitBlock(afterArrangeBlock, afterArrangeBlock->begin());
  afterRedBlock->addArgument(operandElemType, loc);
  auto *partialReductionBlock = rewriter.createBlock(
      afterArrangeBlock->getParent(),
      std::next(Region::iterator(afterArrangeBlock)));
  rewriter.setInsertionPointToEnd(afterArrangeBlock);
  Value reductionCond = b.icmp_sgt(coopMaxCnt, b.i32_val(1));
  rewriter.create<LLVM::CondBrOp>(loc, reductionCond, partialReductionBlock,
                                  ValueRange(), afterRedBlock,
                                  ValueRange({coopOperand}));
  rewriter.setInsertionPointToEnd(partialReductionBlock);

  auto performOp = [&](Value res, Value v) -> Value {
    switch (opKind) {
    case LLVM::AtomicBinOp::_and:
      return b.and_(res, v);
    case LLVM::AtomicBinOp::_or:
      return b.or_(res, v);
    case LLVM::AtomicBinOp::_xor:
      return b.xor_(res, v);
    case LLVM::AtomicBinOp::add:
      return b.add(res, v);
    case LLVM::AtomicBinOp::fadd:
      return b.fadd(res, v);
    case LLVM::AtomicBinOp::max:
    case LLVM::AtomicBinOp::umax:
      return b.umax(v, res);
    case LLVM::AtomicBinOp::min:
    case LLVM::AtomicBinOp::umin:
      return b.umin(v, res);
    case LLVM::AtomicBinOp::xchg:
      return v;
    default:
      llvm_unreachable("Unsupported atomic binary operation.");
    }
  };
  std::array<Value, 6> permuteOffsets{};
  auto strideToIndex = [](int stride) -> unsigned {
    switch (stride) {
    case 32:
      return 0;
    case 16:
      return 1;
    case 8:
      return 2;
    case 4:
      return 3;
    case 2:
      return 4;
    case 1:
      return 5;
    default:
      llvm_unreachable("Unexpected stride value");
    }
  };

  auto getPermuteOffset = [&](int stride) -> Value {
    unsigned index = strideToIndex(stride);
    Value cached = permuteOffsets[index];
    if (cached)
      return cached;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(partialReductionBlock);
    Value offsetVal = b.add(idxScaledForPermute, b.i32_val(stride * 4));
    permuteOffsets[index] = offsetVal;
    return offsetVal;
  };

  auto computeTmp = [&](Value currentAcc, int stride) -> Value {
    int dppCtrl = 0;
    int rowMask = 0;
    int bankMask = 0;
    bool boundCtrl = false;
    if (getRowShrDppParams(stride, dppCtrl, rowMask, bankMask, boundCtrl))
      return genI32TiledOp(rewriter, generateI32DppMove, currentAcc, dppCtrl,
                           rowMask, bankMask, boundCtrl);
    Value offset = getPermuteOffset(stride);
    return genI32TiledOp(rewriter, genBPermute, currentAcc, offset);
  };

  Value acc = coopOperand;
  rewriter.setInsertionPointToEnd(partialReductionBlock);
  constexpr int reductionStrides[] = {32, 16, 8, 4, 2, 1};

  if (isScalarFloat && opKind == LLVM::AtomicBinOp::fadd) {
    for (int stride : reductionStrides) {
      Value strideVal = b.i32_val(stride);
      Value reductionGuard = b.and_(
          b.icmp_ugt(coopMaxCnt, strideVal),
          b.icmp_ult(strideVal, coopCntRes));
      Value tmp = computeTmp(acc, stride);
      Value combined = b.fadd(acc, tmp);
      acc = b.select(reductionGuard, combined, acc);
    }
  } else if (isScalarInt && opKind == LLVM::AtomicBinOp::add) {
    for (int stride : reductionStrides) {
      Value strideVal = b.i32_val(stride);
      Value reductionGuard = b.and_(
          b.icmp_ugt(coopMaxCnt, strideVal),
          b.icmp_ult(strideVal, coopCntRes));
      Value tmp = computeTmp(acc, stride);
      Value combined = b.add(acc, tmp);
      acc = b.select(reductionGuard, combined, acc);
    }
  } else {
    for (int stride : reductionStrides) {
      Value strideVal = b.i32_val(stride);
      Value reductionGuard = b.and_(
          b.icmp_ugt(coopMaxCnt, strideVal),
          b.icmp_ult(strideVal, coopCntRes));
      Value tmp = computeTmp(acc, stride);
      Value combined = performOp(acc, tmp);
      acc = b.select(reductionGuard, combined, acc);
    }
  }

  rewriter.create<LLVM::BrOp>(loc, ValueRange({acc}), afterRedBlock);
  rewriter.setInsertionPointToEnd(afterRedBlock);

  auto *endBlock = afterRedBlock->splitBlock(rewriter.getInsertionPoint());
  endBlock->addArgument(operandElemType, loc);
  rewriter.setInsertionPointToEnd(afterRedBlock);
  Value leaderCond = coopLeaderRes;
  Value defaultRes = b.undef(operandElemType);
  SmallVector<Value, 2> leaderTrueArgs = {coopRmwPtr,
                                          afterRedBlock->getArgument(0)};
  SmallVector<Value, 1> leaderFalseArgs = {defaultRes};
  rewriter.create<LLVM::CondBrOp>(loc, leaderCond, atomicBlock, leaderTrueArgs,
                                  endBlock, leaderFalseArgs);
  rewriter.setInsertionPointToEnd(atomicBlock);
  // Utilize global atomic only by leader threads
  Value addr = atomicBlock->getArgument(0);
  Value atomAddr = b.inttoptr(origPtrType, addr);
  Value atom = LLVM::AtomicRMWOp::create(rewriter, loc, opKind, atomAddr,
                                         atomicBlock->getArgument(1),
                                         memOrdering, scope);
  LLVM::BrOp::create(rewriter, loc, atom, endBlock);
  rewriter.setInsertionPointToStart(endBlock);

  return endBlock->getArgument(0);
}

} // namespace mlir::LLVM::AMD
