#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using mlir::getSingleCombinerFromReduceOp;

namespace mlir::triton::AMD {

namespace {
template <typename T>
LLVM::LLVMFuncOp getOrInsertFunction(T &moduleOp, const Location loc,
                                     RewriterBase &rewriter, StringRef name,
                                     LLVM::LLVMFunctionType type) {
  LLVM::LLVMFuncOp ret;
  if (!(ret = moduleOp.template lookupSymbol<LLVM::LLVMFuncOp>(name))) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    ret = rewriter.create<LLVM::LLVMFuncOp>(loc, name, type,
                                            LLVM::Linkage::External);
  }
  return ret;
}

// Extend all values to 64-bit per printf call requirements.
Value printfPromoteValue(RewriterBase &rewriter, Value value) {
  auto *context = rewriter.getContext();
  auto loc = UnknownLoc::get(context);
  auto type = value.getType();

  if (isa<LLVM::LLVMPointerType>(type)) {
    // The llvm.ptrtoint op requires signless integer types.
    return ptrtoint(i64_ty, value);
  }

  assert(type.getIntOrFloatBitWidth() <= 64);

  if (auto floatType = dyn_cast<FloatType>(type)) {
    Value newValue = value;
    if (!floatType.isF64())
      newValue = fpext(f64_ty, newValue);
    return bitcast(newValue, i64_ty);
  }

  assert(type.isIntOrIndex());
  if (type.getIntOrFloatBitWidth() < 64) {
    if (type.isUnsignedInteger())
      return zext(ui64_ty, value);
    if (type.isSignedInteger())
      return sext(i64_ty, value);
    // Signless integers are printed using unsigned integer formats.
    return zext(i64_ty, value);
  }

  return value;
}
} // namespace

int TargetInfo::getSharedMemorySize() const { return 64 * 1024; }

bool TargetInfo::supportMaximumMinimum() const { return false; }

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  // On AMD hardware we don't have CTA clusters like NVIDIA. So this will always
  // be zero. Whoever calling into this should make sure the whole program does
  // not try to utilize CTA clusters.
  return rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.amdgcn.ballot",
                                         type, cmp)
      ->getResult(0);
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  if (ctaId.has_value()) {
    llvm::report_fatal_error(
        "AMDGPU does not support cross-CTA shared memory transfers");
  }
  mlir::LLVM::AMD::llStore(rewriter, loc, ptr, val, pred);
}

void TargetInfo::storeMatrixShared(RewriterBase &rewriter, Location loc,
                                   Value ptr, Value val) const {
  llvm::report_fatal_error("AMDGPU does not support stmatrix");
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred) const {
  if (ctaId.has_value()) {
    llvm::report_fatal_error(
        "AMDGPU does not support cross-CTA shared memory transfers");
  }
  Value falseVal = rewriter.create<LLVM::ConstantOp>(
      loc, elemTy, rewriter.getZeroAttr(elemTy));
  return mlir::LLVM::AMD::llLoad(rewriter, loc, ptr, elemTy, pred, falseVal);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::AMD::shuffleXor(loc, rewriter, val, i, getISAFamily());
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  return LLVM::AMD::shuffleUp(loc, rewriter, val, i, getISAFamily());
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::AMD::shuffleIdx(loc, rewriter, val, i, getISAFamily());
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  return LLVM::AMD::shuffleIdx(loc, rewriter, val, i, getISAFamily());
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, int axis) const {
  return LLVM::AMD::llGetPid(loc, rewriter, moduleOp, axis);
}

static inline Type castToInt(RewriterBase &rewriter, Location loc, Value &val,
                             Type valType, unsigned bits) {
  unsigned originalBits = valType.getIntOrFloatBitWidth();
  Type actualType = valType;

  if (!valType.isIntOrIndex()) {
    val = bitcast(val, int_ty(originalBits));
    actualType = int_ty(originalBits);
  }

  if (originalBits < bits) {
    val = sext(int_ty(bits), val);
    actualType = int_ty(bits);
  }

  return actualType;
}

static inline void castFromInt(RewriterBase &rewriter, Location loc, Value &val,
                               Type valType, unsigned bits) {
  unsigned originalBits = valType.getIntOrFloatBitWidth();

  if (originalBits < bits) {
    val = trunc(int_ty(originalBits), val);
  }

  if (!valType.isIntOrIndex()) {
    val = bitcast(val, valType);
  }
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  if (numLaneToReduce != 64)
    return false;

  if (auto family = getISAFamily();
      family != ISAFamily::CDNA3 && family != ISAFamily::CDNA2) {
    return false;
  }

  Operation *reduxOp = getSingleCombinerFromReduceOp(op);
  if (!reduxOp)
    return false;

  auto createDppReduxOp = [&](Type valType, Value &src, int dppCtrl,
                              int rowMask, int bankMask,
                              bool boundCtrl) -> Value {
    // DPP has limited support for data types, so here we need to
    // cast non-integer types or integer types shorter than 32 bits
    // to int32, except for fp32.
    Type actualType = valType;
    if (!valType.isF32()) {
      actualType = castToInt(rewriter, loc, src, valType, 32);
    }

    Value dppResult =
        rewriter
            .create<ROCDL::DPPUpdateOp>(loc, actualType, src, src,
                                        rewriter.getI32IntegerAttr(dppCtrl),
                                        rewriter.getI32IntegerAttr(rowMask),
                                        rewriter.getI32IntegerAttr(bankMask),
                                        rewriter.getBoolAttr(boundCtrl))
            .getRes();

    if (!valType.isF32()) {
      castFromInt(rewriter, loc, src, valType, 32);
      castFromInt(rewriter, loc, dppResult, valType, 32);
    }

    IRMapping mapping;
    mapping.map(reduxOp->getOperand(0), src);
    mapping.map(reduxOp->getOperand(1), dppResult);
    return rewriter.clone(*reduxOp, mapping)->getResult(0);
  };

  for (int i = 0; i < acc.size(); i++) {
    Value buf;
    auto valType = acc[i].getType();

    /*
      Here's the implementation of full-wavefront reduction using dpp.
      https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/

      Each step has a v_mov_dpp instruction following the redux op. In
      some cases, the lower-level compiler could merge them into single
      instruction. For example, v_mov_dpp + max => v_max_dpp.

      In DPP, each row consists of 16 consecutive lanes.
      So the modifier row_shr and row_bcast mean they have the same operations
      in each row, so in the following instructions, we only take row 0
      as an example:

      Step 1: Right shift for 8 lanes.
          lane 8-15 = redux(lane 0-7, lane 8-15)

      Step 2: Right shift for 4 lanes.
          lane 12-15 = redux(lane 8-11, lane 12-15)

      Step 3: Right shift for 2 lanes.
          lane 14-15 = redux(lane 12-13, lane 14-15)

      Step 4: Right shift for 1 lane.
          lane 15 = redux(lane 14, lane 15)

      Step 5: Broadcast lane 15 of each row to all the lanes of its next row.
          lane 16-31 = redux(lane 15, lane 16-31)

      Step 6: Broadcast lane 31 to lane 32-63.
          lane 32-63 = redux(lane 31, lane 32-63)

      Now the reduction result is stored in lane 63.

      Step 7: Read the reduction result from lane 63 and broadcast with
      readlane.
    */

    // row_shr:8
    buf = createDppReduxOp(valType, acc[i], 8 + DppCtrl::ROW_SHR0, 0xf, 0xf,
                           true);

    // row_shr:4
    buf = createDppReduxOp(valType, buf, 4 + DppCtrl::ROW_SHR0, 0xf, 0xf, true);

    // row_shr:2
    buf = createDppReduxOp(valType, buf, 2 + DppCtrl::ROW_SHR0, 0xf, 0xf, true);

    // row_shr:1
    buf = createDppReduxOp(valType, buf, 1 + DppCtrl::ROW_SHR0, 0xf, 0xf, true);

    // row_bcast:15 row_mask:0xa
    buf = createDppReduxOp(valType, buf, DppCtrl::BCAST15, 0xa, 0xf, true);

    // row_bcast:31
    buf = createDppReduxOp(valType, buf, DppCtrl::BCAST31, 0xf, 0xf, true);

    // Similarly, we need to cast data types for readlane instruction.
    Type actualType = castToInt(rewriter, loc, buf, valType, 16);

    // Get reduction result from lane 63
    std::string intrinsic = "llvm.amdgcn.readlane";
    Value result =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, actualType,
                                        ValueRange{buf, i32_val(63)})
            ->getResult(0);

    castFromInt(rewriter, loc, result, valType, 16);

    acc[i] = result;
  }

  return true;
}

void TargetInfo::printfImpl(Value formatStrStart, int formatStrByteCount,
                            ValueRange args, RewriterBase &rewriter,
                            bool useStdErr) const {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto *ctx = rewriter.getContext();
  mlir::Location loc = UnknownLoc::get(ctx);

  // See
  // https://github.com/ROCm/ROCm-Device-Libs/blob/rocm-6.0.x/ockl/src/services.cl#L263-L361
  // for details about the following HIP device print functions.
  LLVM::LLVMFuncOp printBeginFn = getOrInsertFunction(
      moduleOp, loc, rewriter,
      useStdErr ? "__ockl_fprintf_stderr_begin" : "__ockl_printf_begin",
      LLVM::LLVMFunctionType::get(i64_ty,
                                  useStdErr ? ArrayRef<Type>() : i64_ty));
  LLVM::LLVMFuncOp printStrFn = getOrInsertFunction(
      moduleOp, loc, rewriter, "__ockl_printf_append_string_n",
      LLVM::LLVMFunctionType::get(
          i64_ty, {i64_ty, ptr_ty(ctx), /*length=*/i64_ty, /*isLast=*/i32_ty}));
  LLVM::LLVMFuncOp printArgsFn;
  if (!args.empty()) {
    printArgsFn = getOrInsertFunction(
        moduleOp, loc, rewriter, "__ockl_printf_append_args",
        LLVM::LLVMFunctionType::get(
            i64_ty, {i64_ty, /*numArgs=*/i32_ty, i64_ty, i64_ty, i64_ty, i64_ty,
                     i64_ty, i64_ty, i64_ty, /*isLast=*/i32_ty}));
  }

  // Emit the intrinsic function call to begin the printf.
  Value zeroI64 = rewriter.create<LLVM::ConstantOp>(loc, i64_ty, 0);
  Value message =
      call(printBeginFn, useStdErr ? ValueRange() : zeroI64).getResult();

  // Emit the intrinsic function call to handle the printf format string.
  Value oneI32 = i32_val(1);
  Value zeroI32 = i32_val(0);
  Value formatStrLen =
      rewriter.create<LLVM::ConstantOp>(loc, i64_ty, formatStrByteCount);
  SmallVector<Value, 4> arguments = {message, formatStrStart, formatStrLen,
                                     args.empty() ? oneI32 : zeroI32};
  message = call(printStrFn, arguments).getResult();

  // Emit the intrinsic function call to handle arguments iteratively.
  // We can only handle at most 7 values each time.
  constexpr size_t kArgsPerGroup = 7;
  for (size_t group = 0; group < args.size(); group += kArgsPerGroup) {
    size_t bound = std::min(group + kArgsPerGroup, args.size());
    size_t numArgs = bound - group;

    SmallVector<Value, 2 + kArgsPerGroup + 1> arguments;
    arguments.push_back(message);
    arguments.push_back(i32_val(numArgs));
    for (size_t i = group; i < bound; ++i) {
      arguments.push_back(printfPromoteValue(rewriter, args[i]));
    }
    // Pad out to 7 arguments since the function always needs 7 args.
    for (size_t extra = numArgs; extra < kArgsPerGroup; ++extra) {
      arguments.push_back(zeroI64);
    }

    Value isLast = (bound == args.size()) ? oneI32 : zeroI32;
    arguments.push_back(isLast);
    message = call(printArgsFn, arguments).getResult();
  }
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "__ockl_mul_hi_u32" : "__ockl_mul_hi_u64";
  return funcName;
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int formatStrByteCount, ValueRange args) const {
  return printfImpl(formatStrStart, formatStrByteCount, args, rewriter,
                    /*useStdError=*/false);
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg,
                        ValueRange args) const {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue =
      LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()), rewriter,
                              "printfFormat_", msgNewline);
  printf(rewriter, msgValue, msgNewline.size_in_bytes(), args);
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  // Compose and print an assert message.
  llvm::SmallString<256> msgBuffer;
  llvm::Twine("device assertion failed: '" + message + "', in " + func +
              " at " + file + ":" + llvm::Twine(line) + "\n\0")
      .toStringRef(msgBuffer);
  Value msgValue =
      LLVM::addStringToModule(loc, rewriter, "printfFormat_", msgBuffer);
  printfImpl(msgValue, msgBuffer.size_in_bytes(), /*args=*/ValueRange(),
             rewriter, /*useStdError=*/true);

  // Set block barrrier before aborting kernel, give a chance for all
  // the threads in a block to check/print the assert failure.
  barrier();
  // Perform the trap to abort the kernel.
  rewriter.create<LLVM::Trap>(loc);
}

int TargetInfo::getSharedAddressSpace() const { return 3; }

bool TargetInfo::supportVectorizedAtomics() const {
  // Note: not currently tested or used, but AMD generally supports vectorized
  // atomics.
  return true;
}

} // namespace mlir::triton::AMD
