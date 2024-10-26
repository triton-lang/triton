#include "Utility.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using mlir::triton::gpu::appendOrGetExternFuncOp;
using mlir::triton::gpu::getFunctionType;

namespace {
enum class ShflKind : uint32_t {
  bfly = 0,
  up = 1,
  down = 2,
  idx = 3,
};

std::string getTypeString(Type ty) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  ty.print(rso);
  rso.flush();
  return str;
}

std::string mangleFunc(std::string name, Type type) {
  auto funcType = dyn_cast<LLVM::LLVMFunctionType>(type);
  assert(funcType && "Expecting an LLVMFunctionType");
  std::string mangled = name + "_";
  auto retTy = funcType.getReturnType();
  mangled += getTypeString(retTy) + "_";
  auto params = funcType.getParams();
  for (auto paramType : params) {
    mangled += getTypeString(paramType) + "_";
  }
  return mangled;
}

// Utility function to create a constant vector mask of length `vecSize` with
// the same `pred` value
Value createVectorMaskFromPredicate(RewriterBase &rewriter, Location loc,
                                    Value pred, int64_t vecSize) {
  auto vecMaskTy = LLVM::getFixedVectorType(rewriter.getI1Type(), vecSize);
  Value maskVal = undef(vecMaskTy);
  for (size_t s = 0; s < vecSize; ++s) {
    Value indexVal =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(s));
    maskVal = insert_element(vecMaskTy, maskVal, pred, indexVal);
  }
  return maskVal;
}

// Utility function to get the number of elements of a vector or a scalar
int64_t getNumElements(Type ty) {
  if (auto vecType = dyn_cast<VectorType>(ty))
    return vecType.getNumElements();
  return 1;
}

// Utility function to cast the given scalar or vector type to a vector type
Type castToVectorType(Type ty) {
  if (isa<VectorType>(ty))
    return ty;
  return LLVM::getFixedVectorType(ty, 1);
}

} // namespace

namespace mlir::LLVM::AMD {
static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value i, int strideInt, ShflKind mode, Value clamp) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  // On AMD, the ds_swizzle_b32 and ds_permute_b32 instructions work on
  // 32bit/dwords so we need promote to 32 here.
  auto valType = val.getType();
  if (!valType.isInteger(32) && bits <= 32) {
    if (!valType.isIntOrIndex())
      val = bitcast(val, int_ty(bits));
    if (bits < 32)
      val = sext(i32_ty, val);

    val = shuffleCommon(loc, rewriter, val, i, strideInt, mode, clamp);

    if (bits < 32)
      val = trunc(int_ty(bits), val);
    if (!valType.isIntOrIndex())
      val = bitcast(val, valType);
    return val;
  }

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = bitcast(val, vecTy);
    Value val0 = extract_element(f32_ty, vec, i32_val(0));
    Value val1 = extract_element(f32_ty, vec, i32_val(1));
    val0 = shuffleCommon(loc, rewriter, val0, i, strideInt, mode, clamp);
    val1 = shuffleCommon(loc, rewriter, val1, i, strideInt, mode, clamp);
    vec = undef(vecTy);
    vec = insert_element(vecTy, vec, val0, i32_val(0));
    vec = insert_element(vecTy, vec, val1, i32_val(1));
    return bitcast(vec, val.getType());
  }

  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  Value threadId =
      rewriter.create<::mlir::gpu::ThreadIdOp>(loc, ::mlir::gpu::Dimension::x);
  threadId = rewriter.create<arith::IndexCastOp>(loc, i32_ty, threadId);
  unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  Value warpSize = i32_val(iWarpSize);
  Value laneId = urem(threadId, warpSize);
  auto bpermute = [&](Value lane) {
    // Multiple lineId by 4. (More on permute instruction semantics:
    // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf#page=180
    Value byteOffset = i32_val(2);
    Value permuteAddr = shl(lane, byteOffset);
    return rewriter.create<ROCDL::DsBpermuteOp>(loc, valType, permuteAddr, val);
  };

  switch (mode) {
  case ShflKind::bfly:
    if (strideInt > 16) {
      Value threadId =
          rewriter
              .create<UnrealizedConversionCastOp>(
                  loc, TypeRange{i32_ty},
                  ValueRange{rewriter.create<::mlir::gpu::ThreadIdOp>(
                      loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x)})
              .getResult(0);
      Value stride = i32_val(32);
      Value lineId = xor_(threadId, stride);
      return bpermute(lineId);
    } else {
      // This map facilates the butterfly shuffle pattern for a stride less
      // than 16. The pattern stride is the key of the map.
      DenseMap<short, unsigned int> masks{
          {16, 0x401F}, {8, 0x201F}, {4, 0x101F}, {2, 0x081F}, {1, 0x041F}};
      Value offset = i32_val(masks[strideInt]);
      return rewriter.create<ROCDL::DsSwizzleOp>(loc, valType, val, offset);
    }
    break;
  case ShflKind::up: {
    Value mask = icmp_slt(laneId, i);
    Value delta = sub(laneId, i);
    Value index = select(mask, laneId, delta);
    return bpermute(index);
  }
  case ShflKind::idx:
    return bpermute(i);
  default:
    assert(false && "Unsupported ShflKind");
    break;
  }
  return Value();
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), i, ShflKind::bfly,
                       i32_val(0x1f));
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), i, ShflKind::up,
                       i32_val(0x0));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleIdx(loc, rewriter, val, i32_val(i));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i) {
  return shuffleCommon(loc, rewriter, val, i, 0, ShflKind::idx, i32_val(0x1f));
}

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               int axis) {
  assert(axis >= 0);
  assert(axis < 3);
  assert(moduleOp);
  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
  Value blockId = rewriter.create<::mlir::gpu::BlockIdOp>(loc, dims[axis]);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, blockId);
}

Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal, int64_t alignmentBytes,
             triton::CacheModifier cm) {

  // Try to emit llvm.intr.masked.load if we can. In theory the backend should
  // be happier because we emit less branchy code to optimize. The backend will
  // lower it down however it wants at some point.
  if (alignmentBytes &&
      (cm == triton::CacheModifier::CG || cm == triton::CacheModifier::NONE)) {
    // `llvm.intr.masked.load` only accepts vectors. If we see a scalar we need
    // to bitcast to `vector<1xelemTy>` (and back)
    int64_t vecSize = getNumElements(elemTy);
    Type vecType = castToVectorType(elemTy);
    falseVal = bitcast(falseVal, vecType);
    Value maskVal = createVectorMaskFromPredicate(rewriter, loc, pred, vecSize);
    bool nt = (cm == triton::CacheModifier::CG);
    Value vecData = rewriter.create<LLVM::MaskedLoadOp>(
        loc, vecType, ptr, maskVal, falseVal, alignmentBytes, nt);
    // If it is not a vector, remember to bitcast back to a scalar
    vecData = bitcast(vecData, elemTy);
    return vecData;
  }

  Type funcType = getFunctionType(elemTy, ValueRange({ptr, pred, falseVal}));
  auto parent = ptr.getParentRegion()->getParentOfType<LLVM::LLVMFuncOp>();
  auto getLoadNameRaw = [](triton::CacheModifier cm) {
    switch (cm) {
    case triton::CacheModifier::CA:
      return predicatedLoadCA;
    case triton::CacheModifier::CG:
      return predicatedLoadCG;
    case triton::CacheModifier::CV:
      return predicatedLoadCV;
    default:
      // Do not fail in compile time in the case of unsupported modifier.
      // Just apply default config.
      return predicatedLoad;
    }
  };

  auto funcName = mangleFunc(getLoadNameRaw(cm), funcType);
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, parent, funcName, funcType);
  return LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                ValueRange({ptr, pred, falseVal}))
      .getResult();
}

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, int64_t alignmentBytes, triton::CacheModifier cm) {
  // Try to emit llvm.intr.masked.store if we can. In theory the backend should
  // be happier because we emit less branchy code to optimize. The backend will
  // lower it down however it wants at some point.
  if (alignmentBytes && cm == triton::CacheModifier::NONE) {
    // `llvm.intr.masked.store` only accepts vectors. If we see a scalar we need
    // to bitcast to `vector<1xelemTy>`
    Type elemTy = val.getType();
    int64_t vecSize = getNumElements(elemTy);
    Type vecType = castToVectorType(elemTy);
    val = bitcast(val, vecType);
    Value maskVal = createVectorMaskFromPredicate(rewriter, loc, pred, vecSize);
    auto op = rewriter.create<LLVM::MaskedStoreOp>(loc, val, ptr, maskVal,
                                                   alignmentBytes);
    return;
  }

  auto ctx = ptr.getContext();
  Type funcType = getFunctionType(void_ty(ctx), ValueRange({ptr, val, pred}));
  auto parent = ptr.getParentRegion()->getParentOfType<LLVM::LLVMFuncOp>();
  auto getStoreNameRaw = [](triton::CacheModifier cm) {
    switch (cm) {
    case triton::CacheModifier::WT:
      return predicatedStoreWT;
    case triton::CacheModifier::CG:
      return predicatedStoreCG;
    case triton::CacheModifier::CS:
      return predicatedStoreCS;
    default:
      // Do not fail in compile time in the case of unsupported modifier.
      // Just apply default config.
      return predicatedStore;
    }
  };
  auto funcName = mangleFunc(getStoreNameRaw(cm), funcType);
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, parent, funcName, funcType);
  LLVM::createLLVMCallOp(rewriter, loc, funcOp, ValueRange({ptr, val, pred}));
}

bool emitTransferBetweenRegistersAndShared(
    RankedTensorType registerTy, MemDescType sharedTy, Type elemLlvmTy,
    std::optional<int32_t> maxVecElems, Value shmemBase,
    ArrayRef<Value> shmemStrides, Location loc, RewriterBase &rewriter,
    const TargetInfoBase &target, bool crossGrain,
    std::function<void(VectorType, Value /*shmemAddr*/)> perVectorCallback) {
  MLIRContext *ctx = rewriter.getContext();

  auto shape = registerTy.getShape();
  // LDBG("shape: " << shape[0] << " " << shape[1]);
  int rank = shape.size();

  StringAttr kBlock = str_attr("block");
  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");

  std::optional<LinearLayout> regLayout = LinearLayout::empty();
  auto regEncoding = registerTy.getEncoding();
  if (crossGrain)
    regLayout =
      mlir::triton::gpu::blockedToLinearLayoutThreadRake(shape, regEncoding);
  else
    regLayout =
      triton::gpu::toLinearLayout(shape, regEncoding);
  std::optional<LinearLayout> sharedLayout = triton::gpu::toLinearLayout(
      shape, sharedTy.getEncoding(), elemLlvmTy.getIntOrFloatBitWidth());
  if (!regLayout.has_value() || !sharedLayout.has_value()) {
    return false;
  }
  LDBG("-----regLayout-----");
  LDBG(regLayout);
  LDBG("-----sharedLayout-----");
  LDBG(sharedLayout);
  auto sharedOrder = triton::gpu::getOrder(sharedTy.getEncoding());
  // LDBG("sharedOrder: " << sharedOrder[0] << " " << sharedOrder[1]);

  // sharedLayout's in-dims are currently (offset, block).  Reshape to
  // (offsetX1, offsetX2, ..., block) so that we can apply the N-dimensional
  // shmem strides.  (The offsetX's appear in minor-to-major order.)
  auto sharedLegacy =
      cast<triton::gpu::SharedEncodingAttr>(sharedTy.getEncoding());
  SmallVector<std::pair<StringAttr, int32_t>> multiDimSharedSize;
  // LDBG("multiDimSharedSize");
  for (int i = 0; i < rank; i++) {
    int dim = sharedOrder[i];
    int64_t size = std::max(
        int64_t{1},
        shape[dim] / sharedLegacy.getCTALayout().getCTASplitNum()[dim]);
    multiDimSharedSize.push_back(
        {str_attr("offset" + std::to_string(dim)), size});
    // LDBG(multiDimSharedSize.back().first << ": " << multiDimSharedSize.back().second);
  }
  multiDimSharedSize.push_back({kBlock, sharedLayout->getInDimSize(kBlock)});
  // LDBG(multiDimSharedSize.back().first << ": " << multiDimSharedSize.back().second);
  sharedLayout = sharedLayout->reshapeIns(multiDimSharedSize);

  // regToSharedLayout maps from (register, lane, warp, block) to (offsetX1,
  // ..., offsetXN, block), where the offsetX's are in minor-to-major order.
  LinearLayout regToSharedLayout = regLayout->invertAndCompose(*sharedLayout);
  LDBG("-----regToSharedLayout-----");
  LDBG(regToSharedLayout);

  for (int inBlock = 1; inBlock < regToSharedLayout.getInDimSize(kBlock);
       inBlock *= 2) {
    auto idx = llvm::to_vector(llvm::make_second_range(regToSharedLayout.apply(
        {{kRegister, 0}, {kLane, 0}, {kWarp, 0}, {kBlock, inBlock}})));
    // offsetX1, ..., offsetXN must all be 0.
    if (!llvm::all_of(ArrayRef(idx).drop_back(1),
                      [&](auto offset) { return offset == 0; })) {
      return false;
    }
    // Check if there's any cross CTA load.
    int32_t outBlock = idx.back();
    if (outBlock != inBlock) {
      return false;
    }
  }

  // Determine how many consecutive registers map to consecutive shmem elements
  // in out-dimension offsetN.  This is our load instruction's vector width.
  //
  // It's OK if the vector width we choose here is wider than the hardware
  // supports; LLVM will legalize it.
  //
  const int vecElems =
      std::min(regToSharedLayout.getNumConsecutiveInOut(),
               maxVecElems.value_or(std::numeric_limits<int>::max()));
  LDBG("vecElems = min(" << regToSharedLayout.getNumConsecutiveInOut() <<
    ", " << maxVecElems.value_or(std::numeric_limits<int>::max()) <<
    ") = " << vecElems);

  Value threadId = getThreadId(rewriter, loc);
  Value threadsPerWarp = i32_val(regToSharedLayout.getInDimSize(kLane));
  Value laneId = urem(threadId, threadsPerWarp);
  Value warpId = udiv(threadId, threadsPerWarp);

  int numElems = regToSharedLayout.getInDimSize(kRegister);
  auto vecTy = vec_ty(elemLlvmTy, vecElems);
  auto ptrTy = ptr_ty(ctx, /*addressSpace=*/3);
  Value zero = i32_val(0);
  SmallVector<Value> ret;
  for (int i = 0; i < numElems / vecElems; i++) {
    // Get the address to load/store.  The multi-dim address is (offsetX1, ...,
    // offsetXN, block), where the offsets appear in minor-to-major order, and
    // we drop_end to drop block, which we know from above will be 0.
    auto multiDimShmemOffset =
        llvm::to_vector(llvm::drop_end(llvm::make_second_range(
            applyLinearLayout(loc, rewriter, regToSharedLayout,
                              {{kRegister, i32_val(i * vecElems)},
                               {kLane, laneId},
                               {kWarp, warpId},
                               {kBlock, zero}}))));

    // Reorder strides according to `order`.  This way they match the
    // multi-dimensional offsets in regToSharedLayout.
    Value shmemOffset = dot(rewriter, loc, multiDimShmemOffset,
                            applyPermutation(shmemStrides, sharedOrder));
    auto vecAddr = gep(ptrTy, elemLlvmTy, shmemBase, shmemOffset);
    vecAddr.setInbounds(true);

    perVectorCallback(vecTy, vecAddr);
  }
  return true;
}

void storeDistributedToShared(MemDescType dstTy, RankedTensorType srcTy,
                              Type elemLlvmTy, ArrayRef<Value> srcVals,
                              Value smemBase, ArrayRef<Value> dstStrides,
                              Location loc, RewriterBase &rewriter,
                              const TargetInfoBase &target, bool crossGrain) {
  bool success;
  std::function<void(VectorType, Value /*shmemAddr*/)> perVectorCallback;
  if (!crossGrain) {
    perVectorCallback = [&](VectorType vecTy, Value vecAddr) {
          ArrayRef<Value> vals = srcVals.take_front(vecTy.getNumElements());
          srcVals = srcVals.drop_front(vecTy.getNumElements());

          Value vec = undef(vecTy);
          for (int i = 0; i < vals.size(); i++) {
            vec = insert_element(vec, vals[i], i32_val(i));
          }
          store(vec, vecAddr)
              .setAlignment(vecTy.getNumElements() *
                            elemLlvmTy.getIntOrFloatBitWidth() / 8);
        };
  } else {
    LDBG("crossGrain: " << crossGrain);
    auto blockedEncoding = dyn_cast<BlockedEncodingAttr>(srcTy.getEncoding());
    auto sizePerThread = blockedEncoding.getSizePerThread();
    auto order = blockedEncoding.getOrder();
    unsigned int numElementsPerIter = product<unsigned>(sizePerThread);
    unsigned int val_counter = 0;
    LDBG("sizePerThread: [" << sizePerThread[0] << ", " << sizePerThread[1] << "]");
    LDBG("order: [" << order[0] << ", " << order[1] << "]");
    unsigned int innerVectorization = sizePerThread[order[0]];
    perVectorCallback = [&](VectorType vecTy, Value vecAddr) {
          Value vec = undef(vecTy);
          for (int i = 0; i < vecTy.getNumElements(); i++) {
              auto idx = val_counter % innerVectorization +
                  val_counter / innerVectorization * numElementsPerIter +
                  i*innerVectorization;
              vec = insert_element(vec, srcVals[idx], i32_val(i));
          }
          val_counter++;
          store(vec, vecAddr)
              .setAlignment(vecTy.getNumElements() *
                            elemLlvmTy.getIntOrFloatBitWidth() / 8);
        };
  }
  success = mlir::LLVM::AMD::emitTransferBetweenRegistersAndShared(
        srcTy, dstTy, elemLlvmTy, /*maxVecElems=*/std::nullopt, smemBase,
        dstStrides, loc, rewriter, target, crossGrain, perVectorCallback);
  if (!success)
    llvm::report_fatal_error("Failed to emit transfer from register to shared");
}
} // namespace mlir::LLVM::AMD
