#include "TargetInfo.h"
#include "Dialect/NVGPU/IR/Dialect.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierMbarAllocator.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir;

using ::mlir::LLVM::linearize;
namespace {
// declare vprintf(i8*, i8*) as external function
LLVM::LLVMFuncOp getVprintfDeclaration(RewriterBase &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("vprintf");
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *context = rewriter.getContext();

  SmallVector<Type> argsType{ptr_ty(context), ptr_ty(context)};
  auto funcType = LLVM::LLVMFunctionType::get(i32_ty, argsType);

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  return LLVM::LLVMFuncOp::create(rewriter, UnknownLoc::get(context), funcName,
                                  funcType);
}

// extend integer to int32, extend float to float64
// this comes from vprintf alignment requirements.
std::pair<Type, Value> printfPromoteValue(RewriterBase &rewriter, Value value,
                                          bool isSigned) {
  auto *context = rewriter.getContext();
  auto type = value.getType();
  Value newOp = value;
  Type newType = type;
  auto loc = UnknownLoc::get(context);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
    newType = i32_ty;
    if (isSigned) {
      newOp = b.sext(newType, value);
    } else {
      newOp = b.zext(newType, value);
    }
  } else if (type.isBF16() || type.isF16() || type.isF32()) {
    newType = f64_ty;
    newOp = b.fpext(newType, value);
  }

  return {newType, newOp};
}

LLVM::LLVMFuncOp getAssertfailDeclaration(RewriterBase &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("__assertfail");
  {
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);
  }
  // void __assert_fail(const char * assertion, const char * file, unsigned
  // int line, const char * function);
  auto *ctx = rewriter.getContext();
  SmallVector<Type> argsType{ptr_ty(ctx), ptr_ty(ctx), i32_ty, ptr_ty(ctx),
                             rewriter.getIntegerType(sizeof(size_t) * 8)};
  auto funcType = LLVM::LLVMFunctionType::get(void_ty(ctx), argsType);
  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  auto funcOp = LLVM::LLVMFuncOp::create(rewriter, UnknownLoc::get(ctx),
                                         funcName, funcType);

  funcOp.setPassthroughAttr(
      ArrayAttr::get(ctx, StringAttr::get(ctx, "noreturn")));
  return funcOp;
}
} // namespace

namespace mlir::triton::NVIDIA {

void registerTargetInfo() {
  TargetInfoBase::registerFactory("cuda", [](ModuleOp moduleOp) {
    return std::make_unique<TargetInfo>(getNVIDIAComputeCapability(moduleOp));
  });
}

// Check if the reduction can use a redux op and return the kind.
static std::optional<NVVM::ReductionKind>
matchReduxKind(triton::ReduceOp op, int computeCapability,
               bool &useNanQualifier) {
  useNanQualifier = false;
  if (computeCapability < 80)
    return std::nullopt;
  Operation *reduceOp = op.getSingleCombiner();
  if (!reduceOp)
    return std::nullopt;
  if (computeCapability / 10 == 10 && reduceOp->getResultTypes()[0].isF32()) {
    if (isa<arith::MinimumFOp, arith::MaximumFOp>(reduceOp))
      useNanQualifier = true;
    if (isa<arith::MaxNumFOp, arith::MaximumFOp>(reduceOp))
      return NVVM::ReductionKind::FMAX;
    if (isa<arith::MinNumFOp, arith::MinimumFOp>(reduceOp))
      return NVVM::ReductionKind::FMIN;
  }
  auto intType = dyn_cast<IntegerType>(reduceOp->getResultTypes()[0]);
  if (!intType)
    return std::nullopt;
  if (intType.getWidth() > 32 &&
      !(intType.getWidth() == 64 &&
        isa<arith::AddIOp, arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp,
            arith::MaxUIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp>(
            reduceOp)))
    return std::nullopt;
  if (isa<arith::AddIOp>(reduceOp))
    return NVVM::ReductionKind::ADD;
  if (isa<arith::AndIOp>(reduceOp))
    return NVVM::ReductionKind::AND;
  if (isa<arith::OrIOp>(reduceOp))
    return NVVM::ReductionKind::OR;
  if (isa<arith::XOrIOp>(reduceOp))
    return NVVM::ReductionKind::XOR;
  if (isa<arith::MinSIOp>(reduceOp))
    return NVVM::ReductionKind::MIN;
  if (isa<arith::MinUIOp>(reduceOp))
    return NVVM::ReductionKind::UMIN;
  if (isa<arith::MaxSIOp>(reduceOp))
    return NVVM::ReductionKind::MAX;
  if (isa<arith::MaxUIOp>(reduceOp))
    return NVVM::ReductionKind::UMAX;
  return std::nullopt;
}

static Value createReduxIdentity(TritonLLVMOpBuilder &b,
                                 NVVM::ReductionKind kind,
                                 bool useNanQualifier) {
  switch (kind) {
  case NVVM::ReductionKind::MIN:
    return b.i32_val(INT32_MAX);
  case NVVM::ReductionKind::MAX:
    return b.i32_val(INT32_MIN);
  case NVVM::ReductionKind::UMIN:
  case NVVM::ReductionKind::AND:
    return b.i32_val(UINT32_MAX);
  case NVVM::ReductionKind::UMAX:
  case NVVM::ReductionKind::ADD:
  case NVVM::ReductionKind::OR:
  case NVVM::ReductionKind::XOR:
    return b.i32_val(0);
  case NVVM::ReductionKind::FMIN:
  case NVVM::ReductionKind::FMAX:
    // NaN-ignoring min/max must return NaN for an all-NaN group. An
    // infinity identity would incorrectly turn that result into infinity.
    if (!useNanQualifier)
      return b.f32_val(std::numeric_limits<float>::quiet_NaN());
    return b.f32_val(kind == NVVM::ReductionKind::FMIN
                         ? std::numeric_limits<float>::infinity()
                         : -std::numeric_limits<float>::infinity());
  }
  llvm_unreachable("invalid redux kind");
}

bool TargetInfo::supportMaximumMinimum() const {
  return targetFeatures.supportMaximumMinimum();
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  if (triton::gpu::lookupNumCTAs(&rewriter.getInsertionBlock()->front()) == 1)
    return arith::ConstantIntOp::create(rewriter, loc, 0, 32);

  return triton::nvgpu::ClusterCTAIdOp::create(rewriter, loc,
                                               rewriter.getI32Type());
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value threadMask = b.int_val(type.getIntOrFloatBitWidth(), -1);
  return NVVM::VoteSyncOp::create(rewriter, loc, type, threadMask, cmp,
                                  NVVM::VoteSyncKind::ballot);
}

Value TargetInfo::getGlobalTimer(RewriterBase &rewriter, Location loc) const {
  return LLVM::createLLVMIntrinsicCallOp(
             rewriter, loc, "llvm.nvvm.read.ptx.sreg.globaltimer", i64_ty, {})
      .getResult(0);
}

StringRef TargetInfo::getAtomicSyncScope(MemSyncScope scope) const {
  switch (scope) {
  case MemSyncScope::CTA:
    return "block";
  case MemSyncScope::GPU:
    return "device";
  case MemSyncScope::SYSTEM:
    return {};
  }
  llvm_unreachable("unknown memory synchronization scope");
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         triton::gpu::AddrSpace targets) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  b.barrier(targets);
}

void TargetInfo::clusterBarrier(Location loc, RewriterBase &rewriter,
                                Operation *sourceOp) const {
  auto barrier = triton::nvidia_gpu::ClusterBarrierOp::create(rewriter, loc);
  triton::nvidia_gpu::copyClusterBarrierMbarOffset(sourceOp, barrier);
}

void TargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  NVVM::SyncWarpOp::create(rewriter, loc, b.i32_val(0xffffffff));
}

static bool isConstantTruePred(Value pred) {
  if (auto constOp = pred.getDefiningOp<LLVM::ConstantOp>()) {
    return cast<IntegerAttr>(constOp.getValue()).getInt() == -1;
  }
  return false;
}

static Value mapa(RewriterBase &rewriter, Location loc, Value ptr,
                  Value ctaid) {
  auto clusterPtrTy = ptr_ty(rewriter.getContext(), /*addrspace=*/7);
  // Address translation is speculatable; the memory access keeps its predicate.
  return NVVM::MapaOp::create(rewriter, loc, clusterPtrTy, ptr, ctaid);
}

Value TargetInfo::mapDShared(RewriterBase &rewriter, Location loc, Value ptr,
                             Value ctaId, Value /*pred*/) const {
  return ctaId ? mapa(rewriter, loc, ptr, ctaId) : ptr;
}

static std::string getConstraintForBitwidth(unsigned bitwidth) {
  switch (bitwidth) {
  case 8:
  case 16:
    return "h";
  case 32:
    return "r";
  case 64:
    return "l";
  default:
    llvm_unreachable("unsupported bitwidth");
  }
}

unsigned TargetInfo::getMaxAtomicLoadStoreVectorSize(unsigned bitWidth) const {
  return 128 / bitWidth;
}

Value TargetInfo::loadRelaxed(RewriterBase &rewriter, Location loc, Value ptr,
                              Type valueTy, Value pred,
                              MemSyncScope scope) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto vectorTy = dyn_cast<VectorType>(valueTy);
  Type elemTy = vectorTy ? vectorTy.getElementType() : valueTy;
  unsigned vec = vectorTy ? vectorTy.getNumElements() : 1;
  unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
  assert(vec <= getMaxAtomicLoadStoreVectorSize(bitWidth));
  unsigned packedBitWidth = std::min(32u, bitWidth * vec);
  if (packedBitWidth > bitWidth) {
    unsigned packedVec = bitWidth * vec / packedBitWidth;
    Type wordTy = int_ty(packedBitWidth);
    Type packedTy = packedVec == 1 ? wordTy : vec_ty(wordTy, packedVec);
    Value loaded = loadRelaxed(rewriter, loc, ptr, packedTy, pred, scope);
    return b.bitcast(loaded, valueTy);
  }
  PTXBuilder builder;
  auto *outputs = builder.newListOperand();
  for (unsigned i = 0; i < vec; ++i)
    outputs->listAppend(builder.newOperand(
        "=" + getConstraintForBitwidth(bitWidth), /*init=*/bool(pred)));
  auto &load = builder.create("ld")
                   ->o("relaxed")
                   .o(stringifyMemSyncScope(scope).str())
                   .o("global")
                   .v(vec)
                   .b(bitWidth);
  load(vec == 1 ? outputs->listGet(0) : outputs,
       builder.newAddrOperand(ptr, "l"))
      .maybePredicate(pred);
  // PTX has no 8-bit register constraint. Load into 16-bit registers and keep
  // the low bytes, then reinterpret floating-point values without conversion.
  Type regTy = int_ty(std::max(16u, bitWidth));
  Type retTy = vec == 1
                   ? regTy
                   : LLVM::LLVMStructType::getLiteral(
                         rewriter.getContext(), SmallVector<Type>(vec, regTy));
  Value loaded = builder.launch(rewriter, loc, retTy, /*hasSideEffect=*/true);
  auto results = unpackLLElements(loc, loaded, rewriter);
  for (Value &result : results) {
    if (bitWidth == 8)
      result = b.trunc(i8_ty, result);
    if (result.getType() != elemTy)
      result = b.bitcast(result, elemTy);
  }
  return vectorTy ? packLLVector(loc, results, rewriter) : results.front();
}

void TargetInfo::storeRelaxed(RewriterBase &rewriter, Location loc, Value ptr,
                              Value value, Value pred,
                              MemSyncScope scope) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto vectorTy = dyn_cast<VectorType>(value.getType());
  Type elemTy = vectorTy ? vectorTy.getElementType() : value.getType();
  unsigned vec = vectorTy ? vectorTy.getNumElements() : 1;
  unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
  assert(vec <= getMaxAtomicLoadStoreVectorSize(bitWidth));
  unsigned packedBitWidth = std::min(32u, bitWidth * vec);
  if (packedBitWidth > bitWidth) {
    unsigned packedVec = bitWidth * vec / packedBitWidth;
    Type wordTy = int_ty(packedBitWidth);
    Type packedTy = packedVec == 1 ? wordTy : vec_ty(wordTy, packedVec);
    storeRelaxed(rewriter, loc, ptr, b.bitcast(value, packedTy), pred, scope);
    return;
  }
  PTXBuilder builder;
  auto *addr = builder.newAddrOperand(ptr, "l");
  auto *inputs = builder.newListOperand();
  for (Value element : unpackLLVector(loc, value, rewriter)) {
    if (!elemTy.isInteger())
      element = b.bitcast(element, int_ty(bitWidth));
    if (bitWidth == 8)
      element = b.zext(i16_ty, element);
    inputs->listAppend(
        builder.newOperand(element, getConstraintForBitwidth(bitWidth)));
  }
  auto &store = builder.create("st")
                    ->o("relaxed")
                    .o(stringifyMemSyncScope(scope).str())
                    .o("global")
                    .v(vec)
                    .b(bitWidth);
  store(addr, vec == 1 ? inputs->listGet(0) : inputs).maybePredicate(pred);
  builder.launch(rewriter, loc, void_ty(rewriter.getContext()),
                 /*hasSideEffect=*/true);
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              Value ctaId, Value val, Value pred) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = rewriter.getContext();
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");

  if (!isa<VectorType>(val.getType())) {
    storeDShared(rewriter, loc, ptr, ctaId, packLLVector(loc, {val}, rewriter),
                 pred);
    return;
  }

  auto vecTy = cast<VectorType>(val.getType());
  Type elemTy = vecTy.getElementType();
  unsigned vec = vecTy.getNumElements();
  unsigned elemBitwidth = getIntOrFloatOrPtrBitWidth(elemTy);
  assert(llvm::isPowerOf2_32(vec));

  if (elemBitwidth < 8) {
    assert(vec == 1 &&
           "don't know how to load/store vectors of sub-byte elems");
    SmallVector<Value> vals = unpackLLVector(loc, val, rewriter);
    for (Value &v : vals) {
      v = b.zext(int_ty(8), b.bitcast(v, int_ty(elemBitwidth)));
    }
    storeDShared(rewriter, loc, ptr, ctaId, packLLVector(loc, vals, rewriter),
                 pred);
    return;
  }

  if (!elemTy.isInteger()) {
    SmallVector<Value> vals = unpackLLVector(loc, val, rewriter);
    for (Value &v : vals) {
      if (isa<LLVM::LLVMPointerType>(v.getType())) {
        v = b.ptrtoint(int_ty(elemBitwidth), v);
      } else {
        v = b.bitcast(v, int_ty(elemBitwidth));
      }
    }
    storeDShared(rewriter, loc, ptr, ctaId, packLLVector(loc, vals, rewriter),
                 pred);
    return;
  }

  // load/store ops only support v2 and v4.  If the vector width is larger than
  // 4, we have two strategies for dealing with it.
  //  1. If the element type is smaller than b32, store b32's instead.
  //  2. Otherwise, split the store into multiple stores.
  if (vec > 4 && elemBitwidth < 32) {
    assert(llvm::isPowerOf2_32(vec));
    int elemsPerPack = 32 / elemBitwidth;
    SmallVector<Value> oldVals = unpackLLVector(loc, val, rewriter);

    SmallVector<Value> newVals;
    for (int i = 0; i < vec / elemsPerPack; i++) {
      Value v = packLLVector(
          loc, ArrayRef(oldVals).slice(i * elemsPerPack, elemsPerPack),
          rewriter);
      newVals.push_back(b.bitcast(v, i32_ty));
    }
    storeDShared(rewriter, loc, ptr, ctaId,
                 packLLVector(loc, newVals, rewriter), pred);
    return;
  }

  if (vec * elemBitwidth > 128) {
    assert(llvm::isPowerOf2_32(vec));
    assert(elemBitwidth == 32 || elemBitwidth == 64);
    int maxVec = 128 / elemBitwidth;

    SmallVector<Value> vals = unpackLLVector(loc, val, rewriter);
    for (int i = 0; i < vec / maxVec; i++) {
      auto newPtr = b.gep(ptr.getType(), elemTy, ptr, b.i32_val(i * maxVec),
                          LLVM::GEPNoWrapFlags::inbounds);
      storeDShared(
          rewriter, loc, newPtr, ctaId,
          packLLVector(loc, ArrayRef(vals).slice(i * maxVec, maxVec), rewriter),
          pred);
    }
    return;
  }

  // At this point we're committed to doing the store!
  assert(elemBitwidth >= 8);
  assert(elemTy.isInteger());
  assert(1 <= vec && vec <= 4);
  assert(vec * elemBitwidth <= 128);

  // Get pointer to remote shared memory if needed.
  if (ctaId) {
    ptr = mapa(rewriter, loc, ptr, ctaId);
  }

  PTXBuilder builder;
  auto st = builder.create("st")
                ->o(ctaId ? "shared::cluster" : "shared::cta")
                .v(vec, /*predicate=*/vec > 1)
                .b(elemBitwidth);
  auto *ptrOpr = builder.newAddrOperand(ptr, "r");

  if (isConstantTruePred(pred)) {
    b.store(val, ptr, /*align=*/vec * elemBitwidth / 8);
  } else {
    PTXBuilder::Operand *valOpr;
    std::string constraint = getConstraintForBitwidth(elemBitwidth);
    if (vec > 1) {
      SmallVector<std::pair<Value, std::string>> vecVals;
      for (int i = 0; i < vec; i++) {
        vecVals.push_back({b.extract_element(val, b.i32_val(i)), constraint});
      }
      valOpr = builder.newListOperand(vecVals);
    } else {
      valOpr = builder.newOperand(val, constraint);
    }
    st(ptrOpr, valOpr).predicate(pred, "b");
    builder.launch(rewriter, loc, void_ty(ctx));
  }
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              Value ctaId, Type loadTy, Value pred,
                              Operation *) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = rewriter.getContext();
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");

  if (!isa<VectorType>(loadTy)) {
    SmallVector<Value> values = unpackLLVector(
        loc, loadDShared(rewriter, loc, ptr, ctaId, vec_ty(loadTy, 1), pred),
        rewriter);
    assert(values.size() == 1);
    return values[0];
  }

  auto vecTy = cast<VectorType>(loadTy);
  Type elemTy = vecTy.getElementType();
  unsigned vec = vecTy.getNumElements();
  unsigned elemBitwidth = getIntOrFloatOrPtrBitWidth(elemTy);
  assert(llvm::isPowerOf2_32(vec));

  if (elemBitwidth < 8) {
    assert(vec == 1 &&
           "don't know how to load/store vectors of sub-byte elems");
    SmallVector<Value> vals = unpackLLVector(
        loc, loadDShared(rewriter, loc, ptr, ctaId, int_ty(8), pred), rewriter);
    assert(vals.size() == 1);
    return b.bitcast(b.trunc(int_ty(elemBitwidth), vals[0]), elemTy);
  }

  // We only know how to load integers.
  if (!elemTy.isInteger()) {
    Type newLoadTy = vec_ty(int_ty(elemBitwidth), vec);
    SmallVector<Value> vals = unpackLLVector(
        loc, loadDShared(rewriter, loc, ptr, ctaId, newLoadTy, pred), rewriter);
    for (Value &v : vals) {
      if (isa<LLVM::LLVMPointerType>(elemTy))
        v = b.inttoptr(elemTy, v);
      else
        v = b.bitcast(v, elemTy);
    }
    return packLLVector(loc, vals, rewriter);
  }

  // load/store ops only support v2 and v4.  If the vector width is larger than
  // 4, we have two strategies for dealing with it.
  //  1. If the element type is smaller than b32, load b32's instead.
  //  2. Otherwise, split the load into multiple loads.
  if (vec > 4 && elemBitwidth < 32) {
    int newVec = vec / (32 / elemBitwidth);
    auto newVecTy = vec_ty(i32_ty, newVec);
    auto res = loadDShared(rewriter, loc, ptr, ctaId, newVecTy, pred);

    // Unpack the b32's into the original vector type.
    SmallVector<Value> vals;
    for (Value v : unpackLLVector(loc, res, rewriter)) {
      Value vv = b.bitcast(v, vec_ty(elemTy, 32 / elemBitwidth));
      for (Value vvv : unpackLLVector(loc, vv, rewriter)) {
        vals.push_back(vvv);
      }
    }
    return packLLVector(loc, vals, rewriter);
  }

  if (vec * elemBitwidth > 128) {
    assert(elemBitwidth == 32 || elemBitwidth == 64);
    assert(llvm::isPowerOf2_32(vec));
    int maxVec = 128 / elemBitwidth;

    SmallVector<Value> vals;
    for (int i = 0; i < vec / maxVec; i++) {
      auto newPtr = b.gep(ptr.getType(), elemTy, ptr, b.i32_val(i * maxVec),
                          LLVM::GEPNoWrapFlags::inbounds);
      auto newVal = loadDShared(rewriter, loc, newPtr, ctaId,
                                vec_ty(elemTy, maxVec), pred);
      for (Value v : unpackLLVector(loc, newVal, rewriter)) {
        vals.push_back(v);
      }
    }
    return packLLVector(loc, vals, rewriter);
  }

  // At this point we're committed to actually do the load!
  assert(elemBitwidth >= 8);
  assert(elemTy.isInteger());
  assert(1 <= vec && vec <= 4);
  assert(vec * elemBitwidth <= 128);

  // Get pointer to remote shared memory if needed.
  if (ctaId) {
    ptr = mapa(rewriter, loc, ptr, ctaId);
  }

  PTXBuilder builder;
  auto ld = builder.create("ld")
                ->o(ctaId ? "shared::cluster" : "shared::cta")
                .v(vec, /*predicate=*/vec > 1)
                .b(elemBitwidth);

  Value load;
  if (isConstantTruePred(pred)) {
    Type resultTy = vec == 1 ? Type(int_ty(elemBitwidth))
                             : Type(vec_ty(int_ty(elemBitwidth), vec));
    load = b.load(resultTy, ptr, /*align=*/vec * elemBitwidth / 8);
    if (vec > 1) {
      Type structTy = struct_ty(SmallVector<Type>(vec, int_ty(elemBitwidth)));
      Value structValue = b.undef(structTy);
      for (int i = 0; i < vec; i++) {
        structValue = b.insert_val(structTy, structValue,
                                   b.extract_element(load, b.i32_val(i)), i);
      }
      load = structValue;
    }
  } else {
    std::string elemConstraint = "=" + getConstraintForBitwidth(elemBitwidth);
    auto *outOpr = vec == 1 ? builder.newOperand(elemConstraint)
                            : builder.newListOperand(vec, elemConstraint);
    ld(outOpr, builder.newAddrOperand(ptr, "r")).predicate(pred, "b");

    Type resultTy =
        vec == 1
            ? Type(int_ty(elemBitwidth))
            : Type(struct_ty(SmallVector<Type>(vec, int_ty(elemBitwidth))));
    load = builder.launch(rewriter, loc, resultTy, /*hasSideEffects=*/true);
  }
  SmallVector<Value> resultVals = unpackLLElements(loc, load, rewriter);
  return packLLVector(loc, resultVals, rewriter);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::NVIDIA::shuffleXor(loc, rewriter, val, i);
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  return LLVM::NVIDIA::shuffleUp(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::NVIDIA::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  return LLVM::NVIDIA::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  return LLVM::NVIDIA::permute(loc, rewriter, a, b, selector);
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  return LLVM::NVIDIA::llGetPid(loc, rewriter, moduleOp, axis);
}
bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned reduceLaneIdMask,
                            unsigned broadcastLaneIdMask) const {

  constexpr unsigned kWarpSize = 32;
  unsigned fullMask = kWarpSize - 1;
  bool partialWarp = reduceLaneIdMask != fullMask;
  unsigned groupMask = fullMask & ~(reduceLaneIdMask | broadcastLaneIdMask);
  bool partitioned = groupMask != 0;
  // Use at most two converged full-warp reductions. Broadcast lanes
  // share a result, so only lane bits identifying distinct groups count.
  if (llvm::popcount(groupMask) > 1)
    return false;
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  bool useNanQualifier = false;
  if (auto kind = matchReduxKind(op, targetFeatures.getComputeCapability(),
                                 useNanQualifier)) {
    assert(acc.size() == 1);
    if (partialWarp) {
      if (acc[0].getType().getIntOrFloatBitWidth() > 32)
        return false;
      // Min/max use the faster CREDUX instructions on SM100+. The minimum
      // group size also depends on whether we need one redux or two.
      bool isMinMax = *kind == NVVM::ReductionKind::MIN ||
                      *kind == NVVM::ReductionKind::MAX ||
                      *kind == NVVM::ReductionKind::UMIN ||
                      *kind == NVVM::ReductionKind::UMAX ||
                      *kind == NVVM::ReductionKind::FMIN ||
                      *kind == NVVM::ReductionKind::FMAX;
      bool useFastMinMax = isMinMax && getComputeCapability() >= 100;
      // Heuristic thresholds based on latency/throughput benchmarks:
      // https://github.com/triton-lang/triton/pull/11823
      unsigned minLanes =
          useFastMinMax ? (partitioned ? 8 : 2) : (partitioned ? 16 : 4);
      unsigned numLanes = 1u << llvm::popcount(reduceLaneIdMask);
      if (numLanes < minLanes)
        return false;
    }
    Value mask = b.i32_val(0xFFFFFFFF);
    bool maskBroadcast =
        broadcastLaneIdMask && (*kind == NVVM::ReductionKind::ADD ||
                                *kind == NVVM::ReductionKind::XOR);
    Value identity, firstGroup, uniqueLane;
    if (partitioned || maskBroadcast) {
      identity = createReduxIdentity(b, *kind, useNanQualifier);
      Value laneId = getLaneId(rewriter, loc);
      if (partitioned) {
        Value group = b.and_(laneId, b.i32_val(groupMask));
        firstGroup = b.icmp_eq(group, b.i32_val(0));
      }
      // Min/max, AND, and OR are idempotent. Add and XOR must count each
      // broadcast value just once, even though all lanes execute the redux.
      if (maskBroadcast) {
        Value duplicate = b.and_(laneId, b.i32_val(broadcastLaneIdMask));
        uniqueLane = b.icmp_eq(duplicate, b.i32_val(0));
      }
    }
    if (acc[0].getType().isInteger(64)) {
      Value high = b.trunc(i32_ty, b.lshr(acc[0], b.i64_val(32)));
      Value highResult = NVVM::ReduxOp::create(rewriter, loc, i32_ty, high,
                                               *kind, mask, false, false);
      Value low = b.trunc(i32_ty, acc[0]);
      if (*kind == NVVM::ReductionKind::ADD) {
        // Each 16-bit sum fits in 21 bits; i64 adds propagate their carries.
        Value lowResult = NVVM::ReduxOp::create(rewriter, loc, i32_ty,
                                                b.and_(low, b.i32_val(0xFFFF)),
                                                *kind, mask, false, false);
        Value middleResult = NVVM::ReduxOp::create(rewriter, loc, i32_ty,
                                                   b.lshr(low, b.i32_val(16)),
                                                   *kind, mask, false, false);
        acc[0] =
            b.add(b.add(b.zext(i64_ty, lowResult),
                        b.shl(b.zext(i64_ty, middleResult), b.i64_val(16))),
                  b.shl(b.zext(i64_ty, highResult), b.i64_val(32)));
        return true;
      }
      auto lowKind = *kind;
      if (*kind == NVVM::ReductionKind::MIN ||
          *kind == NVVM::ReductionKind::UMIN ||
          *kind == NVVM::ReductionKind::MAX ||
          *kind == NVVM::ReductionKind::UMAX) {
        bool isMin = *kind == NVVM::ReductionKind::MIN ||
                     *kind == NVVM::ReductionKind::UMIN;
        lowKind = isMin ? NVVM::ReductionKind::UMIN : NVVM::ReductionKind::UMAX;
        // Only matching high words can win; compare their low words unsigned.
        low = b.select(b.icmp_eq(high, highResult), low,
                       b.i32_val(isMin ? 0xFFFFFFFF : 0));
      }
      Value lowResult = NVVM::ReduxOp::create(rewriter, loc, i32_ty, low,
                                              lowKind, mask, false, false);
      acc[0] = b.or_(b.shl(b.zext(i64_ty, highResult), b.i64_val(32)),
                     b.zext(i64_ty, lowResult));
      return true;
    }
    for (unsigned i = 0; i < acc.size(); ++i) {
      unsigned bitwidth = acc[i].getType().getIntOrFloatBitWidth();
      if (acc[i].getType().isInteger()) {
        if (bitwidth < 32) {
          if (*kind == NVVM::ReductionKind::MIN ||
              *kind == NVVM::ReductionKind::MAX)
            acc[i] = b.sext(i32_ty, acc[i]);
          else
            acc[i] = b.zext(i32_ty, acc[i]);
        }
      }
      auto redux = [&](Value value) -> Value {
        return NVVM::ReduxOp::create(rewriter, loc, value.getType(), value,
                                     *kind, mask, /*abs=*/false,
                                     /*nan=*/useNanQualifier);
      };
      if (maskBroadcast)
        acc[i] = b.select(uniqueLane, acc[i], identity);
      if (partitioned) {
        Value first = redux(b.select(firstGroup, acc[i], identity));
        Value second = redux(b.select(firstGroup, identity, acc[i]));
        acc[i] = b.select(firstGroup, first, second);
      } else {
        acc[i] = redux(acc[i]);
      }
      if (acc[i].getType().isInteger()) {
        if (bitwidth < 32)
          acc[i] = b.trunc(int_ty(bitwidth), acc[i]);
      }
    }
    return true;
  }
  return false;
}

unsigned TargetInfo::getReductionTreeArity(Operation *combinerOp) const {
  int computeCapability = getComputeCapability();
  Type resultType = combinerOp->getResult(0).getType();
  // Consumer Blackwell lacks ternary forms; Thor only supports FP32.
  if (computeCapability < 90 || computeCapability / 10 == 12 ||
      (computeCapability / 10 == 11 && !resultType.isF32()))
    return 2;

  if (computeCapability >= 100 && getPtxVersion() >= 88 &&
      isa<arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp,
          arith::MinNumFOp>(combinerOp) &&
      resultType.isF32())
    return 3;

  if (resultType.isInteger(32) &&
      isa<arith::MinSIOp, arith::MaxSIOp, arith::MinUIOp, arith::MaxUIOp>(
          combinerOp))
    return 3;

  auto vectorType = dyn_cast<VectorType>(resultType);
  if (!vectorType || vectorType.getNumElements() != 2)
    return 2;

  Type elementType = vectorType.getElementType();
  if (elementType.isInteger(16) &&
      isa<LLVM::SMinOp, LLVM::SMaxOp, LLVM::UMinOp, LLVM::UMaxOp>(combinerOp))
    return 3;
  if ((elementType.isF16() || elementType.isBF16()) &&
      isa<LLVM::MinNumOp, LLVM::MaxNumOp, LLVM::MinimumOp, LLVM::MaximumOp>(
          combinerOp))
    return 3;

  return 2;
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int /*formatStrByteCount*/, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  auto *ctx = rewriter.getContext();
  Type ptr = ptr_ty(ctx);
  auto funcOp = getVprintfDeclaration(rewriter);
  auto loc = UnknownLoc::get(ctx);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Value one = b.i32_val(1);
  Value zero = b.i32_val(0);

  Value bufferPtr = b.null(ptr);

  SmallVector<Value, 16> newArgs;
  if (args.size() >= 1) {
    SmallVector<Type> argTypes;
    for (auto [i, arg] : llvm::enumerate(args)) {
      Type newType;
      Value newArg;
      std::tie(newType, newArg) = printfPromoteValue(
          rewriter, arg, isSigned.empty() ? true : isSigned[i]);
      argTypes.push_back(newType);
      newArgs.push_back(newArg);
    }

    Type structTy = LLVM::LLVMStructType::getLiteral(ctx, argTypes);
    auto allocated =
        LLVM::AllocaOp::create(rewriter, loc, ptr_ty(ctx), structTy, one,
                               /*alignment=*/0);

    for (const auto &entry : llvm::enumerate(newArgs)) {
      auto index = b.i32_val(entry.index());
      auto fieldPtr =
          b.gep(ptr_ty(ctx), structTy, allocated, ArrayRef<Value>{zero, index});
      b.store(entry.value(), fieldPtr);
    }
    bufferPtr = b.bitcast(allocated, ptr);
  }

  SmallVector<Value> operands{formatStrStart, bufferPtr};
  b.call(funcOp, operands);
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue =
      LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()), rewriter,
                              "printfFormat_", msgNewline);
  printf(rewriter, msgValue, msgNewline.size_in_bytes(), args, isSigned);
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto funcOp = getAssertfailDeclaration(rewriter);
  llvm::SmallString<64> messageString(message), fileString(file),
      funcString(func);
  messageString.push_back('\0');
  fileString.push_back('\0');
  funcString.push_back('\0');
  Value messageStringVal =
      LLVM::addStringToModule(loc, rewriter, "assertMessage_", messageString);
  Value fileStringVal =
      LLVM::addStringToModule(loc, rewriter, "assertFile_", fileString);
  Value funcStringVal =
      LLVM::addStringToModule(loc, rewriter, "assertFunc_", funcString);
  Value lineNumber = b.i32_val(line);
  Value charSize = b.int_val(sizeof(size_t) * 8, sizeof(char));
  SmallVector<Value> operands = {messageStringVal, fileStringVal, lineNumber,
                                 funcStringVal, charSize};
  b.call(funcOp, operands);
}

int TargetInfo::getSharedAddressSpace() const { return 3; }

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  int spaceId = 0;
  if (isa<triton::gpu::SharedMemorySpaceAttr,
          triton::nvidia_gpu::TensorMemorySpaceAttr>(addressSpace)) {
    spaceId = 3;
  } else {
    llvm::report_fatal_error(
        "Only support SharedMemorySpace, TensorMemorySpace for now");
  }
  return spaceId;
}

bool TargetInfo::supportVectorizedAtomics() const {
  return targetFeatures.getComputeCapability() >= 90 && ptxVersion >= 81;
}

} // namespace mlir::triton::NVIDIA
