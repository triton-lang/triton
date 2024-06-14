#include "Utility.h"
#include "Dialect/NVGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

namespace mlir {
namespace LLVM {
namespace NVIDIA {
using namespace mlir::triton;

static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value i, NVVM::ShflKind mode, Value clamp) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = bitcast(val, vecTy);
    Value val0 = extract_element(f32_ty, vec, i32_val(0));
    Value val1 = extract_element(f32_ty, vec, i32_val(1));
    val0 = shuffleCommon(loc, rewriter, val0, i, mode, clamp);
    val1 = shuffleCommon(loc, rewriter, val1, i, mode, clamp);
    vec = undef(vecTy);
    vec = insert_element(vecTy, vec, val0, i32_val(0));
    vec = insert_element(vecTy, vec, val1, i32_val(1));
    return bitcast(vec, val.getType());
  }
  Type type = val.getType();
  if (type != i32_ty) {
    val = bitcast(val, int_ty(bits));
    if (bits < 32)
      val = zext(i32_ty, val);
  }
  Value mask = i32_val(0xFFFFFFFF);
  Value result = rewriter.create<NVVM::ShflOp>(loc, i32_ty, mask, val, i, clamp,
                                               mode, UnitAttr());
  if (type != i32_ty) {
    if (bits < 32)
      result = trunc(int_ty(bits), result);
    result = bitcast(result, type);
  }
  return result;
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), NVVM::ShflKind::bfly,
                       i32_val(0x1f));
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), NVVM::ShflKind::up,
                       i32_val(0x0));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleIdx(loc, rewriter, val, i32_val(i));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i) {
  return shuffleCommon(loc, rewriter, val, i, NVVM::ShflKind::idx,
                       i32_val(0x1f));
}

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               int axis) {
  assert(axis >= 0);
  assert(axis < 3);
  assert(moduleOp);

  // It is not easy to get the compute capability here, so we use numCTAs to
  // decide the semantic of GetProgramIdOp. If numCTAs = 1, then
  // GetProgramIdOp is converted to "%ctaid", otherwise it is converted to
  // "%clusterid".
  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

  std::string sreg = numCTAs == 1 ? "%ctaid." : "%clusterid.";
  sreg.append(1, 'x' + axis); // 0 -> 'x', 1 -> 'y', 2 -> 'z'
  return getSRegValue(rewriter, loc, sreg);
}

Value getSRegValue(OpBuilder &b, Location loc, const std::string &sRegStr) {
  PTXBuilder builder;
  auto &mov = builder.create("mov")->o("u32");
  auto *destOpr = builder.newOperand("=r");
  auto *sRegOpr = builder.newConstantOperand(sRegStr);
  mov(destOpr, sRegOpr);
  Value val = builder.launch(b, loc, b.getIntegerType(32), false);
  return val;
}

Value permute(Location loc, RewriterBase &rewriter, Value a, Value b,
              Value mask) {
  PTXBuilder builder;
  auto &prmt = builder.create("prmt")->o("b32");
  auto *destOpr = builder.newOperand("=r");
  auto *aOperand = builder.newOperand(a, "r");
  auto *bOperand = builder.newOperand(b, "r");
  auto *maskOperand = builder.newOperand(mask, "r");
  prmt(destOpr, aOperand, bOperand, maskOperand);
  return builder.launch(rewriter, loc, rewriter.getIntegerType(32), false);
}

// A wrapper of LoadDSmemOp when vec = 1
// (1) Get bitwidth from elemTy
// (2) Create LoadDSmemOp
// (3) Bitcast result from dataTy (u16/u32/u64) back to elemTy
Value createLoadDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Type elemTy) {
  assert(isa<LLVMPointerType>(addr.getType()) && "addr must be a pointer type");
  auto ptrTy = cast<LLVMPointerType>(addr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  Value ret =
      rewriter.create<triton::nvgpu::LoadDSmemOp>(loc, addr, ctaId, bitwidth);
  return bitcast(ret, elemTy);
}

// A wrapper of LoadDSmemOp when vec > 1
// (1) Get bitwidth from elemTy
// (2) Create LoadDSmemOp and extract results from retStruct
// (3) Bitcast results from dataTy (u16/u32/u64) back to elemTy
SmallVector<Value> createLoadDSmem(Location loc, PatternRewriter &rewriter,
                                   Value addr, Value ctaId, unsigned vec,
                                   Type elemTy) {
  assert(isa<LLVMPointerType>(addr.getType()) && "addr must be a pointer type");
  auto ptrTy = cast<LLVMPointerType>(addr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  Value retStruct = rewriter.create<triton::nvgpu::LoadDSmemOp>(
      loc, addr, ctaId, bitwidth, vec);
  SmallVector<Value> retVals;
  for (unsigned i = 0; i < vec; ++i) {
    auto dataTy = rewriter.getIntegerType(bitwidth);
    Value data = extract_val(dataTy, retStruct, i);
    retVals.push_back(bitcast(data, elemTy));
  }
  return retVals;
}

// A wrapper of StoreDSmemOp when vec = 1
// (1) Get bitwidth from elemTy
// (2) Bitcast value from elemTy to dataTy (u16/u32/u64)
// (3) Create StoreDSmemOp
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Value value, Value pred) {
  assert(isa<LLVMPointerType>(addr.getType()) && "addr must be a pointer type");
  auto ptrTy = cast<LLVMPointerType>(addr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  unsigned bitwidth = value.getType().getIntOrFloatBitWidth();
  auto dataTy = rewriter.getIntegerType(bitwidth);
  Value data = bitcast(value, dataTy);
  rewriter.create<triton::nvgpu::StoreDSmemOp>(loc, addr, ctaId, data, pred);
}

// A wrapper of StoreDSmemOp when vec = 1 and pred = 1
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Value value) {
  Value pred = int_val(/*width=*/1, 1);
  createStoreDSmem(loc, rewriter, addr, ctaId, value, pred);
}

// A wrapper of StoreDSmemOp when vec > 1
// (1) Get bitwidth from elemTy
// (2) Bitcast values from elemTy to dataTy (u16/u32/u64)
// (3) Create StoreDSmemOp
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, ArrayRef<Value> values, Value pred) {
  assert(isa<LLVMPointerType>(addr.getType()) && "addr must be a pointer type");
  auto ptrTy = cast<LLVMPointerType>(addr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");
  unsigned bitwidth = 0;
  if (!values.empty()) {
    bitwidth = values.back().getType().getIntOrFloatBitWidth();
  }
  auto dataTy = rewriter.getIntegerType(bitwidth);
  SmallVector<Value> data;
  for (unsigned i = 0; i < values.size(); ++i)
    data.push_back(bitcast(values[i], dataTy));
  rewriter.create<triton::nvgpu::StoreDSmemOp>(loc, addr, ctaId, data, pred);
}

// A wrapper of StoreDSmemOp when vec > 1 and pred = 1
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, ArrayRef<Value> values) {
  Value pred = int_val(/*width=*/1, 1);
  createStoreDSmem(loc, rewriter, addr, ctaId, values, pred);
}

/// Create a predicate with just single active thread.
Value createElectPredicate(Location loc, PatternRewriter &rewriter) {
  PTXBuilder ptxBuilder;
  auto &elect = *ptxBuilder.create<>("elect.sync _|$0, 0xffffffff;");
  elect({ptxBuilder.newOperand("=b")}, /*onlyAttachMLIRArgs=*/true);
  // The instruction is technically not pure as it depends on simt control flow
  // however since we it outside of simt control flow in triton we can consider
  // it as pure to allow cse to work on it.
  return ptxBuilder.launch(rewriter, loc, i1_ty, /*hasSideEffect=*/false);
}

} // namespace NVIDIA
} // namespace LLVM
} // namespace mlir
