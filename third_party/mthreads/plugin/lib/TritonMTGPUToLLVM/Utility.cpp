#include "Utility.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/MTGPUDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

using mlir::triton::gpu::appendOrGetExternFuncOp;
using mlir::triton::gpu::getFunctionType;

namespace {
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
} // anonymous namespace

namespace mlir {
namespace LLVM {
namespace MUSA {

static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value value,
                           Value i, const MTGPU::ShflKind &mode, int widthInt) {
  auto valueTy = value.getType();
  unsigned bits = valueTy.getIntOrFloatBitWidth();

  auto int8Ty = rewriter.getI8Type();
  auto int32Ty = rewriter.getI32Type();
  auto nullPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 5);
  Value zero = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
                                                 rewriter.getI32IntegerAttr(0));
  Value one = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
                                                rewriter.getI32IntegerAttr(1));
  Value seven = rewriter.create<LLVM::ConstantOp>(
      loc, int32Ty, rewriter.getI32IntegerAttr(7));
  Value num_128 = rewriter.create<LLVM::ConstantOp>(
      loc, int32Ty, rewriter.getI32IntegerAttr(128));
  Value width = rewriter.create<LLVM::ConstantOp>(
      loc, int32Ty, rewriter.getI32IntegerAttr(widthInt));
  Value nullPtr = rewriter.create<LLVM::ZeroOp>(loc, nullPtrTy);
  Value offset = i;

  Value maskAndClamp;

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = bitcast(value, vecTy);
    Value val0 = extract_element(f32_ty, vec, i32_val(0));
    Value val1 = extract_element(f32_ty, vec, i32_val(1));
    val0 = shuffleCommon(loc, rewriter, val0, i, mode, widthInt);
    val1 = shuffleCommon(loc, rewriter, val1, i, mode, widthInt);
    vec = undef(vecTy);
    vec = insert_element(vecTy, vec, val0, i32_val(0));
    vec = insert_element(vecTy, vec, val1, i32_val(1));
    return bitcast(vec, value.getType());
  }
  if (valueTy != i32_ty) {
    value = bitcast(value, int_ty(bits));
    if (bits < 32)
      value = zext(i32_ty, value);
  }

  // maskAndClamp is set to 0 when in 'up' mode.
  if (mode == MTGPU::ShflKind::up) {
    maskAndClamp = zero;
  } else {
    Value Clamp = rewriter.create<LLVM::SubOp>(loc, int32Ty, width, one);
    Value SegMask = rewriter.create<LLVM::SubOp>(loc, int32Ty, num_128, width);
    SegMask = rewriter.create<LLVM::ShlOp>(loc, int32Ty, SegMask, seven);
    maskAndClamp = rewriter.create<LLVM::OrOp>(loc, int32Ty, SegMask, Clamp);
  }

  // shuffle argument pred is default nullptr if not given.
  Value result = rewriter.create<MTGPU::ShflOp>(loc, int32Ty, value, offset,
                                                maskAndClamp, mode, nullPtr);

  if (valueTy != i32_ty) {
    if (bits < 32)
      result = trunc(int_ty(bits), result);
    result = bitcast(result, valueTy);
  }

  return result;
}

Value MTGPU_shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i,
                       unsigned width) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), MTGPU::ShflKind::bfly,
                       width);
}

Value MTGPU_shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i,
                      unsigned width) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), MTGPU::ShflKind::up,
                       width);
}

Value MTGPU_shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i,
                       unsigned width) {
  return shuffleCommon(loc, rewriter, val, i32_val(i), MTGPU::ShflKind::idx,
                       width);
}

Value MTGPU_shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i,
                       unsigned width) {
  return shuffleCommon(loc, rewriter, val, i, MTGPU::ShflKind::idx, width);
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

Value llLoad(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
             Type elemTy, Value pred, Value falseVal) {
  Type funcType = getFunctionType(elemTy, ValueRange({ptr, pred, falseVal}));
  auto parent = ptr.getParentRegion()->getParentOfType<LLVM::LLVMFuncOp>();
  auto funcName = mangleFunc(mlir::LLVM::MUSA::Predicated_Load, funcType);
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, parent, funcName, funcType);
  auto loadVal =
      rewriter
          .create<LLVM::CallOp>(loc, funcOp, ValueRange({ptr, pred, falseVal}))
          .getResult();
  return loadVal;
}

void llStore(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
             Value val, Value pred) {
  auto ctx = ptr.getContext();
  Type funcType = getFunctionType(void_ty(ctx), ValueRange({ptr, val, pred}));
  auto parent = ptr.getParentRegion()->getParentOfType<LLVM::LLVMFuncOp>();
  auto funcName = mangleFunc(mlir::LLVM::MUSA::Predicated_Store, funcType);
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, parent, funcName, funcType);
  rewriter.create<LLVM::CallOp>(loc, funcOp, ValueRange({ptr, val, pred}));
}

} // namespace MUSA
} // namespace LLVM
} // namespace mlir
