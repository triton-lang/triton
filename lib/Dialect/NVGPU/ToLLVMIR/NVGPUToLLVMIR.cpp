/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "triton/Dialect/NVGPU/ToLLVMIR/NVGPUToLLVMIR.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
static llvm::FunctionCallee
getExternalFuncOP(llvm::Module *module, llvm::StringRef funcName,
                  llvm::Type *retTy, ArrayRef<llvm::Type *> argTys = {}) {
  return module->getOrInsertFunction(
      funcName, llvm::FunctionType::get(retTy, argTys, false),
      llvm::AttributeList{});
}

llvm::Value *createExternalCall(llvm::IRBuilderBase &builder,
                                llvm::StringRef funcName,
                                ArrayRef<llvm::Value *> args = {},
                                ArrayRef<llvm::Type *> tys = {}) {
  auto *module = builder.GetInsertBlock()->getModule();
  auto *func = module->getFunction(funcName);

  if (func == nullptr) {
    llvm::SmallVector<llvm::Type *> argTys;
    for (auto *arg : args) {
      argTys.push_back(arg->getType());
    }

    llvm::Type *retTy;
    if (tys.empty())
      retTy = builder.getVoidTy();
    else
      retTy = tys[0];

    func = dyn_cast<llvm::Function>(
        getExternalFuncOP(module, funcName, retTy, argTys).getCallee());
  }

  return builder.CreateCall(func, args);
}

static std::string getTMALoadFuncName(bool tiled, bool mcast,
                                      uint32_t dimSize) {
  std::string funcName;
  llvm::raw_string_ostream os(funcName);
  os << "__nv_tma_load";
  if (tiled)
    os << "_tiled";
  else
    os << "_im2col";

  if (mcast)
    os << "_mcast";

  os << "_" << dimSize << "d";

  return funcName;
}

void createTMALoadIm2col(llvm::IRBuilderBase &builder, llvm::Value *dst,
                         llvm::Value *mbarrier, llvm::Value *tmaDesc,
                         llvm::Value *l2Desc, uint16_t mcastMask,
                         llvm::Value *im2colOffsets, llvm::Value *pred,
                         llvm::SmallVector<llvm::Value *> coords) {
  assert(coords.size() >= 3 && coords.size() <= 5 &&
         "invalid coords.size() for im2col");
  auto funcName = getTMALoadFuncName(false, mcastMask != 0, coords.size());
  llvm::Type *retTy = builder.getVoidTy();
  llvm::SmallVector<llvm::Value *> args;
  llvm::SmallVector<llvm::Type *> argTys;

  argTys.push_back(tmaDesc->getType());
  args.push_back(tmaDesc);

  argTys.push_back(dst->getType());
  args.push_back(dst);

  argTys.push_back(mbarrier->getType());
  args.push_back(mbarrier);
  for (auto *c : coords) {
    argTys.push_back(c->getType());
    args.push_back(c);
  }

  {
    auto offsetsType = dyn_cast<llvm::StructType>(im2colOffsets->getType());
    auto subTypes = offsetsType->elements();
    assert((coords.size() - subTypes.size() == 2) && "wrong imcolOffsets");
    unsigned idx = 0;
    for (auto subType : subTypes) {
      argTys.push_back(subType);
      args.push_back(builder.CreateExtractValue(im2colOffsets, {idx}));
      idx++;
    }
  }

  argTys.push_back(l2Desc->getType());
  args.push_back(l2Desc);

  if (mcastMask != 0) {
    argTys.push_back(builder.getInt16Ty());
    llvm::Value *mcastMask_ = builder.getInt16(mcastMask);
    args.push_back(mcastMask_);
  }

  argTys.push_back(pred->getType());
  args.push_back(pred);

  auto *module = builder.GetInsertBlock()->getModule();
  auto *func = dyn_cast<llvm::Function>(
      getExternalFuncOP(module, funcName, retTy, argTys).getCallee());
  builder.CreateCall(func, args);

  return;
}

void createStoreSharedCluster(llvm::IRBuilderBase &builder, llvm::Value *addr,
                              llvm::Value *ctaId,
                              llvm::SmallVector<llvm::Value *> values,
                              llvm::Value *pred, unsigned bitwidth,
                              unsigned vec) {
  assert(
      (bitwidth == 8 || bitwidth == 16 || bitwidth == 32 || bitwidth == 64) &&
      "invalid bitwidth");
  assert((vec == 1 || vec == 2 || vec == 4) && vec == values.size() &&
         "invalid vec size");

  // PTX string
  std::string ptxStr;
  llvm::raw_string_ostream asmOs(ptxStr);
  asmOs << "{\n\t"
        << ".reg .u32 remoteAddr;\n\t"
        << "mapa.shared::cluster.u32 remoteAddr, $0, $1;\n\t"
        << ".reg .pred p;\n\t"
        << "mov.pred p, $2;\n\t";
  asmOs << "@p st.shared::cluster";
  if (vec > 1)
    asmOs << ".v" << vec;
  asmOs << ".u" << bitwidth << " [remoteAddr], ";
  if (vec == 1)
    asmOs << "$3";
  else if (vec == 2)
    asmOs << "{$3, $4}";
  else if (vec == 4)
    asmOs << "{$3, $4, $5, $6}";
  asmOs << ";\n\t"
        << "}\n";

  // Constraints
  std::string constraints;
  llvm::raw_string_ostream conOs(constraints);
  std::string c = bitwidth == 16 ? "h" : (bitwidth == 32 ? "r" : "l");
  conOs << "r,r,b";
  for (unsigned i = 0; i < vec; ++i)
    conOs << "," << c;

  // Arguments
  llvm::SmallVector<llvm::Type *> argTypes;
  llvm::SmallVector<llvm::Value *> args;
  argTypes.push_back(addr->getType());
  args.push_back(addr);
  argTypes.push_back(ctaId->getType());
  args.push_back(ctaId);
  argTypes.push_back(pred->getType());
  args.push_back(pred);
  for (llvm::Value *value : values) {
    argTypes.push_back(value->getType());
    args.push_back(value);
  }

  // Call InlineAsm
  llvm::InlineAsm *iasm = llvm::InlineAsm::get(
      llvm::FunctionType::get(builder.getVoidTy(), argTypes, false), ptxStr,
      constraints, /*hasSideEffect*/ true);
  builder.CreateCall(iasm, args);
}

llvm::Value *createOffsetOfSts64(llvm::IRBuilderBase &builder,
                                 llvm::Value *threadId, llvm::Value *rowOfWarp,
                                 llvm::Value *elemIdx,
                                 uint32_t leadingDimOffset, uint32_t rowStride,
                                 bool swizzleEnabled) {
  if (swizzleEnabled) {
    assert((rowStride == 32 || rowStride == 64 || rowStride == 128) &&
           "wrong rowString for swizzleEnabled");
  }
  llvm::Type *retTy = builder.getInt32Ty();
  llvm::SmallVector<llvm::Value *> args;
  llvm::SmallVector<llvm::Type *> argTys;

  argTys.push_back(threadId->getType());
  args.push_back(threadId);

  argTys.push_back(rowOfWarp->getType());
  args.push_back(rowOfWarp);

  argTys.push_back(elemIdx->getType());
  args.push_back(elemIdx);

  argTys.push_back(builder.getInt32Ty());
  args.push_back(builder.getInt32(leadingDimOffset));

  argTys.push_back(builder.getInt32Ty());
  args.push_back(builder.getInt32(rowStride));

  std::string funcName("__nv_offset_of_sts64");
  auto *module = builder.GetInsertBlock()->getModule();
  auto *func = dyn_cast<llvm::Function>(
      getExternalFuncOP(module, funcName, retTy, argTys).getCallee());
  return builder.CreateCall(func, args);
}

void createSts64(llvm::IRBuilderBase &builder, llvm::Value *offset,
                 llvm::Value *d0, llvm::Value *d1) {
  std::string funcName("__nv_sts64");

  llvm::Type *retTy = builder.getVoidTy();
  llvm::SmallVector<llvm::Value *> args;
  llvm::SmallVector<llvm::Type *> argTys;
  auto i32Ty = builder.getInt32Ty();
  argTys.push_back(i32Ty);
  args.push_back(offset);

  argTys.push_back(i32Ty);
  args.push_back(builder.CreateBitCast(d0, i32Ty));

  argTys.push_back(i32Ty);
  args.push_back(builder.CreateBitCast(d1, i32Ty));

  auto *module = builder.GetInsertBlock()->getModule();
  auto *func = dyn_cast<llvm::Function>(
      getExternalFuncOP(module, funcName, retTy, argTys).getCallee());
  builder.CreateCall(func, args);

  return;
}

class NVGPUDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "triton/Dialect/NVGPU/IR/OpsConversions.inc"

    return failure();
  }
};
} // namespace

void mlir::registerNVGPUDialectTranslation(DialectRegistry &registry) {
  registry.insert<mlir::triton::nvgpu::NVGPUDialect>();
  registry.addExtension(
      +[](MLIRContext *ctx, mlir::triton::nvgpu::NVGPUDialect *dialect) {
        dialect->addInterfaces<NVGPUDialectLLVMIRTranslationInterface>();
      });
}

void mlir::registerNVGPUDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerNVGPUDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
