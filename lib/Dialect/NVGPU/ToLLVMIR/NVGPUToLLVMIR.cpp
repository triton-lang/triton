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

void createMBarrierArrive(llvm::IRBuilderBase &builder,
                          mlir::triton::nvgpu::MBarriveType arriveType,
                          llvm::Value *barrier, llvm::Value *pred,
                          llvm::Value *ctaId, uint32_t txCount) {
  auto *module = builder.GetInsertBlock()->getModule();

  llvm::SmallVector<llvm::Type *> argTys;
  argTys.push_back(barrier->getType());
  llvm::Type *retTy = builder.getVoidTy();

  if (arriveType == mlir::triton::nvgpu::MBarriveType::normal) {
    argTys.push_back(pred->getType());
    auto *func = dyn_cast<llvm::Function>(
        getExternalFuncOP(module, "__nv_mbarrier_arrive_normal", retTy, argTys)
            .getCallee());
    builder.CreateCall(func, {barrier, pred});
  } else if (arriveType == mlir::triton::nvgpu::MBarriveType::cp_async) {
    auto *func = dyn_cast<llvm::Function>(
        getExternalFuncOP(module, "__nv_mbarrier_arrive_cp_async", retTy,
                          argTys)
            .getCallee());
    builder.CreateCall(func, {barrier});
  } else if (arriveType == mlir::triton::nvgpu::MBarriveType::expect_tx) {
    assert(txCount > 0 && "txCount should be valid");
    argTys.push_back(builder.getInt32Ty());
    argTys.push_back(pred->getType());

    auto *func = dyn_cast<llvm::Function>(
        getExternalFuncOP(module, "__nv_mbarrier_arrive_expect_tx", retTy,
                          argTys)
            .getCallee());
    builder.CreateCall(func, {barrier, builder.getInt32(txCount), pred});
  } else if (arriveType == mlir::triton::nvgpu::MBarriveType::remote) {
    assert(ctaId && "ctaId should have a valid value");
    argTys.push_back(ctaId->getType());
    argTys.push_back(pred->getType());

    auto *func = dyn_cast<llvm::Function>(
        getExternalFuncOP(module, "__nv_mbarrier_arrive_remote", retTy, argTys)
            .getCallee());
    builder.CreateCall(func, {barrier, ctaId, pred});
  }

  return;
}

llvm::Value *createWGMMADesc(llvm::IRBuilderBase &builder, llvm::Value *buffer,
                             mlir::triton::nvgpu::WGMMADescMode mode,
                             llvm::Value *height) {
  llvm::SmallVector<llvm::Type *> argTys;
  argTys.push_back(buffer->getType());
  argTys.push_back(builder.getInt32Ty());
  argTys.push_back(height->getType());
  llvm::Type *retTy = builder.getInt64Ty();

  llvm::Value *mode_ = builder.getInt32((uint32_t)mode);
  auto *module = builder.GetInsertBlock()->getModule();
  auto *func = dyn_cast<llvm::Function>(
      getExternalFuncOP(module, "__nv_get_wgmma_desc", retTy, argTys)
          .getCallee());
  return builder.CreateCall(func, {buffer, mode_, height});
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

void createTMALoadTiled(llvm::IRBuilderBase &builder, llvm::Value *dst,
                        llvm::Value *mbarrier, llvm::Value *tmaDesc,
                        llvm::Value *l2Desc, llvm::Value *mcastMask,
                        llvm::Value *pred,
                        llvm::SmallVector<llvm::Value *> coords) {
  assert(coords.size() >= 2 && coords.size() <= 5 && "invalid coords.size()");
  auto funcName = getTMALoadFuncName(true, mcastMask != 0, coords.size());
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
  argTys.push_back(l2Desc->getType());
  args.push_back(l2Desc);

  if (mcastMask != nullptr) {
    argTys.push_back(builder.getInt16Ty());
    args.push_back(mcastMask);
  }

  argTys.push_back(pred->getType());
  args.push_back(pred);

  auto *module = builder.GetInsertBlock()->getModule();
  auto *func = dyn_cast<llvm::Function>(
      getExternalFuncOP(module, funcName, retTy, argTys).getCallee());
  builder.CreateCall(func, args);

  return;
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

llvm::Value *createWGMMA(llvm::IRBuilderBase &builder, uint32_t m, uint32_t n,
                         uint32_t k, mlir::triton::nvgpu::WGMMAEltType eltTypeC,
                         mlir::triton::nvgpu::WGMMAEltType eltTypeA,
                         mlir::triton::nvgpu::WGMMAEltType eltTypeB,
                         mlir::triton::nvgpu::WGMMALayout layoutA,
                         mlir::triton::nvgpu::WGMMALayout layoutB,
                         llvm::Value *opA, llvm::Value *opB, llvm::Value *opC) {
  // Simplify enum namespace
  using namespace mlir::triton::nvgpu;

  // Register checks
  auto typeA = opA->getType();
  auto typeB = opB->getType();
  auto typeC = opC->getType();
  auto structTypeA = dyn_cast<llvm::StructType>(typeA);
  auto structTypeB = dyn_cast<llvm::StructType>(typeB);
  auto structTypeC = dyn_cast<llvm::StructType>(typeC);
  assert(!structTypeB && "Operand B can not be registers");
  assert(structTypeC && "Operand C must be registers");

  // Element type, MNK shape and transposing support check
  // Reference:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma
  bool transA = layoutA == WGMMALayout::col;
  bool transB = layoutB == WGMMALayout::row;
  bool supported = false, needTransArgs = false, floatTypeWGMMA = false;
  assert(m % 8 == 0 && n % 8 == 0 && k % 8 == 0);
  // Below instructions do support transposing, must pass `trans` arguments
  supported |=
      (eltTypeA == WGMMAEltType::f16) && (eltTypeB == WGMMAEltType::f16) &&
      (eltTypeC == WGMMAEltType::f16 || eltTypeC == WGMMAEltType::f32) &&
      (m == 64 && 8 <= n && n <= 256 && k == 16);
  supported |= (eltTypeA == WGMMAEltType::bf16) &&
               (eltTypeB == WGMMAEltType::bf16) &&
               (eltTypeC == WGMMAEltType::f32) &&
               (m == 64 && 8 <= n && n <= 256 && k == 16);
  needTransArgs = supported;
  floatTypeWGMMA = supported;
  // Below instructions do not support transposing
  if (!supported && !transA && !transB) {
    supported |= (eltTypeA == WGMMAEltType::tf32) &&
                 (eltTypeB == WGMMAEltType::tf32) &&
                 (eltTypeC == WGMMAEltType::f32) &&
                 (m == 64 && 8 <= n && n <= 256 && k == 8);
    supported |=
        (eltTypeA == WGMMAEltType::e4m3 || eltTypeA == WGMMAEltType::e5m2) &&
        (eltTypeB == WGMMAEltType::e4m3 || eltTypeB == WGMMAEltType::e5m2) &&
        (eltTypeC == WGMMAEltType::f16 || eltTypeC == WGMMAEltType::f32) &&
        (m == 64 && 8 <= n && n <= 256 && k == 32);
    floatTypeWGMMA = supported;
    // Below instructions are integer-based
    supported |= (eltTypeA == WGMMAEltType::s8) &&
                 (eltTypeB == WGMMAEltType::s8) &&
                 (eltTypeC == WGMMAEltType::s32) &&
                 (m == 64 && 8 <= n && n <= 224 && k == 32);
  }
  assert(supported && "WGMMA type or shape is not supported");

  // Build PTX asm
  std::string ptxAsm;
  std::string constraints;
  llvm::raw_string_ostream asmOs(ptxAsm);
  llvm::raw_string_ostream conOs(constraints);
  llvm::SmallVector<llvm::Type *> argTypes;
  llvm::SmallVector<llvm::Value *> args;

  // MMA instruction
  asmOs << "wgmma.mma_async.sync.aligned"
        << ".m" << m << "n" << n << "k" << k << "." << stringifyEnum(eltTypeC)
        << "." << stringifyEnum(eltTypeA) << "." << stringifyEnum(eltTypeB)
        << " ";

  // Operands
  uint32_t asmOpIdx = 0;

  // Operand C
  uint32_t numCRegs = m * n / 128;
  assert(numCRegs == structTypeC->getStructNumElements());
  asmOs << "{";
  for (uint32_t i = 0; i < numCRegs; ++i) {
    argTypes.push_back(structTypeC->getElementType(i));
    args.push_back(builder.CreateExtractValue(opC, {i}));
    asmOs << "$" << asmOpIdx++ << (i == numCRegs - 1 ? "" : ",");
    // LLVM does not support `+` semantic, we must repeat the arguments for both
    // input and outputs
    if (structTypeC->getElementType(0)->isFloatTy())
      conOs << "=f,";
    else
      conOs << "=r,";
  }
  asmOs << "}, ";
  for (uint32_t i = asmOpIdx - numCRegs; i < asmOpIdx; ++i)
    conOs << i << ",";
  // Note that LLVM will not skip the indexed repeating placeholders
  asmOpIdx += numCRegs;

  // Operand A
  if (structTypeA) {
    uint32_t numARegs = m * k / 128;
    assert(numARegs == structTypeA->getNumElements());
    asmOs << "{";
    for (uint32_t i = 0; i < numARegs; ++i) {
      argTypes.push_back(structTypeA->getElementType(i));
      args.push_back(builder.CreateExtractValue(opA, {i}));
      asmOs << "$" << asmOpIdx++ << (i == numARegs - 1 ? "" : ",");
      conOs << "f,";
    }
    asmOs << "}, ";
  } else {
    argTypes.push_back(typeA);
    args.push_back(opA);
    asmOs << "$" << asmOpIdx++ << ", ";
    conOs << "l,";
  }

  // Operand B (must be `desc`)
  argTypes.push_back(typeB);
  args.push_back(opB);
  asmOs << "$" << asmOpIdx++ << ", ";
  conOs << "l";

  // `scale-d` is 1 by default
  asmOs << "1";

  // `imm-scale-a`, and `imm-scale-b` are 1 by default only for float-based
  // WGMMA
  if (floatTypeWGMMA)
    asmOs << ", 1, 1";

  // Push `trans-a` and `trans-b` args if needed (determined as constant)
  if (needTransArgs)
    asmOs << ", " << transA << ", " << transB;
  asmOs << ";";

  // Finally build `llvm::InlineAsm` and call it
  auto inlineAsm = llvm::InlineAsm::get(
      llvm::FunctionType::get(structTypeC, argTypes, false), ptxAsm,
      constraints, true);
  return builder.CreateCall(inlineAsm, args);
}

void createWGMMAFence(llvm::IRBuilderBase &builder) {
  std::string asmStr = "wgmma.fence.sync.aligned;";
  llvm::InlineAsm *iasm =
      llvm::InlineAsm::get(llvm::FunctionType::get(builder.getVoidTy(), {}),
                           asmStr, "", /*hasSideEffect*/ true);
  builder.CreateCall(iasm, {});
}

void createWGMMACommitGroup(llvm::IRBuilderBase &builder) {
  std::string asmStr = "wgmma.commit_group.sync.aligned;";
  llvm::InlineAsm *iasm =
      llvm::InlineAsm::get(llvm::FunctionType::get(builder.getVoidTy(), {}),
                           asmStr, "", /*hasSideEffect*/ true);
  builder.CreateCall(iasm, {});
}

void createWGMMAWaitGroup(llvm::IRBuilderBase &builder, uint32_t pendings) {
  std::string asmStr = (llvm::Twine("wgmma.wait_group.sync.aligned ") +
                        llvm::Twine(pendings) + ";")
                           .str();
  llvm::InlineAsm *iasm =
      llvm::InlineAsm::get(llvm::FunctionType::get(builder.getVoidTy(), {}),
                           asmStr, "", /*hasSideEffect*/ true);
  builder.CreateCall(iasm, {});
}

void createBarSync(llvm::IRBuilderBase &builder, llvm::Value *bar,
                   llvm::Value *numThreads) {
  std::string funcName;
  llvm::Type *retTy = builder.getVoidTy();
  llvm::SmallVector<llvm::Value *> args;
  llvm::SmallVector<llvm::Type *> argTys;
  argTys.push_back(bar->getType());
  args.push_back(bar);

  if (numThreads) {
    funcName = "__nv_bar_cta_sync";
    auto i32Tye = builder.getInt32Ty();
    argTys.push_back(i32Tye);
    args.push_back(builder.CreateBitCast(numThreads, i32Tye));
  } else {
    funcName = "__nv_bar_cta_sync_all";
  }

  auto *module = builder.GetInsertBlock()->getModule();
  auto *func = dyn_cast<llvm::Function>(
      getExternalFuncOP(module, funcName, retTy, argTys).getCallee());
  builder.CreateCall(func, args);
}

llvm::Value *createLoadSharedCluster(llvm::IRBuilderBase &builder,
                                     llvm::Value *addr, llvm::Value *ctaId,
                                     unsigned bitwidth, unsigned vec) {
  assert(
      (bitwidth == 8 || bitwidth == 16 || bitwidth == 32 || bitwidth == 64) &&
      "invalid bitwidth");
  assert((vec == 1 || vec == 2 || vec == 4) && "invalid vec size");

  // PTX string
  std::string ptxStr;
  llvm::raw_string_ostream asmOs(ptxStr);
  unsigned addrArgId = vec, ctaIdArgId = vec + 1;
  asmOs << "{\n\t"
        << ".reg .u32 remoteAddr;\n\t"
        << "mapa.shared::cluster.u32 remoteAddr, $" << addrArgId << ", $"
        << ctaIdArgId << ";\n\t";
  asmOs << "ld.shared::cluster";
  if (vec > 1)
    asmOs << ".v" << vec;
  asmOs << ".u" << bitwidth << " ";
  if (vec == 1)
    asmOs << "$0";
  else if (vec == 2)
    asmOs << "{$0, $1}";
  else
    asmOs << "{$0, $1, $2, $3}";
  asmOs << ", [remoteAddr];\n\t"
        << "}\n";

  // Constraints
  std::string constraints;
  llvm::raw_string_ostream conOs(constraints);
  std::string c = bitwidth == 16 ? "h" : (bitwidth == 32 ? "r" : "l");
  for (unsigned i = 0; i < vec; ++i)
    conOs << "=" << c << ",";
  conOs << "r,r";

  // Arguments
  llvm::SmallVector<llvm::Type *> argTypes;
  llvm::SmallVector<llvm::Value *> args;
  argTypes.push_back(addr->getType());
  args.push_back(addr);
  argTypes.push_back(ctaId->getType());
  args.push_back(ctaId);

  // Return type
  llvm::Type *retTy = builder.getIntNTy(bitwidth);
  llvm::SmallVector<llvm::Type *> retTys(vec, retTy);
  if (vec > 1)
    retTy = llvm::StructType::get(builder.getContext(), retTys);

  // Call InlineAsm
  llvm::InlineAsm *iasm =
      llvm::InlineAsm::get(llvm::FunctionType::get(retTy, argTypes, false),
                           ptxStr, constraints, /*hasSideEffect*/ false);
  return builder.CreateCall(iasm, args);
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

static std::string getTMAStoreFuncName(bool tiled, uint32_t dimSize) {
  std::string funcName;
  llvm::raw_string_ostream os(funcName);
  os << "__nv_tma_store";
  if (tiled)
    os << "_tiled";
  else
    os << "_im2col";

  os << "_" << dimSize << "d";

  return funcName;
}

void createTMAStoreTiled(llvm::IRBuilderBase &builder, llvm::Value *tmaDesc,
                         llvm::Value *src, llvm::Value *pred,
                         llvm::SmallVector<llvm::Value *> coords) {
  assert(coords.size() >= 2 && coords.size() <= 5 && "invalid coords.size()");
  auto funcName = getTMAStoreFuncName(true, coords.size());
  llvm::Type *retTy = builder.getVoidTy();
  llvm::SmallVector<llvm::Value *> args;
  llvm::SmallVector<llvm::Type *> argTys;

  argTys.push_back(tmaDesc->getType());
  args.push_back(tmaDesc);

  argTys.push_back(src->getType());
  args.push_back(src);

  for (auto *c : coords) {
    argTys.push_back(c->getType());
    args.push_back(c);
  }
  argTys.push_back(pred->getType());
  args.push_back(pred);

  auto *module = builder.GetInsertBlock()->getModule();
  auto *func = dyn_cast<llvm::Function>(
      getExternalFuncOP(module, funcName, retTy, argTys).getCallee());
  builder.CreateCall(func, args);

  return;
}

void createStoreMatrix(llvm::IRBuilderBase &builder, llvm::Value *addr,
                       llvm::SmallVector<llvm::Value *> datas) {
  auto size = datas.size();
  assert((size == 1 || size == 2 || size == 4) &&
         "not support size with stmatrix");

  std::string funcName;
  llvm::raw_string_ostream os(funcName);
  os << "__nv_stmatrix_x" << size;

  llvm::Type *retTy = builder.getVoidTy();
  llvm::SmallVector<llvm::Value *> args;
  llvm::SmallVector<llvm::Type *> argTys;

  argTys.push_back(addr->getType());
  args.push_back(addr);

  for (size_t i = 0; i < datas.size(); ++i) {
    argTys.push_back(datas[i]->getType());
    args.push_back(datas[i]);
  }

  auto *module = builder.GetInsertBlock()->getModule();
  auto *func = dyn_cast<llvm::Function>(
      getExternalFuncOP(module, funcName, retTy, argTys).getCallee());
  builder.CreateCall(func, args);
}

llvm::Value *createOffsetOfStmatrixV4(llvm::IRBuilderBase &builder,
                                      llvm::Value *threadId,
                                      llvm::Value *rowOfWarp,
                                      llvm::Value *elemIdx,
                                      uint32_t leadingDimOffset,
                                      uint32_t rowStride, bool swizzleEnabled) {
  if (swizzleEnabled) {
    assert((rowStride == 16 || rowStride == 32 || rowStride == 64) &&
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

  std::string funcName("__nv_offset_of_stmatrix_v4");
  if (!swizzleEnabled)
    funcName = "__nv_offset_of_stmatrix_v4_no_swizzle";
  auto *module = builder.GetInsertBlock()->getModule();
  auto *func = dyn_cast<llvm::Function>(
      getExternalFuncOP(module, funcName, retTy, argTys).getCallee());
  return builder.CreateCall(func, args);
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

llvm::Value *createCvtPack(llvm::IRBuilderBase &builder, llvm::Value *d0,
                           llvm::Value *d1) {
  std::string funcName("__nv_cvt_pack");

  llvm::Type *retTy = builder.getInt32Ty();
  llvm::SmallVector<llvm::Value *> args;
  llvm::SmallVector<llvm::Type *> argTys;
  auto i16Ty = builder.getInt16Ty();

  argTys.push_back(i16Ty);
  args.push_back(builder.CreateBitCast(d0, i16Ty));
  argTys.push_back(i16Ty);
  args.push_back(builder.CreateBitCast(d1, i16Ty));

  auto *module = builder.GetInsertBlock()->getModule();
  auto *func = dyn_cast<llvm::Function>(
      getExternalFuncOP(module, funcName, retTy, argTys).getCallee());
  return builder.CreateCall(func, args);
}

static llvm::Value *getSRegValue(llvm::IRBuilderBase &builder,
                                 llvm::StringRef name) {
  std::string ptxStr;
  llvm::raw_string_ostream asmOs(ptxStr);
  asmOs << "mov.u32 $0, " << name << ";";
  std::string constraints = "=r";
  llvm::InlineAsm *inlineAsm =
      llvm::InlineAsm::get(llvm::FunctionType::get(builder.getInt32Ty(), false),
                           ptxStr, constraints, /*hasSideEffect*/ false);
  return builder.CreateCall(inlineAsm);
}

static llvm::Value *createClusterId(llvm::IRBuilderBase &builder) {
  llvm::Value *x = getSRegValue(builder, "%cluster_ctaid.x");
  llvm::Value *y = getSRegValue(builder, "%cluster_ctaid.y");
  llvm::Value *z = getSRegValue(builder, "%cluster_ctaid.z");
  llvm::Value *nx = getSRegValue(builder, "%cluster_nctaid.x");
  llvm::Value *ny = getSRegValue(builder, "%cluster_nctaid.y");
  llvm::Value *clusterCTAId = builder.CreateAdd(
      x, builder.CreateMul(builder.CreateAdd(y, builder.CreateMul(z, ny)), nx));
  return clusterCTAId;
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
