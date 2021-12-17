#pragma once

#include <numeric>
#include <sstream>
#include <iomanip>
#include "triton/codegen/selection/generator.h"
#include "triton/codegen/target.h"
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/analysis/swizzle.h"
#include "triton/codegen/transform/coalesce.h"
#include "triton/ir/context.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/type.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

namespace triton::codegen{

using namespace llvm;


// types
#define void_ty              builder_->getVoidTy()
#define f16_ty               builder_->getHalfTy()
#define f32_ty               builder_->getFloatTy()
#define i8_ty               builder_->getInt8Ty()
#define i32_ty               builder_->getInt32Ty()
#define vec_ty(type, num_el) VectorType::get(type, num_el, false)
#define ptr_ty(...)          PointerType::get(__VA_ARGS__)
// constants
#define i32(...)             builder_->getInt32(__VA_ARGS__)
// ops
#define and_(...)            builder_->CreateAnd(__VA_ARGS__)
#define atomic_cmp_xchg(...) builder_->CreateAtomicCmpXchg(__VA_ARGS__)
#define atomic_rmw(...)      builder_->CreateAtomicRMW(__VA_ARGS__)
#define bin_op(...)          builder_->CreateBinOp(__VA_ARGS__)
#define bit_cast(...)        builder_->CreateBitCast(__VA_ARGS__)
#define br(...)              builder_->CreateBr(__VA_ARGS__)
#define call(...)            builder_->CreateCall(__VA_ARGS__)
#define cast(...)            builder_->CreateCast(__VA_ARGS__)
#define cond_br(...)         builder_->CreateCondBr(__VA_ARGS__)
#define exact_udiv(...)      builder_->CreateExactUDiv(__VA_ARGS__)
#define extract_elt(...)     builder_->CreateExtractElement(__VA_ARGS__)
#define extract_val(...)     builder_->CreateExtractValue(__VA_ARGS__)
#define fadd(...)            builder_->CreateFAdd(__VA_ARGS__)
#define fcmp(...)            builder_->CreateFCmp(__VA_ARGS__)
#define fmul(...)            builder_->CreateFMul(__VA_ARGS__)
#define fpcast(...)          builder_->CreateFPCast(__VA_ARGS__)
#define fsub(...)            builder_->CreateFSub(__VA_ARGS__)
#define icmp(...)            builder_->CreateICmp(__VA_ARGS__)
#define icmp_eq(...)         builder_->CreateICmpEQ(__VA_ARGS__)
#define icmp_sge(...)        builder_->CreateICmpSGE(__VA_ARGS__)
#define icmp_sle(...)        builder_->CreateICmpSLE(__VA_ARGS__)
#define icmp_ult(...)        builder_->CreateICmpULT(__VA_ARGS__)
#define insert_elt(...)      builder_->CreateInsertElement(__VA_ARGS__)
#define intrinsic(...)       builder_->CreateIntrinsic(__VA_ARGS__)
#define load(...)            builder_->CreateLoad(__VA_ARGS__)
#define lshr(...)            builder_->CreateLShr(__VA_ARGS__)
#define max_num(...)         builder_->CreateMaxNum(__VA_ARGS__)
#define min_num(...)         builder_->CreateMinNum(__VA_ARGS__)
#define neg(...)             builder_->CreateNeg(__VA_ARGS__)
#define phi(...)             builder_->CreatePHI(__VA_ARGS__)
#define ret(...)             builder_->CreateRet(__VA_ARGS__)
#define select(...)          builder_->CreateSelect(__VA_ARGS__)
#define store(...)           builder_->CreateStore(__VA_ARGS__)
#define sub(...)             builder_->CreateSub(__VA_ARGS__)
#define shl(...)             builder_->CreateShl(__VA_ARGS__)
#define udiv(...)            builder_->CreateUDiv(__VA_ARGS__)
#define urem(...)            builder_->CreateURem(__VA_ARGS__)
#define splat(...)           builder_->CreateVectorSplat(__VA_ARGS__)
#define xor_(...)            builder_->CreateXor(__VA_ARGS__)

}