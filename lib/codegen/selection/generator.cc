#include <numeric>
#include <sstream>
#include <iomanip>
#include <stdexcept>
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
#include "triton/ir/utils.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

namespace triton{
namespace codegen{

using namespace llvm;

Value* adder::operator()(Value *x, Value *y, const std::string& name) {
  // (x + cst) + y -> (x + y) + cst
  if(auto* bin = dyn_cast<BinaryOperator>(x))
  if(bin->getOpcode() == llvm::BinaryOperator::BinaryOps::Add)
  if(dyn_cast<Constant>(bin->getOperand(1))){
    return (*builder_)->CreateAdd((*builder_)->CreateAdd(bin->getOperand(0), y),
                                  bin->getOperand(1));
  }
  // (x + (y + cst)) -> (x + y) + cst
  if(auto* bin = dyn_cast<BinaryOperator>(y))
  if(bin->getOpcode() == llvm::BinaryOperator::BinaryOps::Add)
  if(dyn_cast<Constant>(bin->getOperand(1))){
    return (*builder_)->CreateAdd((*builder_)->CreateAdd(x, bin->getOperand(0)),
                                  bin->getOperand(1));
  }

  // default
  return (*builder_)->CreateAdd(x, y, name);
}

Value* multiplier::operator()(Value *x, Value *y, const std::string &name) {
  // (x + cst1) * cst2 -> (x * cst2) + (cst1 * cst2)
  if(auto* bin = dyn_cast<BinaryOperator>(x))
  if(bin->getOpcode() == llvm::BinaryOperator::BinaryOps::Add)
  if(dyn_cast<Constant>(bin->getOperand(1)))
  if(dyn_cast<Constant>(y)){
    return (*builder_)->CreateAdd((*builder_)->CreateMul(bin->getOperand(0), y),
                                  (*builder_)->CreateMul(bin->getOperand(1), y));
  }
  // default
  return (*builder_)->CreateMul(x, y, name);
}

Value* geper::operator()(Value *ptr, Value* off, const std::string& name){
  // (ptr + cst1) + (cst2) -> ptr + (cst1 + cst2)
  if(auto* gep = dyn_cast<GetElementPtrInst>(ptr))
  if(ConstantInt* cst1 = dyn_cast<ConstantInt>(gep->idx_begin()))
  if(ConstantInt* cst2 = dyn_cast<ConstantInt>(off)){
    return (*builder_)->CreateGEP(gep->getPointerOperand()->getType()->getScalarType()->getPointerElementType(),
                                  gep->getPointerOperand(), (*builder_)->CreateAdd(cst1, cst2));
  }
  // ptr + (off + cst) -> (ptr + off) + cst
  if(auto* bin = dyn_cast<BinaryOperator>(off))
  if(bin->getOpcode() == llvm::BinaryOperator::BinaryOps::Add)
  if(ConstantInt* cst = dyn_cast<ConstantInt>(bin->getOperand(1))){
    Value *gep = (*builder_)->CreateGEP(ptr->getType()->getScalarType()->getPointerElementType(),
                                        ptr, bin->getOperand(0));
    return (*builder_)->CreateGEP(gep->getType()->getScalarType()->getPointerElementType(),
                                  gep, bin->getOperand(1));
  }
  // default
  return (*builder_)->CreateGEP(ptr->getType()->getScalarType()->getPointerElementType(),
                                ptr, off, name);
}

//Value* geper::operator()(Type *ty, Value *ptr, std::vector<Value *> vals, const std::string &name) {
//  return (*builder_)->CreateGEP(ty, ptr, vals, name);
//}

// types
#define void_ty              builder_->getVoidTy()
#define f16_ty               builder_->getHalfTy()
#define bf16_ty              builder_->getInt16Ty()
#define f32_ty               builder_->getFloatTy()
#define i1_ty                builder_->getInt1Ty()
#define i8_ty                builder_->getInt8Ty()
#define i16_ty               builder_->getInt16Ty()
#define i32_ty               builder_->getInt32Ty()
#define i64_ty               builder_->getInt64Ty()
#define vec_ty(type, num_el) VectorType::get(type, num_el, false)
#define ptr_ty(...)          PointerType::get(__VA_ARGS__)
// constants
#define i16(...)             builder_->getInt16(__VA_ARGS__)
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
#define fcmp_oge(...)        builder_->CreateFCmpOGE(__VA_ARGS__)
#define fcmp_ole(...)        builder_->CreateFCmpOLE(__VA_ARGS__)
#define fmul(...)            builder_->CreateFMul(__VA_ARGS__)
#define fpcast(...)          builder_->CreateFPCast(__VA_ARGS__)
#define fsub(...)            builder_->CreateFSub(__VA_ARGS__)
#define icmp(...)            builder_->CreateICmp(__VA_ARGS__)
#define icmp_eq(...)         builder_->CreateICmpEQ(__VA_ARGS__)
#define icmp_sge(...)        builder_->CreateICmpSGE(__VA_ARGS__)
#define icmp_sle(...)        builder_->CreateICmpSLE(__VA_ARGS__)
#define icmp_uge(...)        builder_->CreateICmpUGE(__VA_ARGS__)
#define icmp_ule(...)        builder_->CreateICmpULE(__VA_ARGS__)
#define icmp_ult(...)        builder_->CreateICmpULT(__VA_ARGS__)
#define insert_elt(...)      builder_->CreateInsertElement(__VA_ARGS__)
#define intrinsic(...)       builder_->CreateIntrinsic(__VA_ARGS__)
#define load(ptr)            builder_->CreateLoad(ptr->getType()->getPointerElementType(), ptr)
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

/**
 * \brief Convert Triton-IR Type to LLVM-IR Type
 */
Type *generator::cvt(ir::type *ty) {
  // struct
  if(ty->is_struct_ty()){
    std::vector<Type*> tys;
    for(size_t i = 0; i < ty->get_struct_numel(); i++)
      tys.push_back(cvt(ty->get_struct_type(i)));
    return StructType::get(builder_->getContext(), tys, true);
  }

  // function
  if(auto* tt = dynamic_cast<ir::function_type*>(ty)){
    Type *ret_ty = cvt(tt->get_return_ty());
    std::vector<Type*> arg_tys(tt->get_num_params());
    for(size_t i = 0; i < arg_tys.size(); i++)
      arg_tys[i] = cvt(tt->get_param_ty(i));
    return FunctionType::get(ret_ty, arg_tys, false);
  }
  // pointer
  if(ty->is_pointer_ty()){
    Type *elt_ty = cvt(ty->get_pointer_element_ty());
    unsigned addr_space = ty->get_pointer_address_space();
    return ptr_ty(elt_ty, addr_space);
  }
  // integer
  if(ty->is_integer_ty()){
    unsigned bitwidth = ty->get_integer_bitwidth();
    return IntegerType::get(*ctx_, bitwidth);
  }
  // primitive types
  switch(ty->get_type_id()){
    case ir::type::VoidTyID:      return Type::getVoidTy(*ctx_);
    case ir::type::FP8TyID:       return Type::getInt8Ty(*ctx_);
    case ir::type::FP16TyID:      return Type::getHalfTy(*ctx_);
    case ir::type::BF16TyID:      return Type::getInt16Ty(*ctx_); // use int16 as storage type
    case ir::type::FP32TyID:      return Type::getFloatTy(*ctx_);
    case ir::type::FP64TyID:      return Type::getDoubleTy(*ctx_);
    case ir::type::LabelTyID:     return Type::getLabelTy(*ctx_);
    case ir::type::MetadataTyID:  return Type::getMetadataTy(*ctx_);
    case ir::type::TokenTyID:     return Type::getTokenTy(*ctx_);
    default: break;
  }
  // unknown type
  throw std::runtime_error("unknown conversion from ir::type to Type");
}

/**
 * \brief Convert Triton-IR Attribute to LLVM-IR Attribute
 */
llvm::Attribute generator::cvt(ir::attribute attr) {
  switch(attr.get_kind()){
    case ir::noalias: return llvm::Attribute::get(*ctx_, llvm::Attribute::NoAlias);
    case ir::readonly: return llvm::Attribute::get(*ctx_, llvm::Attribute::ReadOnly);
    case ir::writeonly: return llvm::Attribute::get(*ctx_, llvm::Attribute::WriteOnly);
    case ir::aligned: return llvm::Attribute::get(*ctx_, llvm::Attribute::Alignment, attr.get_value());
    case ir::retune: return llvm::Attribute::get(*ctx_, llvm::Attribute::None);
    default: throw std::runtime_error("cannot convert ir::attribute_t to llvm::Attribute");
  }
}

/**
 * \brief Constructor of LLVM code generator
 */
generator::generator(analysis::axes *a_axes,
                    analysis::layouts *layouts,
                    analysis::align *alignment,
                    analysis::allocation *alloc,
                    analysis::swizzle *swizzle,
                    target *tgt,
                    unsigned num_warps)
  : a_axes_(a_axes), layouts_(layouts), alignment_(alignment), alloc_(alloc), swizzle_(swizzle),
    tgt_(tgt), num_warps_(num_warps), add(&builder_), mul(&builder_), gep(&builder_) {

}

/**
 * \brief Code Generation for `value`
 */
void generator::visit_value(ir::value* v) {
  if(!seen_.insert(v).second)
    return;
  if(v->get_type()->is_block_ty()){
    if(analysis::shared_layout* layout = layouts_->get(v)->to_shared()){
      analysis::N_buffer_info_t *n_buffer = layout->get_N_buffer();
      analysis::double_buffer_info_t *double_buffer = layout->get_double_buffer();

      // offset
      Value *offset = nullptr;
      // base pointer
      Value *ptr = shared_ptr_[layout];

      if (n_buffer) {
        // ptr = base (shared_ptr_[layout]) + smem_idx * size
        // read_smem_idx
        if (v == n_buffer->phi) {
          ptr = shared_ptr_[layout];
        }
        // write_smem_idx
        if (std::find(n_buffer->firsts.begin(), n_buffer->firsts.end(), v) != n_buffer->firsts.end()) {
          int write_smem_idx = /*stage_idx*/n_buffer->firsts_idx.at(v);
          int elements = write_smem_idx * layout->get_per_stage_elements();
          ptr = gep(shared_pre_ptr_[layout], i32(elements));
        } else if (v == n_buffer->latch) {
          Value* write_smem_idx = write_smem_idx_[layout];
          Value* elements = mul(write_smem_idx, i32(layout->get_per_stage_elements()));
          ptr = gep(shared_pre_ptr_[layout], elements);
        }
      } else if (double_buffer) {
        if(v == double_buffer->phi)
          offset = shared_off_[layout];
        if(v == double_buffer->latch)
          ptr = shared_next_ptr_[layout];
        else if(v == double_buffer->first)
          ptr = shared_pre_ptr_[layout];
      } // else do nothing
      // what visit_dot & vist_cts & ... see
      shmems_[v] = ptr;
      // now only latches have offset (PHINode), only used by finalize_share_layout()
      shoffs_[v] = offset;
    }
  }
  // visit operands
  BasicBlock *current = builder_->GetInsertBlock();
  auto *inst = dynamic_cast<ir::instruction*>(v);
  if(inst)
    for(ir::value *op: inst->ops()){
      if(dynamic_cast<ir::constant*>(op) || !dynamic_cast<ir::phi_node*>(v))
        visit_value(op);
    }
  init_idx(v);
  // change insert point for phi node
  builder_->SetInsertPoint(current);
  auto *phi = dynamic_cast<ir::phi_node*>(v);
  if(phi && !current->empty() && current->getFirstNonPHI())
    builder_->SetInsertPoint(&*current->getFirstNonPHI());
  // visit user
  if(auto *usr = dynamic_cast<ir::user*>(v)){
    if(!dynamic_cast<ir::function*>(usr))
      usr->accept(this);
  }
  // revert insert point
  if(phi && !current->empty() && current->getFirstNonPHI())
    builder_->SetInsertPoint(current);
}

/**
 * \brief Code Generation for `phi`
 */
void generator::visit_phi_node(ir::phi_node* x) {
  Type *ty = cvt(x->get_type()->get_scalar_ty());
  for(indices_t idx: idxs_.at(x))
    vals_[x][idx] = phi(ty, x->get_num_operands());
}

/**
 * \brief Code Generation for `call`
 */
void generator::visit_call_inst(ir::call_inst* call) {
  throw std::runtime_error("call not supported! Triton should be inlining everything.");
}

void generator::visit_launch_inst(ir::launch_inst *launch) {
  ir::function* fn = (ir::function*)launch->get_operand(0);
  // forward-declare cudaGetParameterBufferV2
  std::vector<Type*> get_param_arg_tys = {PointerType::get(builder_->getInt8Ty(), 0),
                                           ArrayType::get(builder_->getInt32Ty(), 3),
                                           ArrayType::get(builder_->getInt32Ty(), 3),
                                           builder_->getInt32Ty()};
  Type* get_param_return_ty = PointerType::get(builder_->getInt8Ty(), 0);
  FunctionType* get_param_ty = FunctionType::get(get_param_return_ty, get_param_arg_tys, false);
  Function* get_param_buffer = Function::Create(get_param_ty, Function::ExternalLinkage, "cudaGetParameterBufferV2", mod_);
  Type* grid_ty = get_param_arg_tys[1];
  Type* block_ty = get_param_arg_tys[2];
  AllocaInst* grid = builder_->CreateAlloca(grid_ty);
  AllocaInst* block = builder_->CreateAlloca(block_ty);
  ConstantInt* _0 = builder_->getInt32(0);
  ConstantInt* _1 = builder_->getInt32(1);
  ConstantInt* _2 = builder_->getInt32(2);
  // create basic block
  BasicBlock* launch_done_bb = BasicBlock::Create(builder_->getContext(), "launch_done", builder_->GetInsertBlock()->getParent());
  BasicBlock* launch_bb = BasicBlock::Create(builder_->getContext(), "launch", launch_done_bb->getParent(), launch_done_bb);
  Value *tid = tgt_->get_local_id(mod_, *builder_, 0);
  Value *is_first_thread = builder_->CreateICmpEQ(tid, i32(0));
  builder_->CreateCondBr(is_first_thread, launch_bb, launch_done_bb);
  builder_->SetInsertPoint(launch_bb);

  //
  builder_->CreateStore(vals_[launch->get_grid()[0]][{}], builder_->CreateGEP(grid_ty, grid, {_0, _0}));
  builder_->CreateStore(vals_[launch->get_grid()[1]][{}], builder_->CreateGEP(grid_ty, grid, {_0, _1}));
  builder_->CreateStore(vals_[launch->get_grid()[2]][{}], builder_->CreateGEP(grid_ty, grid, {_0, _2}));
  Value* num_warps = mul(builder_->getInt32(32), vals_[launch->get_num_warps()][{}]);
  builder_->CreateStore(num_warps, builder_->CreateGEP(block_ty, block, {_0, _0}));
  builder_->CreateStore(builder_->getInt32(1), builder_->CreateGEP(block_ty, block, {_0, _1}));
  builder_->CreateStore(builder_->getInt32(1), builder_->CreateGEP(block_ty, block, {_0, _2}));
  Function* called_fn = fns_[fn];
  Value* callee = ConstantExpr::getCast(Instruction::BitCast, called_fn, get_param_arg_tys[0]);
  Value* arg_ptr = builder_->CreateCall(get_param_buffer, {callee, builder_->CreateLoad(grid_ty, grid), builder_->CreateLoad(block_ty, block), builder_->getInt32(0)});
  // forwrd-declare cudaLaunchDeviceV2
  std::vector<Type*> launch_device_arg_tys = {get_param_ty->getReturnType(), builder_->getInt64Ty()};
  FunctionType* launch_device_ty = FunctionType::get(builder_->getInt32Ty(), launch_device_arg_tys, false);
  Function* launch_device = Function::Create(launch_device_ty, Function::ExternalLinkage, "cudaLaunchDeviceV2", mod_);
  // TODO: add branch
  Value* do_not_launch = builder_->CreateICmpEQ(builder_->CreatePtrToInt(arg_ptr, builder_->getInt64Ty()),
                                                builder_->getInt64(0));
  BasicBlock* launch2_bb = BasicBlock::Create(builder_->getContext(), "launch2", launch_done_bb->getParent(), launch_done_bb);
  builder_->CreateCondBr(do_not_launch, launch_done_bb, launch2_bb);
  builder_->SetInsertPoint(launch2_bb);

  unsigned addr_space = arg_ptr->getType()->getPointerAddressSpace();
  unsigned off = 0;
  unsigned last_size = 0;
  for(ir::value* arg: launch->get_values()){
    Value* curr_arg = vals_[arg][{}];
    Type* curr_arg_ty = curr_arg->getType();
    // handle struct alignment
    off += last_size;
    unsigned size = curr_arg_ty->isPointerTy() ? 8 : curr_arg_ty->getPrimitiveSizeInBits() / 8;
    off = (off + size - 1) / size * size;
    // get pointer to current arg
    Value* curr_arg_ptr = builder_->CreateGEP(get_param_return_ty, arg_ptr, builder_->getInt32(off));
    curr_arg_ptr = builder_->CreateBitCast(curr_arg_ptr, curr_arg_ty->getPointerTo(addr_space));
    // store arg
    builder_->CreateStore(curr_arg, curr_arg_ptr);
    last_size = size;
  }
  builder_->CreateCall(launch_device, {arg_ptr, builder_->getInt64(0)});
  builder_->CreateBr(launch_done_bb);
  // done
  builder_->SetInsertPoint(launch_done_bb);

}

/**
 * \brief Code Generation for `binary_operator`
 */
void generator::visit_binary_operator(ir::binary_operator*x) {
  using ll = llvm::Instruction::BinaryOps;
  using tt = ir::binary_op_t;
  auto cvt = [](ir::binary_op_t op){
    switch(op) {
      case tt::Add: return ll::Add;
      case tt::FAdd: return ll::FAdd;
      case tt::Sub: return ll::Sub;
      case tt::FSub: return ll::FSub;
      case tt::Mul: return ll::Mul;
      case tt::FMul: return ll::FMul;
      case tt::UDiv: return ll::UDiv;
      case tt::SDiv: return ll::SDiv;
      case tt::FDiv: return ll::FDiv;
      case tt::URem: return ll::URem;
      case tt::SRem: return ll::SRem;
      case tt::FRem: return ll::FRem;
      case tt::Shl: return ll::Shl;
      case tt::LShr: return ll::LShr;
      case tt::AShr: return ll::AShr;
      case tt::And: return ll::And;
      case tt::Or: return ll::Or;
      case tt::Xor: return ll::Xor;
      default: throw std::runtime_error("unreachable switch");
    }
  };
//  x->print(std::cout);
  for(indices_t idx: idxs_.at(x)){
    Value *lhs = vals_[x->get_operand(0)][idx];
    Value *rhs = vals_[x->get_operand(1)][idx];
    // manually select bf16 bin op
    if (x->get_operand(0)->get_type()->get_scalar_ty()->is_bf16_ty()) {
      assert(x->get_operand(1)->get_type()->get_scalar_ty()->is_bf16_ty());
      if (x->get_op() == tt::FAdd) {  // a + b = a * 1.0 + b
        InlineAsm *bf16_add_asm =
            InlineAsm::get(FunctionType::get(bf16_ty, {bf16_ty, bf16_ty}, false),
                           "{ .reg .b16 c;         \n\t"
                           "   mov.b16 c, 0x3f80U; \n\t" // 1.0
                           "   fma.rn.bf16 $0, $1, c, $2; } \n\t",
                           "=h,h,h", false);
        vals_[x][idx] = builder_->CreateCall(bf16_add_asm, {lhs, rhs});
      } else if (x->get_op() == tt::FSub) {  // a - b = b * (-1.0) + a
        InlineAsm *bf16_sub_asm =
            InlineAsm::get(FunctionType::get(bf16_ty, {bf16_ty, bf16_ty}, false),
                           " { .reg .b16 c;         \n\t"
                           "    mov.b16 c, 0xbf80U; \n\t" // -1.0
                           "    fma.rn.bf16 $0, $2, c, $1;} \n\t",
                           "=h,h,h", false);
        vals_[x][idx] = builder_->CreateCall(bf16_sub_asm, {lhs, rhs});
      } else if (x->get_op() == tt::FMul) {  // a * b = a*b + 0
        InlineAsm *bf16_mul_asm =
          InlineAsm::get(FunctionType::get(bf16_ty, {bf16_ty, bf16_ty}, false),
                           " { .reg .b16 c;        \n\t"
                           "    mov.b16 c, 0x8000U; \n\t" // 0.0
                           "    fma.rn.bf16 $0, $1, $2, c;} \n\t",
                           "=h,h,h", false);
        vals_[x][idx] = builder_->CreateCall(bf16_mul_asm, {lhs, rhs});
      } else
        throw std::runtime_error("invalid bin op for bf16");
    } else {  // not bf16
      auto op = cvt(x->get_op());
      if(op == ll::Add)
        vals_[x][idx] = add(lhs, rhs);
      else if(op == ll::Mul)
        vals_[x][idx] = mul(lhs, rhs);
      else if(op == ll::FDiv && !x->get_fdiv_ieee_rounding() &&
              x->get_type()->get_scalar_ty()->is_fp32_ty()){
        InlineAsm *ptx = InlineAsm::get(FunctionType::get(f32_ty, {f32_ty, f32_ty}, false),
                                        " div.full.f32 $0, $1, $2;", "=r,r,r", false);
        vals_[x][idx] = builder_->CreateCall(ptx, {lhs, rhs});

      }
      else
        vals_[x][idx] = bin_op(op, lhs, rhs);
    }
  }
}

/**
 * \brief Code Generation for `getelementptr`
 */
void generator::visit_getelementptr_inst(ir::getelementptr_inst* x) {
  for(indices_t idx: idxs_.at(x)){
    Value *ptr = vals_[x->get_pointer_operand()][idx];
    std::vector<Value*> vals;
    for(auto it= x->idx_begin(); it != x->idx_end(); it++)
      vals.push_back(vals_[*it][idx]);
    assert(vals.size() == 1);
    vals_[x][idx] = gep(ptr, vals[0]);
  }
}

/**
 * \brief Code Generation for `icmp`
 */
void generator::visit_icmp_inst(ir::icmp_inst* x) {
  auto cvt = [](ir::cmp_pred_t pred) {
    using ll = llvm::CmpInst::Predicate;
    using tt = ir::cmp_pred_t;
    switch(pred){
      case tt::FIRST_ICMP_PREDICATE: return ll::FIRST_ICMP_PREDICATE;
      case tt::ICMP_EQ: return ll::ICMP_EQ;
      case tt::ICMP_NE: return ll::ICMP_NE;
      case tt::ICMP_UGT: return ll::ICMP_UGT;
      case tt::ICMP_UGE: return ll::ICMP_UGE;
      case tt::ICMP_ULT: return ll::ICMP_ULT;
      case tt::ICMP_ULE: return ll::ICMP_ULE;
      case tt::ICMP_SGT: return ll::ICMP_SGT;
      case tt::ICMP_SGE: return ll::ICMP_SGE;
      case tt::ICMP_SLT: return ll::ICMP_SLT;
      case tt::ICMP_SLE: return ll::ICMP_SLE;
      case tt::LAST_ICMP_PREDICATE: return ll::LAST_ICMP_PREDICATE;
      default: throw std::runtime_error("unreachable switch");
    }
  };

  for(indices_t idx: idxs_.at(x)){
    Value *lhs = vals_[x->get_operand(0)][idx];
    Value *rhs = vals_[x->get_operand(1)][idx];
    vals_[x][idx] = icmp(cvt(x->get_pred()), lhs, rhs);
  }
}

/**
 * \brief Code Generation for `fcmp`
 */
void generator::visit_fcmp_inst(ir::fcmp_inst* x) {
  auto cvt = [](ir::cmp_pred_t pred) {
    using ll = llvm::CmpInst::Predicate;
    using tt = ir::cmp_pred_t;
    switch(pred){
      case tt::FIRST_FCMP_PREDICATE: return ll::FIRST_FCMP_PREDICATE;
      case tt::FCMP_FALSE: return ll::FCMP_FALSE;
      case tt::FCMP_OEQ: return ll::FCMP_OEQ;
      case tt::FCMP_OGT: return ll::FCMP_OGT;
      case tt::FCMP_OGE: return ll::FCMP_OGE;
      case tt::FCMP_OLT: return ll::FCMP_OLT;
      case tt::FCMP_OLE: return ll::FCMP_OLE;
      case tt::FCMP_ONE: return ll::FCMP_ONE;
      case tt::FCMP_ORD: return ll::FCMP_ORD;
      case tt::FCMP_UNO: return ll::FCMP_UNO;
      case tt::FCMP_UEQ: return ll::FCMP_UEQ;
      case tt::FCMP_UGT: return ll::FCMP_UGT;
      case tt::FCMP_UGE: return ll::FCMP_UGE;
      case tt::FCMP_ULT: return ll::FCMP_ULT;
      case tt::FCMP_ULE: return ll::FCMP_ULE;
      case tt::FCMP_UNE: return ll::FCMP_UNE;
      case tt::FCMP_TRUE: return ll::FCMP_TRUE;
      case tt::LAST_FCMP_PREDICATE: return ll::LAST_FCMP_PREDICATE;
      default: throw std::runtime_error("unreachable switch");
    }
  };
  for(indices_t idx: idxs_.at(x)){
    Value *lhs = vals_[x->get_operand(0)][idx];
    Value *rhs = vals_[x->get_operand(1)][idx];
    vals_[x][idx] = fcmp(cvt(x->get_pred()), lhs, rhs);
  }
}


std::tuple<Value*, Value*, Value*, Value*> generator::fp32x4_to_fp8x4(Value *in0, Value *in1, Value *in2, Value *in3){
  in0 = cast(llvm::Instruction::FPTrunc, in0, f16_ty);
  in1 = cast(llvm::Instruction::FPTrunc, in1, f16_ty);
  in2 = cast(llvm::Instruction::FPTrunc, in2, f16_ty);
  in3 = cast(llvm::Instruction::FPTrunc, in3, f16_ty);
  Value *ret0, *ret1, *ret2, *ret3;
  std::tie(ret0, ret1, ret2, ret3) = fp16x4_to_fp8x4(in0, in1, in2, in3);
  return std::make_tuple(ret0, ret1, ret2, ret3);
}

std::tuple<Value*, Value*, Value*, Value*> generator::fp8x4_to_fp32x4(Value *in0, Value *in1, Value *in2, Value *in3){
   Value *ret0, *ret1, *ret2, *ret3;
   std::tie(ret0, ret1, ret2, ret3) = fp8x4_to_fp16x4(in0, in1, in2, in3);
   ret0 = cast(llvm::Instruction::FPExt, ret0, f32_ty);
   ret1 = cast(llvm::Instruction::FPExt, ret1, f32_ty);
   ret2 = cast(llvm::Instruction::FPExt, ret2, f32_ty);
   ret3 = cast(llvm::Instruction::FPExt, ret3, f32_ty);
   return std::make_tuple(ret0, ret1, ret2, ret3);
}


std::tuple<Value*, Value*, Value*, Value*> generator::fp8x4_to_fp16x4(Value *in0, Value *in1, Value *in2, Value *in3){
  Type *ret_ty = StructType::get(*ctx_, {vec_ty(f16_ty, 2), vec_ty(f16_ty, 2)});
  InlineAsm *ptx = InlineAsm::get(FunctionType::get(ret_ty, {i32_ty}, false),
  "{"
  ".reg .b32 a<2>, b<2>;                  \n\t"
  "prmt.b32 a0, 0, $2, 0x5040;            \n\t" // If input is 0xdcba set a0 to 0xb0a0
  "prmt.b32 a1, 0, $2, 0x7060;            \n\t" // If input is 0xdcba set a1 to 0xd0c0
  "lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;  \n\t" // b0 = a0 & 0x7fff7fff (strip sign)
  "lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;  \n\t" // b1 = a1 & 0x7fff7fff (strip sign)
  "shr.b32  b0, b0, 1;                    \n\t" // b0 >>= 1 (shift into fp16 position)
  "shr.b32  b1, b1, 1;                    \n\t" // b1 >>= 1 (shift into fp16 position)
  "lop3.b32 $0, b0, 0x80008000, a0, 0xf8; \n\t" // out0 = b0 | (0x80008000 & a0) (restore sign)
  "lop3.b32 $1, b1, 0x80008000, a1, 0xf8; \n\t" // out1 = b1 | (0x80008000 & a1) (restore sign)
  "}", "=r,=r,r", false);
  Value *packed_in = UndefValue::get(vec_ty(i8_ty, 4));
  packed_in = insert_elt(packed_in, in0, (uint64_t)0);
  packed_in = insert_elt(packed_in, in1, (uint64_t)1);
  packed_in = insert_elt(packed_in, in2, (uint64_t)2);
  packed_in = insert_elt(packed_in, in3, (uint64_t)3);
  Value *in = bit_cast(packed_in, i32_ty);
  Value *ret = call(ptx, {in});
  Value *packed_ret0 = extract_val(ret, {0});
  Value *packed_ret1 = extract_val(ret, {1});
  Value *ret0 = extract_elt(packed_ret0, (uint64_t)0);
  Value *ret1 = extract_elt(packed_ret0, (uint64_t)1);
  Value *ret2 = extract_elt(packed_ret1, (uint64_t)0);
  Value *ret3 = extract_elt(packed_ret1, (uint64_t)1);
  return std::make_tuple(ret0, ret1, ret2, ret3);
}

std::tuple<Value*, Value*, Value*, Value*> generator::fp16x4_to_fp8x4(Value *in0, Value *in1, Value *in2, Value *in3) {
  /* fp16 bit representation is seeeeemmmmmmmmmm (s=sign, e=exponent, m=mantissa)
   * fp8  bit representation is seeeemmm
   * The 4 fp8 exponent bits are the low order 4 exponent bits in fp16.
   * The 3 fp8 mantissa bits are the high order 3 mantissa bits in fp16.
   * Note that the low order exponent bits and high order mantissa bits in fp16 are contiguous.
   * We want to round to nearest fp8 value. To do that add 1 to 4th mantissa bit in fp16 (that's
   * one more than the number of mantissa bits in fp8).
   * fp8 = (fp16 & 0x8000) | (((f16 << 1) + 0x0080) & 0x7fff)
   *
   * We compute two fp16s in one uint32. The addition could cause bit flips from one fp16 to the
   * other. To avoid this we zero out the most significant exponent bit. If that bit is set then
   * the value isn't representable in float8 anyway so we assume it's never set (and give garbage
   * output if it is). If we were willing to assume the most significant exponent was never set
   * we could save the first two lop3.b32 instructions below.
   */
  InlineAsm *ptx = InlineAsm::get(FunctionType::get({vec_ty(i8_ty, 4)}, {i32_ty, i32_ty}, false),
  "{"
  ".reg .b32 a<2>, b<2>;                  \n\t"
  "shl.b32 a0, $1, 1;                     \n\t" // a0 = input0 << 1
  "shl.b32 a1, $2, 1;                     \n\t" // a1 = input1 << 1
  "lop3.b32 a0, a0, 0x7fff7fff, 0, 0xc0;  \n\t" // a0 = (a0 & 0x7fff7fff)
  "lop3.b32 a1, a1, 0x7fff7fff, 0, 0xc0;  \n\t" // a1 = (a1 & 0x7fff7fff)
  "add.u32 a0, a0, 0x00800080;            \n\t" // a0 += 0x00800080
  "add.u32 a1, a1, 0x00800080;            \n\t" // a1 += 0x00800080
  "lop3.b32 b0, $1, 0x80008000, a0, 0xea; \n\t" // b0 = (input0 & 0x80008000) | a0
  "lop3.b32 b1, $2, 0x80008000, a1, 0xea; \n\t" // b1 = (input1 & 0x80008000) | a1
  "prmt.b32 $0, b0, b1, 0x7531;           \n\t" // If b0 = 0xabcd and b1=0x0123 sets output to 0xac02
  "}", "=r,r,r", false);
  Value *packed_in0 = UndefValue::get(vec_ty(f16_ty, 2));
  Value *packed_in1 = UndefValue::get(vec_ty(f16_ty, 2));
  packed_in0 = insert_elt(packed_in0, in0, (int)0);
  packed_in0 = insert_elt(packed_in0, in1, (int)1);
  packed_in1 = insert_elt(packed_in1, in2, (int)0);
  packed_in1 = insert_elt(packed_in1, in3, (int)1);
  Value *in_arg0 = bit_cast(packed_in0, i32_ty);
  Value *in_arg1 = bit_cast(packed_in1, i32_ty);
  Value *ret = call(ptx, {in_arg0, in_arg1});
  Value *ret0 = extract_elt(ret, (int)0);
  Value *ret1 = extract_elt(ret, (int)1);
  Value *ret2 = extract_elt(ret, (int)2);
  Value *ret3 = extract_elt(ret, (int)3);
  return std::make_tuple(ret0, ret1, ret2, ret3);
}

std::tuple<Value*, Value*, Value*, Value*> generator::fp8x4_to_bf16x4(Value *in0, Value *in1, Value *in2, Value *in3) {
  // current exp offset: 15
  // Add 112 (127-15) to compensate the difference in exponent bias
  // bf16 = (nosign >> (8-4) + 112 << 7) | sign;
  // bf16 = (nosign >> 4 + 0x3800) | sign;
  Type *ret_ty = StructType::get(*ctx_, {vec_ty(bf16_ty, 2), vec_ty(bf16_ty, 2)});
  InlineAsm *ptx = InlineAsm::get(FunctionType::get(ret_ty, {i32_ty}, false),
  "{"
  ".reg .b32 a<2>, sign<2>, nosign<2>, b<2>; \n\t"
  "prmt.b32 a0, 0, $2, 0x5040; \n\t" // 0xdcba => 0xb0a0
  "prmt.b32 a1, 0, $2, 0x7060; \n\t" // 0xdcba => 0xd0c0
  "and.b32 sign0, a0, 0x80008000; \n\t"
  "and.b32 sign1, a1, 0x80008000; \n\t"
  "and.b32 nosign0, a0, 0x7fff7fff; \n\t"
  "and.b32 nosign1, a1, 0x7fff7fff; \n\t"
  "shr.b32 nosign0, nosign0, 4; \n\t"
  "shr.b32 nosign1, nosign1, 4; \n\t"
  "add.u32 nosign0, nosign0, 0x38003800; \n\t"
  "add.u32 nosign1, nosign1, 0x38003800; \n\t"
  "or.b32 $0, sign0, nosign0; \n\t"
  "or.b32 $1, sign1, nosign1; \n\t"
  "}", "=r,=r,r", false);
  Value *packed_in = UndefValue::get(vec_ty(i8_ty, 4));
  packed_in = insert_elt(packed_in, in0, (uint64_t)0);
  packed_in = insert_elt(packed_in, in1, (uint64_t)1);
  packed_in = insert_elt(packed_in, in2, (uint64_t)2);
  packed_in = insert_elt(packed_in, in3, (uint64_t)3);
  Value *in = bit_cast(packed_in, i32_ty);
  Value *ret = call(ptx, {in});
  Value *packed_ret0 = extract_val(ret, {0});
  Value *packed_ret1 = extract_val(ret, {1});
  Value *ret0 = extract_elt(packed_ret0, (uint64_t)0);
  Value *ret1 = extract_elt(packed_ret0, (uint64_t)1);
  Value *ret2 = extract_elt(packed_ret1, (uint64_t)0);
  Value *ret3 = extract_elt(packed_ret1, (uint64_t)1);
  return std::make_tuple(ret0, ret1, ret2, ret3);
}

std::tuple<Value*, Value*, Value*, Value*> generator::bf16x4_to_fp8x4(Value *in0, Value *in1, Value *in2, Value *in3) {
  /* Assuming fp8 exponent offset is 16. bf16 exponent offset is 127.
     Max value in fp8: 0b01111111 (0x7f),
                  bf16: 3ff0
     Min value in fp8: 0b00000000 (0x00)
                  bf16: 0x3c00
     // @note: +0x8 is for "rounding to nearest zero"
     fp8 = (nosign(bf16) - (112 << 7) + 0x8) << 4;
     return fp8 | sign;  // also permute bytes
  */
  InlineAsm *ptx = InlineAsm::get(FunctionType::get({vec_ty(i8_ty, 4)}, {i32_ty, i32_ty}, false),
  "{\n\t"
  ".reg .u32 sign, sign<2>, nosign, nosign<2>; \n\t"
  ".reg .u32 fp8_min, fp8_max, rn_, zero; \n\t"
  "mov.u32 fp8_min, 0x38003800; \n\t"
  "mov.u32 fp8_max, 0x3ff03ff0; \n\t"
  "mov.u32 rn_, 0x80008; \n\t"
  "mov.u32 zero, 0; \n\t"
  "and.b32 sign0, $1, 0x80008000;  \n\t"
  "and.b32 sign1, $2, 0x80008000;  \n\t"
  "prmt.b32 sign, sign0, sign1, 0x7531; \n\t"
  "and.b32 nosign0, $1, 0x7fff7fff; \n\t"
  "and.b32 nosign1, $2, 0x7fff7fff; \n\t"

  ".reg .u32 nosign_0_<2>, nosign_1_<2>; \n\t"  // nosign = clamp(nosign, min, max)
  "and.b32 nosign_0_0, nosign0, 0xffff0000; \n\t"
  "max.u32 nosign_0_0, nosign_0_0, 0x38000000; \n\t"
  "min.u32 nosign_0_0, nosign_0_0, 0x3ff00000; \n\t"
  "and.b32 nosign_0_1, nosign0, 0x0000ffff; \n\t"
  "max.u32 nosign_0_1, nosign_0_1, 0x3800; \n\t"
  "min.u32 nosign_0_1, nosign_0_1, 0x3ff0; \n\t"
  "or.b32 nosign0, nosign_0_0, nosign_0_1; \n\t"
  "and.b32 nosign_1_0, nosign1, 0xffff0000; \n\t"
  "max.u32 nosign_1_0, nosign_1_0, 0x38000000; \n\t"
  "min.u32 nosign_1_0, nosign_1_0, 0x3ff00000; \n\t"
  "and.b32 nosign_1_1, nosign1, 0x0000ffff; \n\t"
  "max.u32 nosign_1_1, nosign_1_1, 0x3800; \n\t"
  "min.u32 nosign_1_1, nosign_1_1, 0x3ff0; \n\t"
  "or.b32 nosign1, nosign_1_0, nosign_1_1; \n\t"

  "add.u32 nosign0, nosign0, rn_; \n\t"  // round to nearest zero
  "add.u32 nosign1, nosign1, rn_; \n\t"
  "sub.u32 nosign0, nosign0, 0x38003800; \n\t"  // compensate offset
  "sub.u32 nosign1, nosign1, 0x38003800; \n\t"
  "shr.u32 nosign0, nosign0, 4; \n\t"
  "shr.u32 nosign1, nosign1, 4; \n\t"
  "prmt.b32 nosign, nosign0, nosign1, 0x6420; \n\t"
  "or.b32 $0, nosign, sign; \n\t"
  ""
  "}", "=r,r,r", false);
  Value *packed_in0 = UndefValue::get(vec_ty(bf16_ty, 2));
  Value *packed_in1 = UndefValue::get(vec_ty(bf16_ty, 2));
  packed_in0 = insert_elt(packed_in0, in0, (int)0);
  packed_in0 = insert_elt(packed_in0, in1, (int)1);
  packed_in1 = insert_elt(packed_in1, in2, (int)0);
  packed_in1 = insert_elt(packed_in1, in3, (int)1);
  Value *in_arg0 = bit_cast(packed_in0, i32_ty);
  Value *in_arg1 = bit_cast(packed_in1, i32_ty);
  Value *ret = call(ptx, {in_arg0, in_arg1});
  Value *ret0 = extract_elt(ret, (int)0);
  Value *ret1 = extract_elt(ret, (int)1);
  Value *ret2 = extract_elt(ret, (int)2);
  Value *ret3 = extract_elt(ret, (int)3);
  return std::make_tuple(ret0, ret1, ret2, ret3);
}

Value* generator::bf16_to_fp32(Value *in0){
  if (tgt_->as_nvidia()->sm() >= 80) {
    InlineAsm *ptx = InlineAsm::get(FunctionType::get(f32_ty, {bf16_ty}, false),
                                    "cvt.rn.f32.bf16 $0, $1;", "=r,h", false);
    return call(ptx, {in0});
  } else {
    Value *ret = UndefValue::get(vec_ty(i16_ty, 2));
    ret = insert_elt(ret, bit_cast(in0, i16_ty), (uint64_t)1);
    ret = insert_elt(ret, bit_cast(builder_->getInt16(0), i16_ty), (uint64_t)0);
    return bit_cast(ret, f32_ty);
  }
}

Value* generator::fp32_to_bf16(Value *in0){
  if(tgt_->as_nvidia()->sm() >= 80){
    InlineAsm *ptx = InlineAsm::get(FunctionType::get(bf16_ty, {f32_ty}, false),
                                    "cvt.rn.bf16.f32 $0, $1;", "=h,r", false);
    return call(ptx, {in0});
  }
  return extract_elt(bit_cast(in0, vec_ty(i16_ty, 2)), (uint64_t)1);
}

/**
 * \brief Code Generation for `cast`
 */
void generator::visit_cast_inst(ir::cast_inst* x) {
  ir::value *op = x->get_operand(0);
  ir::type* ret_sca_ty = x->get_type()->get_scalar_ty();
  ir::type* op_sca_ty = op->get_type()->get_scalar_ty();
  auto x_idxs = idxs_.at(x);
  auto op_idxs = idxs_.at(op);

  // <> FP8
  if(ret_sca_ty->is_fp8_ty() || op_sca_ty->is_fp8_ty()){
    // ensure that conversions can be vectorized
    int ld = layouts_->get(x)->get_order(0);
    int contiguous = layouts_->get(x)->to_scanline()->nts(ld);
    if(contiguous % 4 != 0)
        throw std::runtime_error("unsupported fp32 -> fp8 conversion");

    // run the conversion
    auto cvt = [&](Value* a, Value* b, Value* c, Value* d){
      if(op_sca_ty->is_fp32_ty() && ret_sca_ty->is_fp8_ty())
        return fp32x4_to_fp8x4(a, b, c, d);
      if(op_sca_ty->is_fp16_ty() && ret_sca_ty->is_fp8_ty())
        return fp16x4_to_fp8x4(a, b, c, d);
      if(op_sca_ty->is_fp8_ty() && ret_sca_ty->is_fp16_ty())
        return fp8x4_to_fp16x4(a, b, c, d);
      if(op_sca_ty->is_fp8_ty() && ret_sca_ty->is_fp32_ty())
        return fp8x4_to_fp32x4(a, b, c, d);
      // fp8 <> bf16
      if(op_sca_ty->is_fp8_ty() && ret_sca_ty->is_bf16_ty())
        return fp8x4_to_bf16x4(a, b, c, d);
      if (op_sca_ty->is_bf16_ty() && ret_sca_ty->is_fp8_ty())
        return bf16x4_to_fp8x4(a, b, c, d);
      throw std::runtime_error("unsupported conversion");
    };
    for(size_t i = 0; i < x_idxs.size(); i+=4){
        std::tie(vals_[x][x_idxs[i+0]],
                 vals_[x][x_idxs[i+1]],
                 vals_[x][x_idxs[i+2]],
                 vals_[x][x_idxs[i+3]]) = cvt(vals_[op][op_idxs[i+0]],
                                              vals_[op][op_idxs[i+1]],
                                              vals_[op][op_idxs[i+2]],
                                              vals_[op][op_idxs[i+3]]);
    }
    return;
  }

  // <> BF16
  if(ret_sca_ty->is_bf16_ty() || op_sca_ty->is_bf16_ty()){
    // FP32 -> BF16
    if(op_sca_ty->is_fp32_ty()){
      for (indices_t idx: idxs_.at(x)) {
        Value *arg = vals_[x->get_operand(0)][idx];
        vals_[x][idx] = fp32_to_bf16(arg); // cast(cvt(x->get_op()), arg, ty);
      }
      return;
    }
    // BF16 -> FP32
    if(ret_sca_ty->is_fp32_ty()){
      for(size_t i = 0; i < x_idxs.size(); i++)
        vals_[x][x_idxs[i + 0]] = bf16_to_fp32(vals_[op][op_idxs[i + 0]]);
      return;
    }
  }


  Type *ty = cvt(x->get_type()->get_scalar_ty());
  auto cvt = [](ir::cast_op_t op){
    using ll = llvm::Instruction::CastOps;
    using tt = ir::cast_op_t;
    switch(op){
      case tt::Trunc: return ll::Trunc;
      case tt::ZExt: return ll::ZExt;
      case tt::SExt: return ll::SExt;
      case tt::FPTrunc: return ll::FPTrunc;
      case tt::FPExt: return ll::FPExt;
      case tt::UIToFP: return ll::UIToFP;
      case tt::SIToFP: return ll::SIToFP;
      case tt::FPToUI: return ll::FPToUI;
      case tt::FPToSI: return ll::FPToSI;
      case tt::PtrToInt: return ll::PtrToInt;
      case tt::IntToPtr: return ll::IntToPtr;
      case tt::BitCast: return ll::BitCast;
      case tt::AddrSpaceCast: return ll::AddrSpaceCast;
      default: throw std::runtime_error("unreachable switch");
    }
  };
  for(indices_t idx: idxs_.at(x)){
    Value *arg = vals_[x->get_operand(0)][idx];
    vals_[x][idx] = cast(cvt(x->get_op()), arg, ty);
  }
}

std::tuple<Value*, Value*, Value*, Value*, Value*, Value*, Value*, Value*> generator::int16_to_float16x8(
  Value *in0, Value *scale_x512, Value *shift
){
  /* unpacking 8 int2s packed into an int16 to 8 float16s
   * the algorithm is similar to
   * https://github.com/pytorch/FBGEMM/blob/6a59bb6621ba9ec7d650ccb78b78ea24d62a3904/
     fbgemm_gpu/include/fbgemm_gpu/fbgemm_cuda_utils.cuh#L1492-L1563
   */
  Type *ret_ty = StructType::get(*ctx_, {vec_ty(f16_ty, 2), vec_ty(f16_ty, 2), vec_ty(f16_ty, 2), vec_ty(f16_ty, 2)});
  InlineAsm *ptx = InlineAsm::get(FunctionType::get(ret_ty, {i32_ty, i32_ty, i32_ty}, false),
  "{"
  ".reg .b32 a<2>, b<4>;                   \n\t" // input is 0xab,cd,ef,gh,ab,cd,ef,gh, each a, b etc occupies two bits.
  "and.b32 a0, 0x30300303, $4;            \n\t" // set a0 to 0x0b,00,0f,00,00,0d,00,0h
  "and.b32 a1, 0xc0c00c0c, $4;            \n\t" // set a1 to 0xa0,00,e0,00,00,c0,00,g0
  "prmt.b32 b0, 0, a0, 0x0504;            \n\t" // set b0 to 0x00,00,00,0d,00,00,00,0h
  "prmt.b32 b1, 0, a1, 0x0504;            \n\t" // set b1 to 0x00,00,00,c0,00,00,00,g0
  "prmt.b32 b2, 0, a0, 0x0706;            \n\t" // set b2 to 0x00,00,0b,00,00,00,0f,00
  "prmt.b32 b3, 0, a1, 0x0706;            \n\t" // set b3 to 0x00,00,a0,00,00,00,e0,00
  "mov.b32 a0, 0x78007800;               \n\t" // a0 = 32768
  "mov.b32 a1, 0x70007000;               \n\t" // a1 = 8192
  "mul.f16x2 b0, b0, a0; \n\t" // b0 = b0 * 32768.
  "mul.f16x2 b1, b1, a1; \n\t" // b1 = b1 * 8192.
  "mov.b32 a0, 0x68006800;               \n\t"  // a0 = 2048
  "mov.b32 a1, 0x60006000;               \n\t"  // a1 = 512
  "mul.f16x2 b2, b2, a0; \n\t" // b2 = b2 * 2048.
  "mul.f16x2 b3, b3, a1; \n\t" // b3 = b3 * 512.
  "fma.rn.f16x2 $0, b0, $5, $6; \n\t" // out0 = b0 * scale + shift.
  "fma.rn.f16x2 $1, b1, $5, $6; \n\t" // out1 = b1 * scale + shift.
  "fma.rn.f16x2 $2, b2, $5, $6; \n\t" // out2 = b2 * scale + shift.
  "fma.rn.f16x2 $3, b3, $5, $6; \n\t" // out3 = b3 * scale + shift.
  "}", "=r,=r,=r,=r,r,r,r", false);

  Value *packed_in = UndefValue::get(vec_ty(i16_ty, 2));
  packed_in = insert_elt(packed_in, in0, (int)0);
  packed_in = insert_elt(packed_in, in0, (int)1);
  Value *in = bit_cast(packed_in, i32_ty);

  Value *ret = call(ptx, {in, scale_x512, shift});
  Value *packed_ret0 = extract_val(ret, {0});
  Value *packed_ret1 = extract_val(ret, {1});
  Value *packed_ret2 = extract_val(ret, {2});
  Value *packed_ret3 = extract_val(ret, {3});
  Value *ret0 = extract_elt(packed_ret0, (uint64_t)0); // h
  Value *ret1 = extract_elt(packed_ret1, (uint64_t)0); // g
  Value *ret2 = extract_elt(packed_ret2, (uint64_t)0); // f
  Value *ret3 = extract_elt(packed_ret3, (uint64_t)0); // e
  Value *ret4 = extract_elt(packed_ret0, (uint64_t)1); // d
  Value *ret5 = extract_elt(packed_ret1, (uint64_t)1); // c
  Value *ret6 = extract_elt(packed_ret2, (uint64_t)1); // b
  Value *ret7 = extract_elt(packed_ret3, (uint64_t)1); // a
  return std::make_tuple(ret0, ret1, ret2, ret3, ret4, ret5, ret6, ret7);
}

std::tuple<Value*, Value*, Value*, Value*, Value*, Value*, Value*, Value*> generator::int32_to_float16x8(
  Value *in0, Value *scale_x512, Value *shift
){
  /* unpacking 8 int4s packed into an int32 to 8 float16s
   * the algorithm is similar to
   * https://github.com/pytorch/FBGEMM/blob/6a59bb6621ba9ec7d650ccb78b78ea24d62a3904/
     fbgemm_gpu/include/fbgemm_gpu/fbgemm_cuda_utils.cuh#L1566-L1619
   */
  Type *ret_ty = StructType::get(*ctx_, {vec_ty(f16_ty, 2), vec_ty(f16_ty, 2), vec_ty(f16_ty, 2), vec_ty(f16_ty, 2)});
  InlineAsm *ptx = InlineAsm::get(FunctionType::get(ret_ty, {i32_ty, i32_ty, i32_ty}, false),
  "{"
  ".reg .b32 a<2>, b<4>;                  \n\t"
  "and.b32 a0, 0x0f0f0f0f, $4;            \n\t" // If input is 0xabcdefgh set a to 0x0b0d0f0h
  "and.b32 a1, 0xf0f0f0f0, $4;            \n\t" // If input is 0xabcdefgh set a to 0xa0c0e0g0
  "prmt.b32 b0, 0, a0, 0x0504;            \n\t" // set b0 to 0x000f000h
  "prmt.b32 b1, 0, a1, 0x0504;            \n\t" // set b1 to 0x00e000g0
  "prmt.b32 b2, 0, a0, 0x0706;            \n\t" // set b2 to 0x000b000d
  "prmt.b32 b3, 0, a1, 0x0706;            \n\t" // set b3 to 0x00a000c0
  "mov.b32 a0, 0x78007800;               \n\t"
  "mov.b32 a1, 0x68006800;               \n\t"
  "mul.f16x2 b0, b0, a0; \n\t" // b0 = b0 * 32768.
  "mul.f16x2 b1, b1, a1; \n\t" // b1 = b1 * 2048.
  "mul.f16x2 b2, b2, a0; \n\t" // b2 = b2 * 32768.
  "mul.f16x2 b3, b3, a1; \n\t" // b3 = b3 * 2048.
  "fma.rn.f16x2 $0, b0, $5, $6; \n\t" // out0 = b0 * scale + shift.
  "fma.rn.f16x2 $1, b1, $5, $6; \n\t" // out1 = b1 * scale + shift.
  "fma.rn.f16x2 $2, b2, $5, $6; \n\t" // out0 = b0 * scale + shift.
  "fma.rn.f16x2 $3, b3, $5, $6; \n\t" // out1 = b1 * scale + shift.
  "}", "=r,=r,=r,=r,r,r,r", false);

  Value *ret = call(ptx, {in0, scale_x512, shift});
  Value *packed_ret0 = extract_val(ret, {0});
  Value *packed_ret1 = extract_val(ret, {1});
  Value *packed_ret2 = extract_val(ret, {2});
  Value *packed_ret3 = extract_val(ret, {3});
  Value *ret0 = extract_elt(packed_ret0, (uint64_t)0); // h
  Value *ret1 = extract_elt(packed_ret1, (uint64_t)0); // g
  Value *ret2 = extract_elt(packed_ret0, (uint64_t)1); // f
  Value *ret3 = extract_elt(packed_ret1, (uint64_t)1); // e
  Value *ret4 = extract_elt(packed_ret2, (uint64_t)0); // d
  Value *ret5 = extract_elt(packed_ret3, (uint64_t)0); // c
  Value *ret6 = extract_elt(packed_ret2, (uint64_t)1); // b
  Value *ret7 = extract_elt(packed_ret3, (uint64_t)1); // a
  return std::make_tuple(ret0, ret1, ret2, ret3, ret4, ret5, ret6, ret7);
}

std::tuple<Value*, Value*, Value*, Value*> generator::int32_to_float16x4(Value *in0, Value *scale_x512, Value *shift){
  /* unpacking 4 int8s packed into an int32 to 4 fp16s
   * the algorithm is similar to
   * https://github.com/pytorch/FBGEMM/blob/6a59bb6621ba9ec7d650ccb78b78ea24d62a3904/
     fbgemm_gpu/include/fbgemm_gpu/fbgemm_cuda_utils.cuh#L1622-L1646
   */
  Type *ret_ty = StructType::get(*ctx_, {vec_ty(f16_ty, 2), vec_ty(f16_ty, 2)});
  InlineAsm *ptx = InlineAsm::get(FunctionType::get(ret_ty, {i32_ty, i32_ty, i32_ty}, false),
  "{"
  ".reg .b32 a, b<2>;                    \n\t"
  "prmt.b32 b0, 0, $2, 0x0504;            \n\t" // If input is 0xabcdefgh set b0 to 0x00ef00gh
  "prmt.b32 b1, 0, $2, 0x0706;            \n\t" // If input is 0xabcdefgh set b1 to 0x00ab00cd
  "mov.b32 a, 0x78007800;               \n\t"
  "mul.f16x2 b0, b0, a; \n\t" // b0 = b0 * 32768.
  "mul.f16x2 b1, b1, a; \n\t" // b1 = b1 * 32768.
  "fma.rn.f16x2 $0, b0, $3, $4; \n\t" // out0 = b0 * scale + shift.
  "fma.rn.f16x2 $1, b1, $3, $4; \n\t" // out1 = b1 * scale + shift.
  "}", "=r,=r,r,r,r", false);

  Value *ret = call(ptx, {in0, scale_x512, shift});
  Value *packed_ret0 = extract_val(ret, {0});
  Value *packed_ret1 = extract_val(ret, {1});
  Value *ret0 = extract_elt(packed_ret0, (uint64_t)0); // gh
  Value *ret1 = extract_elt(packed_ret0, (uint64_t)1); // ef
  Value *ret2 = extract_elt(packed_ret1, (uint64_t)0); // cd
  Value *ret3 = extract_elt(packed_ret1, (uint64_t)1); // ab
  return std::make_tuple(ret0, ret1, ret2, ret3);
}

std::tuple<Value*, Value*> generator::prepare_scale_shift(Value *scale, Value *shift){
  Value *scale_x512 = fmul(scale, bit_cast(i16(0x6000), f16_ty));
  Value *p_scale_x512 = UndefValue::get(vec_ty(f16_ty, 2));
  p_scale_x512 = insert_elt(p_scale_x512, scale_x512, (int)0);
  p_scale_x512 = insert_elt(p_scale_x512, scale_x512, (int)1);
  p_scale_x512 = bit_cast(p_scale_x512, i32_ty);

  Value *p_shift = UndefValue::get(vec_ty(f16_ty, 2));
  p_shift = insert_elt(p_shift, shift, (int)0);
  p_shift = insert_elt(p_shift, shift, (int)1);
  p_shift = bit_cast(p_shift, i32_ty);

  return std::make_tuple(p_scale_x512, p_shift);
}

/**
 * \brief Code Generation for `dequantize`
 */
void generator::visit_dequantize_inst(ir::dequantize_inst* x) {
  ir::value *op = x->get_operand(0);

  auto src_ty_size_in_bits = op->get_type()->get_scalar_ty()->get_primitive_size_in_bits();

  auto ret_last_dim = (x->get_type()->get_block_shapes()).back();
  auto op_last_dim = (op->get_type()->get_block_shapes()).back();

  auto x_idxs = idxs_.at(x);
  auto op_idxs = idxs_.at(op);

  ir::value *scale = x->get_operand(1);
  ir::value *shift = x->get_operand(2);

  Value *p_scale_x512, *p_shift;
  std::tie(p_scale_x512, p_shift) = prepare_scale_shift(vals_[scale][{}], vals_[shift][{}]);

  int ld = layouts_->get(x)->get_order(0);
  int contiguous = layouts_->get(x)->to_scanline()->nts(ld);

  int op_ld = layouts_->get(op)->get_order(0);
  int op_contiguous = layouts_->get(op)->to_scanline()->nts(op_ld);

  std::string err_msg;
  err_msg = "unsupported dequantization, cannot vectorize properly. x_idxs.size(): "
            + std::to_string(x_idxs.size()) + "; op_idxs.size(): "
            + std::to_string(op_idxs.size()) + "; contiguous: "
            + std::to_string(contiguous) + "; op_contiguous: "
            + std::to_string(op_contiguous) + ". if the condition "
            "is not met, please try adjusting block_size, num_warps or "
            "using tl.multiple_of to hint the input/output ptr address.";

  if (ret_last_dim == 8 * op_last_dim) {
    if((x_idxs.size() != 8 * op_idxs.size()) || (contiguous != 8 * op_contiguous)) {
      throw std::runtime_error(err_msg);
    }

    auto cvt = [&](
      Value* a, Value* scale, Value* shift
    ){
      if (src_ty_size_in_bits == 16){ // int2 quantization, int16 to 8 fp16s
        return int16_to_float16x8(a, scale, shift);
      } else if (src_ty_size_in_bits == 32) { // int4 quantization, int32 to 8 fp16s
        return int32_to_float16x8(a, scale, shift);
      } else {
        throw std::runtime_error("unsupported conversion");
      }
    };

    for(size_t j = 0; j < op_idxs.size(); j++){
        size_t i = j * 8;
        std::tie(vals_[x][x_idxs[i+0]],
                  vals_[x][x_idxs[i+1]],
                  vals_[x][x_idxs[i+2]],
                  vals_[x][x_idxs[i+3]],
                  vals_[x][x_idxs[i+4]],
                  vals_[x][x_idxs[i+5]],
                  vals_[x][x_idxs[i+6]],
                  vals_[x][x_idxs[i+7]]) = cvt(vals_[op][op_idxs[j]], p_scale_x512, p_shift);
    }
  } else if (ret_last_dim == 4 * op_last_dim && src_ty_size_in_bits == 32) { // int8 quantization, int32 to 4 fp16s
    if((x_idxs.size() != 4 * op_idxs.size()) || (contiguous != 4 * op_contiguous)) {
      throw std::runtime_error(err_msg);
    }

    auto cvt = [&](Value* a, Value* scale, Value* shift){
      return int32_to_float16x4(a, scale, shift);
    };

    for(size_t j = 0; j < op_idxs.size(); j++){
        size_t i = j * 4;
        std::tie(vals_[x][x_idxs[i+0]],
                  vals_[x][x_idxs[i+1]],
                  vals_[x][x_idxs[i+2]],
                  vals_[x][x_idxs[i+3]]) = cvt(vals_[op][op_idxs[j]], p_scale_x512, p_shift);
    }
  } else {
    throw std::runtime_error("unsupported dequantization");
  }
  return;
}

/**
 * \brief Code Generation for `return`
 */
void generator::visit_return_inst(ir::return_inst* rr) {
  ir::value *ret_val = rr->get_return_value();
  ret(ret_val ? vals_[ret_val][{}] : nullptr);
}

/**
 * \brief Code Generation for `cond_branch`
 */
void generator::visit_cond_branch_inst(ir::cond_branch_inst* br) {
  BasicBlock *true_dest  = bbs_.at(br->get_true_dest());
  BasicBlock *false_dest = bbs_.at(br->get_false_dest());
  Value *cond = vals_[br->get_cond()][{}];
  cond_br(cond, true_dest, false_dest);
}

/**
 * \brief Code Generation for `uncond_branch`
 */
void generator::visit_uncond_branch_inst(ir::uncond_branch_inst* br) {
  BasicBlock *dest = bbs_.at(br->get_dest());
  br(dest);
}

/**
 * \brief Code Generation for a (synchronous) `load`
 */
void generator::visit_load_inst(ir::load_inst* x){
  BasicBlock *current = builder_->GetInsertBlock();
  Module *module = current->getModule();
  Value *tid = tgt_->get_local_id(module, *builder_, 0);
  Value *lane = urem(tid, i32(32));
  ir::value *op = x->get_pointer_operand();
  ir::masked_load_inst *mx = dynamic_cast<ir::masked_load_inst*>(x);
  Type* ty  = cvt(op->get_type()->get_scalar_ty()->get_pointer_element_ty());
  // compute vector width
  size_t vec = 1;
  bool is_mma_first_row = false;
  if(op->get_type()->is_block_ty()){
    auto   ord = ords_.at(op);
    size_t aln = alignment_->get(op, ord[0]);
    if(mx){
      size_t max_eq = alignment_->get_cst_info(mx->get_mask_operand())[ord[0]].num_cst;
      max_eq = std::max<size_t>(max_eq, 1);
      aln = std::min(aln, max_eq);
    }
    analysis::distributed_layout* layout = dynamic_cast<analysis::distributed_layout*>(layouts_->get(x));
    assert(layout);

    vec = std::min<size_t>(layout->contig_per_thread(ord[0]), aln);
    // TODO: generalize
    is_mma_first_row = (ord.size() >= 1) && layout->to_mma() &&
                       (a_axes_->get(x, ord[0]) == layouts_->get(x)->get_axis(1));
    if(is_mma_first_row)
      vec = std::min<size_t>(2, aln);
  }
  // code generation
  auto idxs = idxs_.at(x);
  for(size_t i = 0; i < idxs.size(); i += vec){
    indices_t idx = idxs[i];
    // pointer value
    Value *ptr = vals_[op][idx];
    // masked load
    size_t dtsize = x->get_type()->get_scalar_ty()->get_primitive_size_in_bits() / 8;
    // input ptr info
    GetElementPtrInst *in_gep = dyn_cast<GetElementPtrInst>(ptr);
    size_t in_off;
    if(in_gep){
        ConstantInt* cst = dyn_cast<ConstantInt>(in_gep->idx_begin());
        in_off = cst ? cst->getValue().getSExtValue()*dtsize : 0;
        ptr = cst ? in_gep->getPointerOperand() : in_gep;
    }
    else{
        in_off = 0;
    }
    Value *pred = mx ? vals_[mx->get_mask_operand()][idx] : builder_->getTrue();
    // if(!op->get_type()->is_block_ty()){
    //   pred = builder_->CreateAnd(pred, icmp_eq(tid, i32(0)));
    // }
    Value *other = mx ? vals_[mx->get_false_value_operand()][idx] : nullptr;
    size_t nbits = dtsize*8;
    // pack sub-words (< 32/64bits) into words
    // each load has width min(nbits*vec, 32/64)
    // and there are (nbits * vec)/width of them
    int max_word_width = std::max<int>(32, nbits);
    int tot_width = nbits*vec;
    int width = std::min(tot_width, max_word_width);
    int n_words = std::max(1, tot_width / width);
    bool has_l2_evict_policy = (x->get_eviction_policy() != ir::load_inst::NORMAL) && tgt_->as_nvidia()->sm() >= 80;
    has_l2_evict_policy = false;
    // has_evict_policy = false; // currently disable until supported in `store`
    // -----
    // create inline asm string
    // -----
    std::ostringstream asm_oss;
    asm_oss << "@$" << n_words; // predicate
    asm_oss << " ld";
    if(x->get_is_volatile())
      asm_oss << ".volatile";
    asm_oss << ".global";
    if (x->get_cache_modifier() == ir::load_inst::CA) asm_oss << ".ca";
    if (x->get_cache_modifier() == ir::load_inst::CG) asm_oss << ".cg";
    if (x->get_eviction_policy() == ir::load_inst::EVICT_FIRST) asm_oss << ".L1::evict_first";
    if (x->get_eviction_policy() == ir::load_inst::EVICT_LAST) asm_oss << ".L1::evict_last";
    if (has_l2_evict_policy) asm_oss << ".L2::cache_hint";
    if(n_words > 1)
      asm_oss << ".v" << n_words; // vector width
    asm_oss << ".b" << width; // word size
    asm_oss << " {";
    for(int i = 0; i < n_words; i++){ // return values
      if(i > 0) asm_oss << ",";
      asm_oss << "$" << i;
    }
    asm_oss << "}";
    asm_oss << ", [ $" << n_words + 1; // load
    asm_oss << " + " << in_off << "]"; // constant offset
    if (has_l2_evict_policy) asm_oss << ", $" << n_words + 2;
    asm_oss << ";";
    bool has_other = other && (other != UndefValue::get(other->getType()));
    std::vector<Value *> others;
    // handle `other` values for indices where the mask
    // is false
    if(has_other)
    for(size_t ii = 0; ii < n_words; ii++){
      size_t size = width / nbits;
      Value *v = UndefValue::get(vec_ty(ty, size));
      for(size_t s = 0; s < size; s++){
        ir::value *false_val = mx->get_false_value_operand();
        v = insert_elt(v, vals_[false_val][idxs[i + ii*size + s]], s);
      }
      v = bit_cast(v, IntegerType::get(*ctx_, width));
      // PTX doesn't support mov.u8, so we need to use mov.u16
      auto mov_width = width < 16 ? 16 : width;
      asm_oss << "\n        ";
      asm_oss << "@!$" << n_words << " mov.u" << mov_width;
      asm_oss << " $" << ii << ", ";
      std::ios_base::fmtflags flags(asm_oss.flags());
      if(ConstantInt* cst = dyn_cast<ConstantInt>(v))
        asm_oss << "0x" << std::hex << cst->getSExtValue();
      else{
        asm_oss << "$" << n_words + has_l2_evict_policy +  2 + ii;
        others.push_back(v);
      }
      asm_oss.flags(flags);
      asm_oss << ";";
    }
    // ----
    // create inline ASM signature
    // ---
    std::vector<Type*> ret_tys(n_words, IntegerType::get(*ctx_, width));
    Type* ret_ty = ret_tys.size() > 1 ? StructType::get(*ctx_, ret_tys) : ret_tys[0];
    // ret_ty->print(llvm::outs());
    std::vector<Type*> arg_tys = {pred->getType(), ptr->getType()};
    for(Value *v: others)
        arg_tys.push_back(v->getType());
    if (has_l2_evict_policy)
      arg_tys.push_back(i64_ty);
    FunctionType *asm_ty = FunctionType::get(ret_ty, arg_tys, false);
    // ---
    // create inline ASM constraints
    // ---
    std::string asm_cstrt;
    for(int ii = 0; ii < n_words; ii++){
      if(ii > 0) asm_cstrt += ",";
      asm_cstrt += (width == 64) ? "=l" : ((width == 32) ? "=r" : "=c");
    }
    asm_cstrt += ",b,l";
    for(size_t ii = 0; ii < others.size(); ii++){
      asm_cstrt += ",";
      asm_cstrt += (width == 64) ? "l" : ((width == 32) ? "r" : "c");
    }
    if (has_l2_evict_policy)
      asm_cstrt += ",l";
    // ---
    // finally call inline ASM
    // ---
    InlineAsm *inlineAsm = InlineAsm::get(asm_ty, asm_oss.str(), asm_cstrt, true);
    std::vector<Value*> args = {pred, ptr};
    for(Value *v: others)
        args.push_back(v);
    if (has_l2_evict_policy)
      args.push_back(policies_.at(x->get_eviction_policy()));


    Value *_ret = call(inlineAsm, args);
    // if(!op->get_type()->is_block_ty()){
    //   Value* cond = icmp_eq(tid, i32(0));
    //   Value* shptr = bit_cast(shmem_, ptr_ty(_ret->getType(), 3));
    //   Instruction* bar = add_barrier();
    //   Instruction *term = llvm::SplitBlockAndInsertIfThen(cond, bar, false);
    //   builder_->SetInsertPoint(term);
    //   store(_ret, shptr);
    //   builder_->SetInsertPoint(bar->getParent());
    //   _ret = load(shptr);
    //   add_barrier();
    // }

    // ---
    // extract and store return values
    // ---
    std::vector<Value *> rets;
    for(unsigned int ii = 0; ii < n_words; ii++){
      Value *curr;
      if(ret_ty->isStructTy())
          curr = extract_val(_ret, {ii});
      else
          curr = _ret;
      rets.push_back(bit_cast(curr, vec_ty(ty, width / (dtsize*8))));
    }
    int tmp = (width / (dtsize * 8));
    for(size_t ii = 0; ii < vec; ii++)
      vals_[x][idxs[i+ii]] = extract_elt(rets[ii/tmp], ii % tmp);
  }
}

void generator::visit_unmasked_load_inst(ir::unmasked_load_inst* x) {
  visit_load_inst(x);
}
void generator::visit_masked_load_inst(ir::masked_load_inst* x) {
  visit_load_inst(x);
}

/**
 * \brief Code Generation for a (synchronous) `store`
 */

void generator::visit_store_inst(ir::store_inst * x){
  ir::masked_store_inst *mx = dynamic_cast<ir::masked_store_inst*>(x);
  // operands
  ir::value *ptr_op = x->get_pointer_operand();
  ir::value *val_op = x->get_value_operand();
  ir::value *msk_op = nullptr;
  if(auto* msk_st = dynamic_cast<ir::masked_store_inst*>(x))
    msk_op = msk_st->get_mask_operand();
  // vector size
  size_t vec = 1;
  if(val_op->get_type()->is_block_ty()){
    auto ord = ords_.at(x->get_pointer_operand());
    size_t aln = alignment_->get(ptr_op, ord[0]);
    size_t nts = axes_.at(a_axes_->get(x->get_pointer_operand(), ord[0])).contiguous;
    if(mx){
      size_t max_eq = alignment_->get_cst_info(mx->get_mask_operand())[ord[0]].num_cst;
      max_eq = std::max<size_t>(max_eq, 1);
      aln = std::min(aln, max_eq);
    }
    analysis::distributed_layout* layout = dynamic_cast<analysis::distributed_layout*>(layouts_->get(ptr_op));
    assert(layout);
    // vec  = std::min(nts, aln);
    vec = std::min<size_t>(layout->contig_per_thread(ord[0]), aln);
    // TODO: generalize
    bool is_mma_first_row = (ord.size() >= 1) && layout->to_mma() &&
                       (a_axes_->get(ptr_op, ord[0]) == layouts_->get(ptr_op)->get_axis(1));
    if(is_mma_first_row)
      vec = std::min<size_t>(2, aln);
  }
  bool has_l2_evict_policy = (x->get_eviction_policy() != ir::load_inst::NORMAL) && tgt_->as_nvidia()->sm() >= 80;
  has_l2_evict_policy = false;
  auto idxs    = idxs_.at(val_op);
  Type *ty = cvt(val_op->get_type()->get_scalar_ty());
  if(ty->isIntegerTy(1))
    ty = builder_->getInt8Ty();
  for(size_t i = 0; i < idxs.size(); i += vec){
    indices_t idx = idxs[i];
    // pointers
    Value *ptr = vals_[ptr_op][idx];
    size_t dtsize = std::max<int>(1, val_op->get_type()->get_scalar_ty()->get_primitive_size_in_bits() / 8);
    GetElementPtrInst *in_gep = dyn_cast<GetElementPtrInst>(ptr);
    size_t in_off;
    if(in_gep){
        ConstantInt* cst = dyn_cast<ConstantInt>(in_gep->idx_begin());
        in_off = cst ? cst->getValue().getSExtValue()*dtsize : 0;
        ptr = cst ? in_gep->getPointerOperand() : in_gep;
    }
    else{
        in_off = 0;
    }
    // mask
    Value *pred = msk_op ? vals_[msk_op][idx] : builder_->getTrue();
    size_t nbits = dtsize*8;
    // pack sub-words (< 32/64bits) into words
    // each load has width min(nbits*vec, 32/64)
    // and there are (nbits * vec)/width of them
    int max_word_width = std::max<int>(32, nbits);
    int tot_width = nbits*vec;
    int width = std::min(tot_width, max_word_width);
    int n_words = std::max(1, tot_width / width);
    // -----
    // create inline asm string
    // -----
    std::ostringstream asm_oss;
    asm_oss << "@$0"; // predicate
    asm_oss << " st.global";
    if (has_l2_evict_policy) asm_oss << ".L2::cache_hint";
    if(n_words > 1)
      asm_oss << ".v" << n_words; // vector width
    asm_oss << ".b" << width; // word size
    asm_oss << " [ $1 + " << in_off << "]";
    asm_oss << " , {";
    for(int i = 0; i < n_words; i++){ // return values
      if(i > 0) asm_oss << ",";
      asm_oss << "$" << 2 + i;
    }
    asm_oss << "}";
    if (has_l2_evict_policy) asm_oss << ", $" << n_words + 2;
    asm_oss << ";";
    // ----
    // create inline ASM signature
    // ---
    Type* val_arg_ty = IntegerType::get(*ctx_, width);
    std::vector<Type*> arg_tys = {pred->getType(), ptr->getType()};
    for(int ii = 0; ii < n_words; ii++)
      arg_tys.push_back(val_arg_ty);
    if (has_l2_evict_policy)
      arg_tys.push_back(i64_ty);
    FunctionType *asm_ty = FunctionType::get(builder_->getVoidTy(), arg_tys, false);
    // ---
    // create inline ASM constraints
    // ---
    std::string asm_cstrt = "b,l";
    for(int ii = 0; ii < n_words; ii++){
      asm_cstrt += ",";
      asm_cstrt += (width == 64) ? "l" : ((width == 32) ? "r" : "c");
    }
    if (has_l2_evict_policy)
      asm_cstrt += ",l";
    // ---
    // finally call inline ASM
    // ---
    InlineAsm *_asm = InlineAsm::get(asm_ty, asm_oss.str(), asm_cstrt, true);
    std::vector<Value*> args = {pred, ptr};
    for(unsigned int ii = 0; ii < n_words; ii++){
      size_t n_subw = width / nbits;
      Value* curr = UndefValue::get(vec_ty(ty, n_subw));
      for(unsigned int jj = 0; jj < n_subw; jj++){
        Value* new_elt = vals_[val_op][idxs[i + ii*n_subw + jj]];
        if(new_elt->getType()->isIntegerTy(1))
          new_elt = builder_->CreateSExt(new_elt, builder_->getInt8Ty());
        new_elt = bit_cast(new_elt, ty);
        curr = builder_->CreateInsertElement(curr, new_elt, jj);
      }
      args.push_back(bit_cast(curr, val_arg_ty));
    }
    if (has_l2_evict_policy)
      args.push_back(policies_.at(x->get_eviction_policy()));
    call(_asm, args);
  }
}
void generator::visit_unmasked_store_inst(ir::unmasked_store_inst* x) {
  visit_store_inst(x);
}
void generator::visit_masked_store_inst(ir::masked_store_inst* x) {
  visit_store_inst(x);
}

// --

void generator::visit_extract_value_inst(ir::extract_value_inst *x) {
  auto idxs    = idxs_.at(x);
  ir::value* agg = x->get_operand(0);
  unsigned insert_idx = x->get_idx();
  for(size_t i = 0; i < idxs.size(); i++){
    auto idx = idxs[i];
    vals_[x][idx] = builder_->CreateExtractValue(vals_[agg][idx], {insert_idx});
  }
}


void generator::visit_insert_value_inst(ir::insert_value_inst *x){
  auto idxs    = idxs_.at(x);
  ir::value* agg = x->get_operand(0);
  ir::value* val = x->get_operand(1);
  unsigned insert_idx = x->get_idx();
  for(size_t i = 0; i < idxs.size(); i++){
    auto idx = idxs[i];
    vals_[x][idx] = builder_->CreateInsertValue(vals_[agg][idx], vals_[val][idx],{insert_idx});
  }
}

// --
/**
 * \brief Code Generation for `cat`
 */
void generator::visit_cat_inst(ir::cat_inst* x) {
  auto idxs = idxs_.at(x);
  ir::value* lhs = x->get_operand(0);
  ir::value* rhs = x->get_operand(1);
  int i = 0;
  for(size_t j = 0; j < idxs_.at(lhs).size(); j ++){
    vals_[x][idxs_[x][i++]] = vals_[lhs][idxs_[lhs][j]];
  }
  for(size_t j = 0; j < idxs_.at(rhs).size(); j ++){
    vals_[x][idxs_[x][i++]] = vals_[rhs][idxs_[rhs][j]];
  }
}



/**
 * \brief Code Generation for `reshape`
 */
void generator::visit_reshape_inst(ir::reshape_inst* x) {
  auto idxs = idxs_.at(x);
  for(size_t i = 0; i < idxs_.at(x).size(); i ++){
    ir::value* op = x->get_operand(0);
    vals_[x][idxs_[x][i]] = vals_[op][idxs_[op][i]];
  };
}

/**
 * \brief Code Generation for `splat`
 */
void generator::visit_splat_inst(ir::splat_inst* x) {
  for(auto idx: idxs_.at(x))
    vals_[x][idx] = vals_[x->get_operand(0)][{}];
}

/**
 * \brief Code Generation for `broadcast`
 */
void generator::visit_broadcast_inst(ir::broadcast_inst* x) {
  ir::value* op = x->get_operand(0);
  const auto& shape = op->get_type()->get_block_shapes();
  for(auto out_idx: idxs_.at(x)){
    indices_t in_idx = out_idx;
    for(size_t k = 0; k < in_idx.size(); k++)
      in_idx[k] = shape[k] == 1 ? i32(0) : in_idx[k];
    vals_[x][out_idx] = vals_[op][in_idx];
  }
//  for(size_t i = 0; i < idxs_.at(x).size(); i++)
//    vals_[x][idxs_[x][i]] = vals_[op][idxs_[op][i]];
}

/**
 * \brief Code Generation for `downcast`
 */
void generator::visit_downcast_inst(ir::downcast_inst* x) {
  vals_[x][{}] = vals_[x->get_operand(0)][{i32(0)}];
}

/**
 * \brief Code Generation for `get_program_id`
 */
void generator::visit_get_program_id_inst(ir::get_program_id_inst* pid) {
  Module *module = builder_->GetInsertBlock()->getModule();
  Value *ret = tgt_->get_block_id(module, *builder_, pid->get_axis());
  vals_[pid][{}] = ret;
}

/**
 * \brief Code Generation for `get_num_programs`
 */
void generator::visit_get_num_programs_inst(ir::get_num_programs_inst* np) {
  Module *module = builder_->GetInsertBlock()->getModule();
  Value *ret = tgt_->get_num_blocks(module, *builder_, np->get_axis());
  vals_[np][{}] = ret;
}

/**
 * \brief Code Generation for `exp`
 */
void generator::visit_exp_inst(ir::exp_inst* x){
  Constant *log2e = ConstantFP::get(f32_ty, 1.4426950408889634);
  std::vector<llvm::Type*> tys = {f32_ty};
  FunctionType *fn_ty = FunctionType::get(f32_ty, tys, false);
  InlineAsm *ex2 = InlineAsm::get(fn_ty, "ex2.approx.f32 $0, $0;", "=f,0", false);
  for(auto idx: idxs_.at(x)){
    Value *ex2arg = fmul(vals_[x->get_operand(0)][idx], log2e);
    // Value *ex2arg = vals_[x->get_operand(0)][idx];
    vals_[x][idx] = call(ex2, std::vector<llvm::Value*>{ex2arg});
  }
}

/**
 * \brief Code Generation for `cos`
 */
void generator::visit_cos_inst(ir::cos_inst* x){
  std::vector<llvm::Type*> tys = {f32_ty};
  FunctionType *fn_ty = FunctionType::get(f32_ty, tys, false);
  InlineAsm *cos = InlineAsm::get(fn_ty, "cos.approx.f32 $0, $0;", "=f,0", false);
  for(auto idx: idxs_.at(x)){
    vals_[x][idx] = call(cos, std::vector<llvm::Value*>{vals_[x->get_operand(0)][idx]});
  }
}

/**
 * \brief Code Generation for `umulhi`
 */
void generator::visit_umulhi_inst(ir::umulhi_inst* x){
  std::vector<llvm::Type*> tys = {i32_ty, i32_ty};
  FunctionType *fn_ty = FunctionType::get(i32_ty, tys, false);
  InlineAsm *umulhi = InlineAsm::get(fn_ty, "mul.hi.u32 $0, $1, $2;", "=r,r,r", false);
  for(auto idx: idxs_.at(x)){
    Value* lhs = vals_[x->get_operand(0)][idx];
    Value* rhs = vals_[x->get_operand(1)][idx];
    vals_[x][idx] = call(umulhi, std::vector<llvm::Value*>{lhs, rhs});
  }
 }

/**
 * \brief Code Generation for `sin`
 */
void generator::visit_sin_inst(ir::sin_inst* x){
  std::vector<llvm::Type*> tys = {f32_ty};
  FunctionType *fn_ty = FunctionType::get(f32_ty, tys, false);
  InlineAsm *sin = InlineAsm::get(fn_ty, "sin.approx.f32 $0, $0;", "=f,0", false);
  for(auto idx: idxs_.at(x)){
    vals_[x][idx] = call(sin, std::vector<llvm::Value*>{vals_[x->get_operand(0)][idx]});
  }
 }

/**
 * \brief Code Generation for `log`
 */
void generator::visit_log_inst(ir::log_inst* x){
  Constant *rcplog2e = ConstantFP::get(f32_ty, 0.6931471805599453);
  std::vector<llvm::Type*> tys = {f32_ty};
  FunctionType *fn_ty = FunctionType::get(f32_ty, tys, false);
  InlineAsm *lg2 = InlineAsm::get(fn_ty, "lg2.approx.f32 $0, $1;", "=f,f", false);
  for(auto idx: idxs_.at(x)){
    Value *lg2arg = call(lg2, std::vector<llvm::Value*>{vals_[x->get_operand(0)][idx]});
    vals_[x][idx] = fmul(lg2arg, rcplog2e);
  }
}

/**
 * \brief Code Generation for `atomic_cas`
 */
void generator::visit_atomic_cas_inst(ir::atomic_cas_inst* cas) {
  BasicBlock *current = builder_->GetInsertBlock();
  Module *module = current->getModule();
  Value *tid = tgt_->get_local_id(module, *builder_, 0);
  Value *pred = icmp_eq(tid, i32(0));
//  BasicBlock *tid_0_bb = BasicBlock::Create(*ctx_, "tid_0", current->getParent());
//  BasicBlock *tid_0_done_bb = BasicBlock::Create(*ctx_, "tid_0_done", current->getParent());
  add_barrier();
  tgt_->add_memfence(module, *builder_);
  Value *atom_ptr;
  atom_ptr = gep(shmem_, i32(alloc_->offset(layouts_->get(layouts_->tmp(cas)))), "");
  atom_ptr = bit_cast(atom_ptr, ptr_ty(cvt(cas->get_type()->get_scalar_ty()), 3));
//  cond_br(pred, tid_0_bb, tid_0_done_bb);
//  builder_->SetInsertPoint(tid_0_bb);
  Value *cas_ptr = vals_[cas->get_operand(0)][{}];
  Value *cas_cmp = vals_[cas->get_operand(1)][{}];
  Value *cas_val = vals_[cas->get_operand(2)][{}];
  std::string asm_str = "@$1 atom.global.cas.b32 $0, [$2], $3, $4;";
  FunctionType *fn_ty = FunctionType::get(i32_ty, {pred->getType(), cas_ptr->getType(), cas_cmp->getType(), cas_val->getType()}, false);
  InlineAsm *iasm = InlineAsm::get(fn_ty, asm_str, "=r,b,l,r,r", true);
  add_barrier();
  Value *old = call(iasm, {pred, cas_ptr, cas_cmp, cas_val});
  add_barrier();

  std::string asm2_str = "@$0 st.shared.b32 [$1], $2;";
  FunctionType *fn2_ty = FunctionType::get(void_ty, {pred->getType(), atom_ptr->getType(), old->getType()}, false);
  InlineAsm *iasm2 = InlineAsm::get(fn2_ty, asm2_str, "b,r,r", true);
  add_barrier();
  call(iasm2, {pred, atom_ptr, old});
  tgt_->add_memfence(module, *builder_);
  add_barrier();
  vals_[cas][{}] = load(atom_ptr);
  add_barrier();
}

/**
 * \brief Code Generation for `atomic_rmw`
 */
void generator::visit_atomic_rmw_inst(ir::atomic_rmw_inst *atom) {
  ir::value* ptr = atom->get_operand(0);
  ir::value* val = atom->get_operand(1);
  ir::value* msk = atom->get_operand(2);

  // vector size
  int vec = 1;
  Value *mask = builder_->getInt1(true);
  if(atom->get_type()->is_block_ty()){
    auto shape = atom->get_type()->get_block_shapes();
    int ld = ords_.at(ptr)[0];
    unsigned alignment = alignment_->get(ptr, ld);
    vec = std::min<int>(layouts_->get(ptr)->to_scanline()->nts(ld), alignment);
    vec = std::min(vec, val->get_type()->get_tile_element_ty()->is_fp16_ty() ? 2 : 1);
    // mask out inactive threads
    analysis::data_layout* layout = layouts_->get(val);
    auto curr_axes = a_axes_->get(val);
    auto layt_axes = layout->get_axes();
    for(unsigned k = 0; k < layt_axes.size(); k++){
      unsigned ax = layt_axes.at(k);
      distributed_axis dax = axes_.at(ax);
      // axis is part of the original layout: thread id should be 0
      // but not the current layout
      if(std::find(curr_axes.begin(), curr_axes.end(), ax) == curr_axes.end())
        mask = and_(mask, icmp_eq(dax.thread_id, i32(0)));
    }
    // last axis may spillover
    Value *thread_id = tgt_->get_local_id(mod_, *builder_, 0);
    int per_thread = 1;
    for(int ax: layt_axes) { per_thread *= axes_.at(ax).contiguous; }
    int numel = 1;
    for(int s: layout->get_shape()) { numel *= s; }
    mask = and_(mask, icmp_ult(mul(thread_id, i32(per_thread)), i32(numel)));
  }


  for(int i = 0; i < idxs_.at(val).size(); i += vec){
    auto idx = idxs_[val][i];
    Value *rmw_val = UndefValue::get(vec_ty(vals_[val][idx]->getType(), vec));
    for(int ii = 0; ii < vec; ii++)
      rmw_val = insert_elt(rmw_val, vals_[val][idxs_[val][i+ii]], ii);
    Value *rmw_ptr = vals_[ptr][idx];
    Value *rmw_msk = vals_[msk][idx];
    rmw_msk = and_(rmw_msk, mask);
    if(vec == 1)
      rmw_val = extract_elt(rmw_val, i32(0));
    Type* ty = rmw_val->getType();
    size_t nbits = ty->getScalarSizeInBits();
    // extract pointer offset
    std::string offset = "";
    if(GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(rmw_ptr))
    if(gep->getNumIndices() == 1)
    if(ConstantInt *cst = dyn_cast<ConstantInt>(gep->idx_begin())){
      offset = " + " + std::to_string(cst->getValue().getSExtValue()*nbits/8);
      rmw_ptr = gep->getPointerOperand();
    }
    rmw_ptr = bit_cast(rmw_ptr, ty->getPointerTo(1));
    // asm argument type
    std::vector<Type*> arg_ty = {rmw_msk->getType(), rmw_ptr->getType(), rmw_val->getType()};
    // asm function type
    FunctionType *fn_ty = FunctionType::get(ty, arg_ty, false);
    // asm string
    std::string s_nbits = std::to_string(nbits);
    std::string name;
    std::string s_ty;
    using tt = ir::atomic_rmw_op_t;
    switch(atom->get_op()){
      case tt::Or: name = "or"; s_ty = "b"; break;
      case tt::And: name = "and"; s_ty = "b"; break;
      case tt::Xor: name = "xor", s_ty = "b"; break;
      case tt::Add: name = "add" , s_ty = "s"; break;
      case tt::Min: name = "min", s_ty = "s"; break;
      case tt::Max: name = "max", s_ty = "s"; break;
      case tt::UMin: name = "min", s_ty = "u"; break;
      case tt::UMax: name = "max", s_ty = "u"; break;
      case tt::FAdd: name = "add", s_ty = "f"; break;
      case tt::Xchg: name = "exch", s_ty = "b"; break;
    }
    std::string s_vec = vec == 2 ? "x2" : "";
    std::string mod = nbits == 16 ? ".noftz" : "";

    std::string asm_str = "@$1 atom.global.gpu." + name + mod + "." + s_ty + s_nbits + s_vec + " $0, [$2" + offset + "], $3;";
    std::string ty_id = nbits*vec == 64 ? "l" : (nbits*vec == 32 ? "r" : "h");
    std::string constraint = "=" + ty_id + ",b,l," + ty_id;
    // create inline asm
    InlineAsm *iasm = InlineAsm::get(fn_ty, asm_str, constraint, true);
    // call asm
    if(atom->get_type()->is_block_ty())
      vals_[atom][idx] = call(iasm, (ArrayRef<Value*>{rmw_msk, rmw_ptr, rmw_val}));
    else{
      Module *mod = builder_->GetInsertBlock()->getModule();
      tgt_->add_memfence(mod, *builder_);
      add_barrier();
      Value *tid = tgt_->get_local_id(mod, *builder_, 0);
      rmw_msk = builder_->CreateAnd(rmw_msk, icmp_eq(tid, i32(0)));
      Value *old = call(iasm, (ArrayRef<Value*>{rmw_msk, rmw_ptr, rmw_val}));
      Value *atom_ptr;
      atom_ptr = gep(shmem_, i32(alloc_->offset(layouts_->get(layouts_->tmp(atom)))), "");
      atom_ptr = bit_cast(atom_ptr, ptr_ty(old->getType(), 3));
      store(old, atom_ptr);
      add_barrier();
      vals_[atom][idx] = load(atom_ptr);
      add_barrier();
    }
  }
}

/**
 * \brief Code Generation for `mma.884` (V100)
 */
//TODO: clean-up
void generator::visit_mma884(ir::dot_inst* C, ir::value *A, ir::value *B, ir::value *D, unsigned NK) {
  // shapes
  auto shape_c = C->get_type()->get_block_shapes();
  auto shape_a = A->get_type()->get_block_shapes();
  auto shape_b = B->get_type()->get_block_shapes();
  // order
  auto ord_a = layouts_->get(A)->get_order();
  auto ord_b = layouts_->get(B)->get_order();
  bool is_a_trans = C->is_trans_a();
  // is_a_trans = false;
  if(C->is_trans_a()){
    std::swap(ord_a[0], ord_a[1]);
    std::swap(shape_a[0], shape_a[1]);
    std::swap(offset_a_m_, offset_a_k_);
  }
  // std::cout << "visiting" << std::endl;
  // if(C->is_trans_b()){
  //   std::swap(ord_b[0], ord_b[1]);
    // std::swap(shape_b[0], shape_b[1]);
  // }
  // layouts
  analysis::mma_layout*    layout_c = layouts_->get(C)->to_mma();
  analysis::shared_layout* layout_a = layouts_->get(A)->to_shared();
  analysis::shared_layout* layout_b = layouts_->get(B)->to_shared();
  // vectorization
  int vec_a = swizzle_->get_vec(layout_a);
  int vec_b = swizzle_->get_vec(layout_b);
  // strides
  bool is_a_row = ord_a[0] != 0;
  bool is_b_row = ord_b[0] != 0;
  int stride_am = is_a_row ? shape_a[1] : 1;
  int stride_ak = is_a_row ? 1 : shape_a[0];
  int stride_a0 = is_a_row ? stride_ak : stride_am;
  int stride_a1 = is_a_row ? stride_am : stride_ak;
  int stride_bn = is_b_row ? 1 : shape_b[0];
  int stride_bk = is_b_row ? shape_b[1] : 1;
  int stride_b0 = is_b_row ? stride_bn : stride_bk;
  int stride_b1 = is_b_row ? stride_bk : stride_bn;
  int stride_rep_m = layout_c->wpt(0) * layout_c->fpw(0) * 8;
  int stride_rep_n = layout_c->wpt(1) * layout_c->fpw(1) * 8;
  int stride_rep_k = 1;
  // swizzling
  int per_phase_a = swizzle_->get_per_phase(layout_a);
  int max_phase_a = swizzle_->get_max_phase(layout_a);
  int step_a0   = is_a_row ? stride_rep_k : stride_rep_m;
  int num_ptr_a = std::max(2 * per_phase_a * max_phase_a / step_a0, 1);
  int per_phase_b = swizzle_->get_per_phase(layout_b);
  int max_phase_b = swizzle_->get_max_phase(layout_b);
  int step_b0   = is_b_row ? stride_rep_n : stride_rep_k;
  int num_ptr_b = std::max(2 * per_phase_b * max_phase_b / step_b0, 1);


  // max_phase_a = 4;
  // vec_a = 8;
  // std::cout << per_phase_a << " " << max_phase_a << " " << step_a0 << " " << num_ptr_a << " " << stride_am << " " << stride_ak << " " << stride_a0 << " " << stride_a1 << std::endl;
  // std::cout << vec_a << " " << vec_b << std::endl;

  /* --------------------------------- */
  /* --- pre-compute pointer lanes --- */
  /* --------------------------------- */
  BasicBlock* curr_bb = builder_->GetInsertBlock();
  BasicBlock* entry = &curr_bb->getParent()->getEntryBlock();
  if(entry != curr_bb)
    builder_->SetInsertPoint(entry->getTerminator());
  Value* off_a0 = is_a_row ? offset_a_k_[layout_c] : offset_a_m_[layout_c];
  Value* off_a1 = is_a_row ? offset_a_m_[layout_c] : offset_a_k_[layout_c];
  Value* phase_a = urem(udiv(off_a1, i32(per_phase_a)), i32(max_phase_a));
  std::vector<Value*> off_a(num_ptr_a);
  for(int i = 0; i < num_ptr_a; i++){
    Value* off_a0i = add(off_a0, i32(i*(is_a_row?4:stride_rep_m)));
    off_a0i = exact_udiv(off_a0i, i32(vec_a));
    off_a0i = xor_(off_a0i, phase_a);
    off_a0i = mul(off_a0i, i32(vec_a));
    off_a[i] = add(mul(off_a0i, i32(stride_a0)), mul(off_a1, i32(stride_a1)));
  }
  Value* off_b0 = is_b_row ? offset_b_n_[layout_c] : offset_b_k_[layout_c];
  Value* off_b1 = is_b_row ? offset_b_k_[layout_c] : offset_b_n_[layout_c];
  Value* phase_b = urem(udiv(off_b1, i32(per_phase_b)), i32(max_phase_b));
  std::vector<Value*> off_b(num_ptr_b);
  for(int i = 0; i < num_ptr_b; i++){
    Value* off_b0i = add(off_b0, i32(i*(is_b_row?stride_rep_n:4)));
    off_b0i = udiv(off_b0i, i32(vec_b));
    off_b0i = xor_(off_b0i, phase_b);
    off_b0i = mul(off_b0i, i32(vec_b));
    off_b[i] = add(mul(off_b0i, i32(stride_b0)), mul(off_b1, i32(stride_b1)));
  }
  builder_->SetInsertPoint(curr_bb);

  /* --------------------------------- */
  /* ---       MMA intrinsic       --- */
  /* --------------------------------- */
  Type *f16x2_ty = vec_ty(f16_ty, 2);
  Type *ret_ty = StructType::get(*ctx_, {f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty});
  std::vector<Type*> arg_ty = {f16x2_ty, f16x2_ty, f16x2_ty, f16x2_ty,
                               f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty, f32_ty};
  InlineAsm *mma = InlineAsm::get(FunctionType::get(ret_ty, arg_ty, false),
                                             " mma.sync.aligned.m8n8k4."
                                             + std::string(is_a_row ? "row" : "col")
                                             + "."
                                             + std::string(is_b_row ? "row" : "col")
                                             + ".f32.f16.f16.f32 "
                                             "{$0, $1, $2, $3, $4, $5, $6, $7}, "
                                             "{$8, $9}, "
                                             "{$10, $11}, "
                                             "{$0, $1, $2, $3, $4, $5, $6, $7};", "=f,=f,=f,=f,=f,=f,=f,=f,r,r,r,r,0,1,2,3,4,5,6,7", false);


  std::vector<Value*> ptr_a(num_ptr_a);
  std::vector<Value*> ptr_b(num_ptr_b);
  std::map<std::pair<int, int>, std::pair<Value*, Value*>> has, hbs;
  for(int i = 0; i < num_ptr_a; i++)
    ptr_a[i] = gep(shmems_[A], off_a[i]);
  for(int i = 0; i < num_ptr_b; i++)
    ptr_b[i] = gep(shmems_[B], off_b[i]);


  // initialize accumulators
  std::vector<Value*> acc;
  for(indices_t idx: idxs_.at(C))
    acc.push_back(vals_[D][idx]);

  unsigned num_m = layout_c->rep(0) * shape_c[0] / layout_c->shape_per_cta(0);
  unsigned num_n = layout_c->rep(1) * shape_c[1] / layout_c->shape_per_cta(1);

  // create mma & unpack result
  auto call_mma = [&](unsigned m, unsigned n, unsigned K) {
    auto ha = has[{m, K}];
    auto hb = hbs[{n, K}];
    // arguments
    std::vector<size_t> idx = {
      (m*2 + 0) + (n*4 + 0)*num_m, (m*2 + 0) + (n*4 + 1)*num_m,
      (m*2 + 1) + (n*4 + 0)*num_m, (m*2 + 1) + (n*4 + 1)*num_m,
      (m*2 + 0) + (n*4 + 2)*num_m, (m*2 + 0) + (n*4 + 3)*num_m,
      (m*2 + 1) + (n*4 + 2)*num_m, (m*2 + 1) + (n*4 + 3)*num_m
    };
    std::vector<Value*> args = {ha.first, ha.second, hb.first, hb.second};
    for(unsigned i = 0; i < 8; i++)
      args.push_back(acc[idx[i]]);
    // execute mma
    Value *nc = call(mma, args);
    // unpack
    for(unsigned i = 0; i < 8; i++)
      acc[idx[i]] = extract_val(nc, {i});
  };

  ir::phi_node* phiA = dynamic_cast<ir::phi_node*>(A);
  ir::phi_node* phiB = dynamic_cast<ir::phi_node*>(B);

  // Cache lds value. If values are prefetched, create phi node
  // @param inc: incoming block (0 = header, 1 = loop)
  auto register_lds =
    [&](decltype(has)& vals, int m, int K, int inc, Value* val0, Value *val1, bool is_prefetch) {
      if (K == 0 && is_prefetch) {
        ir::basic_block* inc_block = phiA->get_incoming_block(inc);
        lazy_phi_incs_.push_back(std::make_tuple((PHINode*)vals[{m, K}].first, val0, inc_block));
        lazy_phi_incs_.push_back(std::make_tuple((PHINode*)vals[{m, K}].second, val1, inc_block));
      } else
        vals[{m, K}] = {val0, val1};
  };

  auto load_a = [&](int m, int K, int inc, bool is_prefetch) {
    int offidx = (is_a_row ? K/4 : m) % num_ptr_a;
    Value* ptra;
    if(K==0 && is_prefetch){
      if(inc == 0)
        ptra = gep(shared_pre_ptr_[layout_a], off_a[offidx]);
      else
        ptra = gep(shared_next_ptr_[layout_a], off_a[offidx]);
    }
    else
      ptra = ptr_a[offidx];
    int step_am = is_a_row ? m : m / (num_ptr_a)*(num_ptr_a);
    int step_ak = is_a_row ? K / (num_ptr_a*vec_a)*(num_ptr_a*vec_a) : K;
    Value* pa =  gep(ptra, i32(step_am*stride_rep_m*stride_am + step_ak*stride_ak));
    Value* ha = load(bit_cast(pa, ptr_ty(vec_ty(i32_ty, vec_a/2), 3)));
    // record lds that needs to be moved
    if (K == 0 && inc == 1 && is_prefetch)
      prefetch_latch_to_bb_[phiA->get_incoming_value(1)].push_back(ha);
    Value *ha00 = bit_cast(extract_elt(ha, i32(0)), f16x2_ty);
    Value *ha01 = bit_cast(extract_elt(ha, i32(1)), f16x2_ty);
    register_lds(has, m, K, inc, ha00, ha01, is_prefetch);
    if(vec_a > 4){
      Value *ha10 = bit_cast(extract_elt(ha, i32(2)), f16x2_ty);
      Value *ha11 = bit_cast(extract_elt(ha, i32(3)), f16x2_ty);
      if(is_a_row)
        register_lds(has, m, K+4, inc, ha10, ha11, is_prefetch);
      else
        register_lds(has, m+1, K, inc, ha10, ha11, is_prefetch);
    }
  };

  auto load_b = [&](int n, int K, int inc, bool is_prefetch) {
    int offidx = (is_b_row? n : K/4) % num_ptr_b;
    Value* ptrb;
    if(K==0 && is_prefetch){
      if(inc == 0)
        ptrb = gep(shared_pre_ptr_[layout_b], off_b[offidx]);
      else
        ptrb = gep(shared_next_ptr_[layout_b], off_b[offidx]);
    } else
      ptrb = ptr_b[offidx];

    int stepbn = is_b_row ? n / (num_ptr_b)*(num_ptr_b) : n;
    int stepbk = is_b_row ? K : K / (num_ptr_b*vec_b)*(num_ptr_b*vec_b);
    Value* pb =   gep(ptrb, i32(stepbn*stride_rep_n*stride_bn + stepbk*stride_bk));
    Value* hb =   load(bit_cast(pb, ptr_ty(vec_ty(i32_ty, vec_b/2), 3)));
    // record lds that needs to be moved
    if (K == 0 && inc == 1 && is_prefetch)
      prefetch_latch_to_bb_[phiB->get_incoming_value(1)].push_back(hb);
    Value *hb00 = bit_cast(extract_elt(hb, i32(0)), f16x2_ty);
    Value *hb01 = bit_cast(extract_elt(hb, i32(1)), f16x2_ty);
    register_lds(hbs, n, K, inc, hb00, hb01, is_prefetch);
    if(vec_b > 4){
      Value *hb10 = bit_cast(extract_elt(hb, i32(2)), f16x2_ty);
      Value *hb11 = bit_cast(extract_elt(hb, i32(3)), f16x2_ty);
      if(is_b_row)
        register_lds(hbs, n+1, K, inc, hb10, hb11, is_prefetch);
      else
        register_lds(hbs, n, K+4, inc, hb10, hb11, is_prefetch);
    }

  };

  // update accumulators
  if (C->is_prefetched()) {
    // create phis
    builder_->SetInsertPoint(curr_bb->getFirstNonPHI());
    for (unsigned m = 0; m < num_m/2; m += is_a_row?1:2) {
      has[{m, 0}].first = phi(f16x2_ty, 2);
      has[{m, 0}].second = phi(f16x2_ty, 2);
      if (!is_a_row && vec_a>4) {
        has[{m+1, 0}].first = phi(f16x2_ty, 2);
        has[{m+1, 0}].second = phi(f16x2_ty, 2);
      }
    }
    for (unsigned n = 0; n < num_n/2; n += is_b_row?2:1) {
      hbs[{n, 0}].first = phi(f16x2_ty, 2);
      hbs[{n, 0}].second = phi(f16x2_ty, 2);
      if (is_b_row && vec_b>4) {
        hbs[{n+1, 0}].first = phi(f16x2_ty, 2);
        hbs[{n+1, 0}].second = phi(f16x2_ty, 2);
      }
    }

    // insert prefetched lds at the end of loop header
    builder_->SetInsertPoint(bbs_[phiA->get_incoming_block(0)]->getTerminator());
    for (unsigned m = 0; m < num_m/2; m += is_a_row?1:2)
      load_a(m, 0, 0, true);
    for (unsigned n = 0; n < num_n/2; n += is_b_row?2:1)
      load_b(n, 0, 0, true);

    // update accumulators
    builder_->SetInsertPoint(curr_bb);
    for (unsigned K = 0; K < NK; K += 4) {
      int NEXTK = (K + 4) % NK;
      // prefetch A
      for (unsigned m = 0; m < num_m/2; m+=is_a_row?1:2)
        load_a(m, NEXTK, 1, true);
      // prefetch B
      for (unsigned n = 0; n < num_n/2; n+=is_b_row?2:1)
        load_b(n, NEXTK, 1, true);
      // tensor core ops
      for(unsigned m = 0; m < num_m/2; m++)
      for(unsigned n = 0; n < num_n/2; n++){
        call_mma(m, n, K);
      }
    }
  } else { // not prefetched
    for(unsigned K = 0; K < NK; K += 4)
    for(unsigned m = 0; m < num_m/2; m++)
    for(unsigned n = 0; n < num_n/2; n++) {
      if(has.find({m, K}) == has.end())
        load_a(m, K, /*inc*/0, /*is_prefetch*/false);
      if(hbs.find({n, K}) == hbs.end())
        load_b(n, K, /*inc*/0, /*is_prefetch*/false);
      call_mma(m, n, K);
    }
  }

  // write back accumulators
  for(size_t i = 0; i < idxs_.at(C).size(); i++)
    vals_[C][idxs_[C][i]] = acc[i];
}

namespace {
class mma16816_smem_loader {
public:
  mma16816_smem_loader(int wpt, std::vector<int> order, int k_order,
                       std::vector<unsigned> tile_shape,
                       std::vector<int> instr_shape, std::vector<int> mat_shape,
                       int per_phase, int max_phase, int dtsize, Builder *builder,
                       adder add, multiplier mul, geper gep)
                      : wpt_(wpt), order_(order), k_order_(k_order), tile_shape_(tile_shape),
                        instr_shape_(instr_shape), mat_shape_(mat_shape),
                        per_phase_(per_phase), max_phase_(max_phase), dtsize_(dtsize), builder_(builder),
                        add(add), mul(mul), gep(gep) {
    // compute compile-time constant variables & types
    c_mat_shape_ = mat_shape[order[0]];
    s_mat_shape_ = mat_shape[order[1]];

    c_stride_ = tile_shape[order[1]];
    s_stride_ = tile_shape[order[0]];

    // rule: k must be the fast-changing axis
    need_trans_ = k_order_ != order_[0];
    can_use_ldmatrix_ = dtsize == 2 || (!need_trans_);

    // we need more pointers at the fast-changing axis,
    if (can_use_ldmatrix_)
      num_ptr_ = tile_shape[order[0]] / (order[0] == k_order? 1 : wpt) / instr_shape[order[0]];
    else // warning: this only works for tf32 & need transpose
      num_ptr_ = tile_shape[order[0]] / wpt / mat_shape[order[0]];
    num_ptr_ = std::max<int>(num_ptr_, 2);

    // special rule for i8/u8, 4 ptrs for each matrix
    if (!can_use_ldmatrix_ && dtsize_ == 1)
      num_ptr_ *= 4;

    // load_v4 stride (in num of mats)
    int load_stride_in_mat[2];
    load_stride_in_mat[k_order] = 2; // instr_shape[k_order] / mat_shape[k_order], always 2
    load_stride_in_mat[k_order^1] = wpt * (instr_shape[k_order^1] / mat_shape[k_order^1]);
    p_load_stride_in_mat_ = load_stride_in_mat[order[0]];
    // stride in mat, used by load_v4
    s_mat_stride_ = load_stride_in_mat[order[1]] / (instr_shape[order[1]]/mat_shape[order[1]]);
  }

  std::vector<Value*> compute_offs(Value *warp_off, Value *lane) {
    // TODO: this needs to be moved to constructor (and extracted to arr_order)
    mat_arr_stride_  = (k_order_ == 1) ? 1 : wpt_;
    warp_off_stride_ = instr_shape_[k_order_^1] / mat_shape_[k_order_^1];
    // start matrix logic offset (rename it as base_mat_off?)
    Value *mat_off[2] = {nullptr, nullptr};

    if (can_use_ldmatrix_) {
      // c: lane idx inside a group (a group is a collection of 8 contiguous threads)
      // s: group idx (0,1,2,3) inside a warp
      Value *c = urem(lane, i32(8));
      Value *s = udiv(lane, i32(8));
      // We can decompose s => s_0, s_1...
      Value *s0 = urem(s, i32(2));
      Value *s1 = udiv(s, i32(2));

      // We use different orders for a & b for better performance.
      Value *k_mat_arr  = (k_order_ == 1) ? s1 : s0;
      Value *nk_mat_arr = (k_order_ == 1) ? s0 : s1;
      mat_off[k_order_^1] = add(mul(warp_off,   i32(warp_off_stride_)),
                                mul(nk_mat_arr, i32(mat_arr_stride_)));
      mat_off[k_order_]   = k_mat_arr;
      // physical offset (before swizzling)
      Value *c_mat_off = mat_off[order_[0]];
      Value *s_mat_off = mat_off[order_[1]];
      // offset inside a matrix
      Value *s_off_in_mat = c;

      std::vector<Value*> offs(num_ptr_);
      Value *phase = urem(udiv(s_off_in_mat, i32(per_phase_)), i32(max_phase_));
      // pre-compute strided offset
      Value *s_off = add(s_off_in_mat, mul(s_mat_off, i32(s_mat_shape_)));
      for (int i=0; i < num_ptr_; ++i) {
        Value *c_mat_off_i = add(c_mat_off, i32(i*p_load_stride_in_mat_));
        c_mat_off_i = xor_(c_mat_off_i, phase); // smem swizzle
        offs[i] = add(mul(c_mat_off_i, i32(c_mat_shape_)), mul(s_off, i32(s_stride_)));
      }
      return offs;
    } else if (dtsize_ == 4 && need_trans_) {
      // load tf32 matrices with lds32
      Value *c_off_in_mat = udiv(lane, i32(4)); // 4 = mat_shape[order[1]]
      Value *s_off_in_mat = urem(lane, i32(4)); //

      Value *phase = urem(udiv(s_off_in_mat, i32(per_phase_)), i32(max_phase_));
      std::vector<Value*> offs(num_ptr_);
      for (int mat = 0; mat < 4; ++mat) { // loads 4 mats each time
        int k_mat_arr_int  = (k_order_ == 1) ? mat/2 : mat%2;
        int nk_mat_arr_int = (k_order_ == 1) ? mat%2 : mat/2;
        if (k_mat_arr_int > 0) // we don't need pointers for k
          continue;
        Value *k_mat_arr  = i32(k_mat_arr_int);
        Value *nk_mat_arr = i32(nk_mat_arr_int);
        // physical offset (before swizzling)
        Value *c_mat_off = add(mul(warp_off, i32(warp_off_stride_)),
                               mul(nk_mat_arr, i32(mat_arr_stride_)));
        Value *s_mat_off = k_mat_arr; // always 0?
        Value *s_off = add(s_off_in_mat, mul(s_mat_off, i32(s_mat_shape_)));
        // FIXME: (k_order_ == 1?) is really dirty hack
        for (int i = 0; i < num_ptr_/2; ++i) {
          Value *c_mat_off_i = add(c_mat_off, i32(i*p_load_stride_in_mat_*(k_order_ == 1?1:2)));
          c_mat_off_i = xor_(c_mat_off_i, phase);
          Value *c_off = add(c_off_in_mat, mul(c_mat_off_i, i32(c_mat_shape_)));
          // TODO: move this out of the loop
          c_off = urem(c_off, i32(tile_shape_[order_[0]]));
          s_off = urem(s_off, i32(tile_shape_[order_[1]]));
          offs[2*i + nk_mat_arr_int] = add(c_off, mul(s_off, i32(s_stride_)));
        }
      }
      return offs;
      // throw std::runtime_error("not implemented");
    } else if (dtsize_ == 1 && need_trans_) {
      // load i8/u8 matrices with lds8
      Value *c_off_in_mat = udiv(lane, i32(4)); //
      Value *s_off_in_mat = mul(urem(lane, i32(4)), i32(4)); // each thread load 4 cols

      // Value *phase = urem(udiv(s_off_in_mat, i32(per_phase_)), i32(max_phase_));
      std::vector<Value*> offs(num_ptr_);
      for (int mat = 0; mat < 4; ++mat) { // loads 4 mats each time
        int k_mat_arr_int  = (k_order_ == 1) ? mat/2 : mat%2;
        int nk_mat_arr_int = (k_order_ == 1) ? mat%2 : mat/2;
        if (k_mat_arr_int > 0) // we don't need pointers for k
          continue;
        Value *k_mat_arr  = i32(k_mat_arr_int);
        Value *nk_mat_arr = i32(nk_mat_arr_int);
        // physical offset (before swizzling)
        Value *c_mat_off = add(mul(warp_off, i32(warp_off_stride_)),
                               mul(nk_mat_arr, i32(mat_arr_stride_)));
        Value *s_mat_off = k_mat_arr; // always 0?

        for (int loadx4_off = 0; loadx4_off < num_ptr_/8; ++loadx4_off) {
          for (int elem_off = 0; elem_off < 4; ++elem_off) {
            int ptr_off = loadx4_off*8 + nk_mat_arr_int*4 + elem_off;

            Value *c_mat_off_i = add(c_mat_off, i32(loadx4_off*p_load_stride_in_mat_*(k_order_ == 1?1:2)));
            Value *s_off_in_mat_elem = add(s_off_in_mat, i32(elem_off));

            // disable swizzling ...
            // Value *phase = urem(udiv(s_off_in_mat, i32(per_phase_)), i32(max_phase_));
            // c_mat_off_i = xor_(c_mat_off_i, phase);

            Value *c_off = add(c_off_in_mat, mul(c_mat_off_i, i32(c_mat_shape_)));
            Value *s_off = add(s_off_in_mat_elem, mul(s_mat_off, i32(s_mat_shape_)));
            // To prevent out-of-bound access when the tile is too small
            c_off = urem(c_off, i32(tile_shape_[order_[0]]));
            s_off = urem(s_off, i32(tile_shape_[order_[1]]));
            offs[ptr_off] = add(c_off, mul(s_off, i32(s_stride_)));
          }
        }
      }
      return offs;
    } else
      throw std::runtime_error("invalid smem load config");
  }

  std::tuple<Value*, Value*, Value*, Value*>
  load_x4(int mat0, int mat1, int inc, bool is_prefetch, ir::phi_node *pn,
          Value *pre_ptr, Value *next_ptr, std::vector<Value*> &off, std::vector<Value*> &ptrs,
          FunctionType *ldmatrix_ty, Type *smem_ptr_ty,
          std::map<ir::value*, std::vector<Value*>> &prefetch_latch_to_bb_) {
    assert(mat0 % 2 == 0 && mat1 % 2 == 0 && "smem matrix load must be aligned");
    int mat_idx[2] = {mat0, mat1};
    int k = mat_idx[k_order_];

    int ptr_idx = -1;
    if (can_use_ldmatrix_)
      ptr_idx = mat_idx[order_[0]] / (instr_shape_[order_[0]] / mat_shape_[order_[0]]);
    else if (dtsize_ == 4 && need_trans_) // tf32 & trans
      ptr_idx = mat_idx[order_[0]];
    else // i8 & trans
      ptr_idx = mat_idx[order_[0]] * 4;

    auto get_ptr = [&](int idx) -> Value* {
      Value *ptr = nullptr;
      if (k == 0 && is_prefetch) {
        if (inc == 0)
          ptr = bit_cast(gep(pre_ptr, off.at(idx)), smem_ptr_ty);
        else
          ptr = bit_cast(gep(next_ptr, off.at(idx)), smem_ptr_ty);
      } else
        ptr = ptrs.at(idx);
      return ptr;
    };
    Value *ptr = get_ptr(ptr_idx);

    Value *res_v4 = nullptr;
    if (can_use_ldmatrix_) {
      std::string trans = need_trans_ ? ".trans" : "";
      // the offset (in byte) on the strided axis is a constant
      int s_offset = mat_idx[order_[1]] * (s_mat_stride_*s_mat_shape_) * s_stride_ * dtsize_;
      InlineAsm *ld_fn = InlineAsm::get(ldmatrix_ty,
                                        "ldmatrix.sync.aligned.m8n8.x4" + trans + ".shared.b16 "
                                        "{$0, $1, $2, $3}, "
                                        "[$4 + " + std::to_string(s_offset) + "];",
                                        "=r,=r,=r,=r,r", true);
      assert(ptr);
      res_v4 = call(ldmatrix_ty, ld_fn, {ptr});
      if (k == 0 && inc == 1 && is_prefetch)
        prefetch_latch_to_bb_[pn->get_incoming_value(1)].push_back(res_v4);
      return {extract_val(res_v4, std::vector<unsigned>{0}),
              extract_val(res_v4, std::vector<unsigned>{1}),
              extract_val(res_v4, std::vector<unsigned>{2}),
              extract_val(res_v4, std::vector<unsigned>{3})};
    } else if (dtsize_ == 4 && need_trans_) { // use lds.32 to load tf32 matrices
      Value *ptr2 = get_ptr(ptr_idx+1);
      assert(s_mat_stride_ == 1);
      int s_offset_elem = mat_idx[order_[1]] * (s_mat_stride_*s_mat_shape_) * s_stride_;
      int s_offset_arr_elem = 1 * (s_mat_stride_*s_mat_shape_) * s_stride_;
      Value *elem0, *elem1, *elem2, *elem3;
      if (k_order_ == 1) {
        elem0 = load(gep(ptr,  i32(s_offset_elem)));
        elem1 = load(gep(ptr2, i32(s_offset_elem)));
        elem2 = load(gep(ptr,  i32(s_offset_elem + s_offset_arr_elem)));
        elem3 = load(gep(ptr2, i32(s_offset_elem + s_offset_arr_elem)));
      } else { // for b (k first)
        elem0 = load(gep(ptr,  i32(s_offset_elem)));
        elem2 = load(gep(ptr2, i32(s_offset_elem)));
        elem1 = load(gep(ptr,  i32(s_offset_elem + s_offset_arr_elem)));
        elem3 = load(gep(ptr2, i32(s_offset_elem + s_offset_arr_elem)));
      }
      if (k == 0 && inc == 1 && is_prefetch) {
        prefetch_latch_to_bb_[pn->get_incoming_value(1)].push_back(elem0);
        prefetch_latch_to_bb_[pn->get_incoming_value(1)].push_back(elem1);
        prefetch_latch_to_bb_[pn->get_incoming_value(1)].push_back(elem2);
        prefetch_latch_to_bb_[pn->get_incoming_value(1)].push_back(elem3);
      }
      return {elem0, elem1, elem2, elem3};
    } else if (dtsize_ == 1 && need_trans_) { // use lds.8 to load i8/u8 matrices
      Value *ptr00 = get_ptr(ptr_idx);
      Value *ptr01 = get_ptr(ptr_idx+1);
      Value *ptr02 = get_ptr(ptr_idx+2);
      Value *ptr03 = get_ptr(ptr_idx+3);

      Value *ptr10 = get_ptr(ptr_idx+4);
      Value *ptr11 = get_ptr(ptr_idx+5);
      Value *ptr12 = get_ptr(ptr_idx+6);
      Value *ptr13 = get_ptr(ptr_idx+7);

      assert(s_mat_stride_ == 1);
      int s_offset_elem = mat_idx[order_[1]] * (s_mat_stride_*s_mat_shape_) * s_stride_;
      int s_offset_arr_elem = 1 * (s_mat_stride_*s_mat_shape_) * s_stride_;

      Value *i8v4_elems[4];
      Value *i32_elems[4];
      for (int i=0; i<4; ++i)
        i8v4_elems[i] = UndefValue::get(vec_ty(i8_ty, 4));

      Value *elem00, *elem01, *elem02, *elem03;
      Value *elem10, *elem11, *elem12, *elem13;
      Value *elem20, *elem21, *elem22, *elem23;
      Value *elem30, *elem31, *elem32, *elem33;
      Value *i8_elems[4*4];
      if (k_order_ == 1) { //
        i8_elems[0*4 + 0] = load(gep(ptr00, i32(s_offset_elem)));
        i8_elems[0*4 + 1] = load(gep(ptr01, i32(s_offset_elem)));
        i8_elems[0*4 + 2] = load(gep(ptr02, i32(s_offset_elem)));
        i8_elems[0*4 + 3] = load(gep(ptr03, i32(s_offset_elem)));

        assert(i8_elems[0*4 + 0]->getType()->isIntegerTy(8));

        i8_elems[1*4 + 0] = load(gep(ptr10, i32(s_offset_elem)));
        i8_elems[1*4 + 1] = load(gep(ptr11, i32(s_offset_elem)));
        i8_elems[1*4 + 2] = load(gep(ptr12, i32(s_offset_elem)));
        i8_elems[1*4 + 3] = load(gep(ptr13, i32(s_offset_elem)));

        i8_elems[2*4 + 0] = load(gep(ptr00, i32(s_offset_elem + s_offset_arr_elem)));
        i8_elems[2*4 + 1] = load(gep(ptr01, i32(s_offset_elem + s_offset_arr_elem)));
        i8_elems[2*4 + 2] = load(gep(ptr02, i32(s_offset_elem + s_offset_arr_elem)));
        i8_elems[2*4 + 3] = load(gep(ptr03, i32(s_offset_elem + s_offset_arr_elem)));

        i8_elems[3*4 + 0] = load(gep(ptr10, i32(s_offset_elem + s_offset_arr_elem)));
        i8_elems[3*4 + 1] = load(gep(ptr11, i32(s_offset_elem + s_offset_arr_elem)));
        i8_elems[3*4 + 2] = load(gep(ptr12, i32(s_offset_elem + s_offset_arr_elem)));
        i8_elems[3*4 + 3] = load(gep(ptr13, i32(s_offset_elem + s_offset_arr_elem)));

        for (int m=0; m<4; ++m) {
          for (int e=0; e<4; ++e)
            i8v4_elems[m] = insert_elt(i8v4_elems[m], i8_elems[m*4 + e], e);
          i32_elems[m] = bit_cast(i8v4_elems[m], i32_ty);
        }
      } else { // for b (k first)
        i8_elems[0*4 + 0] = load(gep(ptr00, i32(s_offset_elem)));
        i8_elems[0*4 + 1] = load(gep(ptr01, i32(s_offset_elem)));
        i8_elems[0*4 + 2] = load(gep(ptr02, i32(s_offset_elem)));
        i8_elems[0*4 + 3] = load(gep(ptr03, i32(s_offset_elem)));

        assert(i8_elems[0*4 + 0]->getType()->isIntegerTy(8));

        i8_elems[2*4 + 0] = load(gep(ptr10, i32(s_offset_elem)));
        i8_elems[2*4 + 1] = load(gep(ptr11, i32(s_offset_elem)));
        i8_elems[2*4 + 2] = load(gep(ptr12, i32(s_offset_elem)));
        i8_elems[2*4 + 3] = load(gep(ptr13, i32(s_offset_elem)));

        i8_elems[1*4 + 0] = load(gep(ptr00, i32(s_offset_elem + s_offset_arr_elem)));
        i8_elems[1*4 + 1] = load(gep(ptr01, i32(s_offset_elem + s_offset_arr_elem)));
        i8_elems[1*4 + 2] = load(gep(ptr02, i32(s_offset_elem + s_offset_arr_elem)));
        i8_elems[1*4 + 3] = load(gep(ptr03, i32(s_offset_elem + s_offset_arr_elem)));

        i8_elems[3*4 + 0] = load(gep(ptr10, i32(s_offset_elem + s_offset_arr_elem)));
        i8_elems[3*4 + 1] = load(gep(ptr11, i32(s_offset_elem + s_offset_arr_elem)));
        i8_elems[3*4 + 2] = load(gep(ptr12, i32(s_offset_elem + s_offset_arr_elem)));
        i8_elems[3*4 + 3] = load(gep(ptr13, i32(s_offset_elem + s_offset_arr_elem)));

        for (int m=0; m<4; ++m) {
          for (int e=0; e<4; ++e)
            i8v4_elems[m] = insert_elt(i8v4_elems[m], i8_elems[m*4 + e], e);
          i32_elems[m] = bit_cast(i8v4_elems[m], i32_ty);
        }
      }
      if (k == 0 && inc == 1 && is_prefetch) {
        for (int m = 0; m < 4; ++m)
        for (int e = 0; e < 4; ++e)
          prefetch_latch_to_bb_[pn->get_incoming_value(1)].push_back(i8_elems[m*4 + e]);
      }
      return {i32_elems[0], i32_elems[1], i32_elems[2], i32_elems[3]};
    } else
      throw std::runtime_error("invalid smem load");
  }

  int get_num_ptr() const { return num_ptr_; }

private:
  int wpt_;
  std::vector<int> order_;
  int k_order_;
  std::vector<unsigned> tile_shape_;
  std::vector<int> instr_shape_;
  std::vector<int> mat_shape_;
  int per_phase_, max_phase_;
  int dtsize_;

  // generated
  int c_mat_shape_, s_mat_shape_;
  int c_stride_, s_stride_;
  // p_: on the pointer axis
  int p_load_stride_in_mat_;
  int s_mat_stride_;
  // stride when moving to next not-k mat
  int warp_off_stride_;
  int mat_arr_stride_; // matrix arrangement (inside a load) stride
  bool need_trans_, can_use_ldmatrix_;
  int num_ptr_;

  Builder *builder_;
  adder add;
  multiplier mul;
  geper gep;
};
}

/**
 * \brief Code Generation for `mma.16816` (A100)
 */
//TODO: clean-up
void generator::visit_mma16816(ir::dot_inst* C, ir::value *A, ir::value *B, ir::value *D, unsigned NK) {
  const std::vector<unsigned>& shapes = C->get_type()->get_block_shapes();
  std::map<std::vector<Value*>, std::vector<Value*>> fcs;
  for(indices_t idx: idxs_.at(C)){
    std::vector<Value*> key(idx.size() - 2);
    std::copy(idx.begin() + 2, idx.end(), key.begin());
    fcs[key].push_back(vals_[D][idx]);
  };
  auto shape_a = A->get_type()->get_block_shapes();
  auto shape_b = B->get_type()->get_block_shapes();
  auto ord_a = layouts_->get(A)->get_order();
  if(C->is_trans_a()){
    std::swap(ord_a[0], ord_a[1]);
    std::swap(shape_a[0], shape_a[1]);
  }
  auto ord_b = layouts_->get(B)->get_order();
  if(C->is_trans_b()){
    std::swap(ord_b[0], ord_b[1]);
    std::swap(shape_b[0], shape_b[1]);
  }
  NK = shape_a[1];
  analysis::mma_layout* layout = layouts_->get(C)->to_mma();

  std::vector<int> mma_instr_shape = layout->get_mma_instr_shape();
  const int mma_instr_m = mma_instr_shape[0];
  const int mma_instr_n = mma_instr_shape[1];
  const int mma_instr_k = mma_instr_shape[2];

  std::vector<int> mat_shape = layout->get_mma_mat_shape();
  const int mat_shape_m = mat_shape[0];
  const int mat_shape_n = mat_shape[1];
  const int mat_shape_k = mat_shape[2];


  const int num_rep_m = shapes[0] / layout->shape_per_cta(0);
  const int num_rep_n = shapes[1] / layout->shape_per_cta(1);
  const int num_rep_k = std::max<int>(NK/mma_instr_k, 1);

  // floating point types
  Type *fp32_ty = f32_ty;
  Type *fp16x2_ty = vec_ty(f16_ty, 2);
  Type *bf16x2_ty = vec_ty(bf16_ty, 2);
  Type *fp16x2_pack4_ty = StructType::get(*ctx_, std::vector<llvm::Type*>{fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty});
  Type *bf16x2_pack4_ty = StructType::get(*ctx_, std::vector<llvm::Type*>{bf16x2_ty, bf16x2_ty, bf16x2_ty, bf16x2_ty});
  Type *fp32_pack4_ty = StructType::get(*ctx_, std::vector<llvm::Type*>{fp32_ty, fp32_ty, fp32_ty, fp32_ty});
  // integer types
  Type *i8x4_ty = vec_ty(i8_ty, 4);
  Type *i8x4_pack4_ty = StructType::get(*ctx_, std::vector<llvm::Type*>{i8x4_ty, i8x4_ty, i8x4_ty, i8x4_ty});
  Type *i32_pack4_ty  = StructType::get(*ctx_, std::vector<llvm::Type*>{i32_ty, i32_ty, i32_ty, i32_ty});


  FunctionType *ldmatrix_ty = nullptr;
  FunctionType *mma_ty = nullptr;
  Type *phi_ty = nullptr;
  Type *smem_ptr_ty = nullptr;

  ir::type *A_ir_ty = A->get_type()->get_scalar_ty();
  ir::type *B_ir_ty = B->get_type()->get_scalar_ty();
  if (A_ir_ty->is_fp16_ty() && B_ir_ty->is_fp16_ty()) {
    mma_ty = FunctionType::get(fp32_pack4_ty, std::vector<llvm::Type*>{fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty}, false);
    smem_ptr_ty = ptr_ty(f16_ty, 3);
    ldmatrix_ty = FunctionType::get(fp16x2_pack4_ty, std::vector<llvm::Type*>{smem_ptr_ty}, false);
    phi_ty = fp16x2_ty;
  } else if (A_ir_ty->is_bf16_ty() && B_ir_ty->is_bf16_ty()) {
    mma_ty = FunctionType::get(fp32_pack4_ty, std::vector<llvm::Type*>{bf16x2_ty, bf16x2_ty, bf16x2_ty, bf16x2_ty, bf16x2_ty, bf16x2_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty}, false);
    smem_ptr_ty = ptr_ty(bf16_ty, 3);
    ldmatrix_ty = FunctionType::get(bf16x2_pack4_ty, std::vector<llvm::Type*>{smem_ptr_ty}, false);
    phi_ty = bf16x2_ty;
  } else if (A_ir_ty->is_fp32_ty() && B_ir_ty->is_fp32_ty()) {
    mma_ty = FunctionType::get(fp32_pack4_ty, std::vector<llvm::Type*>{fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty}, false);
    smem_ptr_ty = ptr_ty(fp32_ty, 3);
    ldmatrix_ty = FunctionType::get(fp32_pack4_ty, std::vector<llvm::Type*>{smem_ptr_ty}, false);
    phi_ty = fp32_ty;
  } else if (A_ir_ty->is_integer_ty(8) && B_ir_ty->is_integer_ty(8)) {
    // FIXME: We should use i8 here (but nvptx will generate extra casts when using i8)
    mma_ty = FunctionType::get(i32_pack4_ty, std::vector<llvm::Type*>{i32_ty, i32_ty, i32_ty, i32_ty, i32_ty, i32_ty, i32_ty, i32_ty, i32_ty, i32_ty}, false);
    smem_ptr_ty = ptr_ty(i8_ty, 3);
    ldmatrix_ty = FunctionType::get(i32_pack4_ty, std::vector<llvm::Type*>{smem_ptr_ty}, false);
    phi_ty = i32_ty;
    // mma_ty = FunctionType::get(i32_pack4_ty, std::vector<llvm::Type*>{i8x4_ty, i8x4_ty, i8x4_ty, i8x4_ty, i8x4_ty, i8x4_ty, i32_ty, i32_ty, i32_ty, i32_ty}, false);
    // smem_ptr_ty = ptr_ty(i8_ty, 3);
    // ldmatrix_ty = FunctionType::get(i8x4_pack4_ty, std::vector<llvm::Type*>{smem_ptr_ty}, false);
    // phi_ty = i8x4_ty;
  } else
    throw std::runtime_error("mma16816 data type not supported");

  // left-hand-side values
  std::map<std::pair<unsigned, unsigned>, Value*> ha;
  std::map<std::pair<unsigned, unsigned>, Value*> hb;

  BasicBlock* CurrBB = builder_->GetInsertBlock();
  BasicBlock* FirstBB = &CurrBB->getParent()->getEntryBlock();

  // if true, this will move pointer declarations to the entry basic block
  // not prefetched cases tend to be more limited in resource usage
  // so we don't pre-compute ptrs to save registers
  bool licm_ptrs = C->is_prefetched() && (FirstBB != CurrBB);
  if(licm_ptrs)
    builder_->SetInsertPoint(FirstBB->getTerminator());

  Value* thread = tgt_->get_local_id(mod_, *builder_, 0);
  Value *lane   = urem(thread, i32(32));
  Value *warp   = udiv(thread, i32(32));
  Value *warp_mn = udiv(warp, i32(layout->wpt(0)));
  Value *warp_m  = urem(warp, i32(layout->wpt(0)));
  Value *warp_n  = urem(warp_mn, i32(layout->wpt(1)));
  std::vector<Value *>& fc = fcs.begin()->second;

  size_t dtsize_a = A->get_type()->get_scalar_ty()->get_primitive_size_in_bits() / 8;
  size_t dtsize_b = B->get_type()->get_scalar_ty()->get_primitive_size_in_bits() / 8;

  ir::phi_node* phiA = dynamic_cast<ir::phi_node*>(A);
  ir::phi_node* phiB = dynamic_cast<ir::phi_node*>(B);
  auto register_lds2 =
    [&](std::map<std::pair<unsigned, unsigned>, Value*>& vals, int mn, int k, int inc, Value* val, bool is_prefetch) {
      if (k < 2 && is_prefetch) {
        ir::basic_block* inc_block = phiA->get_incoming_block(inc);
        lazy_phi_incs_.push_back(std::make_tuple((PHINode*)vals[{mn, k}], val, inc_block));
      } else
        vals[{mn, k}] = val;
  };

  // | -> k (row-major), since we have ldmatrix.trans, we only need to change stride
  // v (s0_0(0), s1_0(2), | *num_rep_k
  // m  s0_1(1), s1_1(3)) |  (stride in num of matrices(mat_stride_ak): 2)
  // -----------
  //   *num_rep_m (stride in num of matrices(mat_stride_am): 2*layout->wpt(0))
  std::function<void(int,int,int,bool)> load_a;
  analysis::shared_layout* layout_a = layouts_->get(C->get_operand(0))->to_shared();
  bool is_a_shared = layout_a != nullptr;
  if(is_a_shared) {
    const int per_phase_a = swizzle_->get_per_phase(layout_a);
    const int max_phase_a = swizzle_->get_max_phase(layout_a);
    mma16816_smem_loader a_loader(layout->wpt(0), ord_a, /*k_order*/1, shape_a,
                                  {mma_instr_m, mma_instr_k}, {mat_shape_m, mat_shape_k},
                                  per_phase_a, max_phase_a, dtsize_a, builder_, add, mul, gep);
    std::vector<Value*> off_a = a_loader.compute_offs(warp_m, lane);
    int num_ptr_a = a_loader.get_num_ptr();
    // pointers
    std::vector<Value*> ptrs_a(num_ptr_a);
    if(licm_ptrs)
      builder_->SetInsertPoint(CurrBB);
    for(int i = 0; i < num_ptr_a; i++)
      ptrs_a[i] = bit_cast(gep(shmems_[A], {off_a[i]}), smem_ptr_ty);
    if(licm_ptrs)
      builder_->SetInsertPoint(FirstBB->getTerminator());
    // loading function
    load_a = [&,a_loader,ptrs_a,off_a](int m, int k, int inc, bool is_prefetch) mutable {
      auto [ha0, ha1, ha2, ha3] = a_loader.load_x4(m, k, inc, is_prefetch, phiA, shared_pre_ptr_[layout_a],
                                                  shared_next_ptr_[layout_a], off_a, ptrs_a,
                                                  ldmatrix_ty, smem_ptr_ty, prefetch_latch_to_bb_);
      register_lds2(ha, m,   k,   inc, ha0, is_prefetch);
      register_lds2(ha, m+1, k,   inc, ha1, is_prefetch);
      register_lds2(ha, m,   k+1, inc, ha2, is_prefetch);
      register_lds2(ha, m+1, k+1, inc, ha3, is_prefetch);
    };
  }
  else {
    load_a = [&](int m, int k, int inc, bool is_prefetch) {
      distributed_axis ax_n = axes_.at(a_axes_->get(A, 1));
      int ldm = ax_n.values.size();
      if(ldm != num_rep_k*4)
        throw std::runtime_error("Internal compiler error when trying to fuse matmuls!");
      // std::cout << m << " " << k << std::endl;
      // std::cout << idxs_[A].size() << std::endl;
      // std::cout << (m+1)*ldm + k*2 + 3 << std::endl;
      // int ldm = num_rep_k*4;
      Value* ha0 = UndefValue::get(phi_ty);  // e.g., fp16x2
      Value* ha1 = UndefValue::get(phi_ty);
      Value* ha2 = UndefValue::get(phi_ty);
      Value* ha3 = UndefValue::get(phi_ty);
      ha0 = builder_->CreateInsertElement(ha0, vals_[A][idxs_[A][(m+0)*ldm + k*2 + 0]], i32(0));
      ha0 = builder_->CreateInsertElement(ha0, vals_[A][idxs_[A][(m+0)*ldm + k*2 + 1]], i32(1));
      ha1 = builder_->CreateInsertElement(ha1, vals_[A][idxs_[A][(m+1)*ldm + k*2 + 0]], i32(0));
      ha1 = builder_->CreateInsertElement(ha1, vals_[A][idxs_[A][(m+1)*ldm + k*2 + 1]], i32(1));
      ha2 = builder_->CreateInsertElement(ha2, vals_[A][idxs_[A][(m+0)*ldm + k*2 + 2]], i32(0));
      ha2 = builder_->CreateInsertElement(ha2, vals_[A][idxs_[A][(m+0)*ldm + k*2 + 3]], i32(1));
      ha3 = builder_->CreateInsertElement(ha3, vals_[A][idxs_[A][(m+1)*ldm + k*2 + 2]], i32(0));
      ha3 = builder_->CreateInsertElement(ha3, vals_[A][idxs_[A][(m+1)*ldm + k*2 + 3]], i32(1));
      ha[{m, k}] = ha0;
      ha[{m+1, k}] = ha1;
      ha[{m, k+1}] = ha2;
      ha[{m+1, k+1}] = ha3;
    };
  }


  // | -> n (col-major)
  // v (s0_0(0), | (stride: wpt(1)) | s1_0(2)  | *num_rep_n
  // k  s0_1(1), |                  | s1_1(3)) | (stride in num of matrices(mat_stride_bn): wpt(1))
  // -----------
  //   *num_rep_k (stride in num of matrices(mat_stride_bk): 2)
  analysis::shared_layout* layout_b = layouts_->get(C->get_operand(1))->to_shared();
  const int per_phase_b = swizzle_->get_per_phase(layout_b);
  const int max_phase_b = swizzle_->get_max_phase(layout_b);
  std::vector<int> mma_instr_b{mma_instr_k, mma_instr_n};
  std::vector<int> mat_shape_b{mat_shape_k, mat_shape_n};
  int k_order_b = 0;
  // if(C->is_trans_b()){
    // std::swap(mma_instr_b[0], mma_instr_b[1]);
    // std::swap(mat_shape_b[0], mat_shape_b[1]);
    // k_order_b = k_order_b ^ 1;
    // std::swap(ord_b[0], ord_b[1]);
    // std::swap(shape_b[0], shape_b[1]);
  // }

  mma16816_smem_loader b_loader(layout->wpt(1), ord_b, k_order_b, shape_b,
                                mma_instr_b, mat_shape_b,
                                per_phase_b, max_phase_b, dtsize_b, builder_, add, mul, gep);
  std::vector<Value*> off_b = b_loader.compute_offs(warp_n, lane);

  if(licm_ptrs)
    builder_->SetInsertPoint(CurrBB);
  // pointers
  int num_ptr_b = b_loader.get_num_ptr();
  std::vector<Value*> ptrs_b(num_ptr_b);
  for(int i = 0; i < num_ptr_b; i++)
    ptrs_b[i] = bit_cast(gep(shmems_[B], {off_b[i]}), smem_ptr_ty);


  // loading function
  std::function<void(int,int,int,bool)> load_b;
  load_b = [&](int n, int k, int inc, bool is_prefetch) {
      auto [hb0, hb1, hb2, hb3] = b_loader.load_x4(k, n, inc, is_prefetch, phiB, shared_pre_ptr_[layout_b],
                                                   shared_next_ptr_[layout_b], off_b, ptrs_b,
                                                   ldmatrix_ty, smem_ptr_ty, prefetch_latch_to_bb_);
      register_lds2(hb, n,   k,   inc, hb0, is_prefetch);
      register_lds2(hb, n+1, k,   inc, hb2, is_prefetch);
      register_lds2(hb, n,   k+1, inc, hb1, is_prefetch);
      register_lds2(hb, n+1, k+1, inc, hb3, is_prefetch);
  };



  // create mma & unpack result, m, n, k are offsets in mat
  auto call_mma = [&](unsigned m, unsigned n, unsigned k) {
      InlineAsm *mma_fn = InlineAsm::get(mma_ty, layout->get_ptx_instr() +
                                             " {$0, $1, $2, $3},"
                                             " {$4, $5, $6, $7},"
                                             " {$8, $9},"
                                             " {$10, $11, $12, $13};",
                                             "=r,=r,=r,=r,r,r,r,r,r,r,0,1,2,3", true);
      unsigned cols_per_thread = num_rep_n * 2;
      std::vector<size_t> idx = {
        (m + 0)*cols_per_thread + (n*2 + 0),
        (m + 0)*cols_per_thread + (n*2 + 1),
        (m + 1)*cols_per_thread + (n*2 + 0),
        (m + 1)*cols_per_thread + (n*2 + 1)
      };
      Value *nc = call(mma_ty, mma_fn,
                       {ha[{m, k}], ha[{m+1, k}], ha[{m, k+1}], ha[{m+1, k+1}],
                        hb[{n, k}], hb[{n, k+1}],
                        fc[idx[0]], fc[idx[1]], fc[idx[2]], fc[idx[3]]});
      fc[idx[0]] = extract_val(nc, std::vector<unsigned>{0});
      fc[idx[1]] = extract_val(nc, std::vector<unsigned>{1});
      fc[idx[2]] = extract_val(nc, std::vector<unsigned>{2});
      fc[idx[3]] = extract_val(nc, std::vector<unsigned>{3});
  };
  if (C->is_prefetched()) {
      // create phis
      builder_->SetInsertPoint(CurrBB->getFirstNonPHI());
      for(unsigned m = 0; m < num_rep_m; m++){
        ha[{2*m, 0}]   = phi(phi_ty, 2);
        ha[{2*m+1, 0}] = phi(phi_ty, 2);
        ha[{2*m, 1}]   = phi(phi_ty, 2);
        ha[{2*m+1, 1}] = phi(phi_ty, 2);
      }
      for(unsigned n = 0; n < num_rep_n; n+=2){
        hb[{n, 0}]   = phi(phi_ty, 2);
        hb[{n+1, 0}] = phi(phi_ty, 2);
        hb[{n, 1}]   = phi(phi_ty, 2);
        hb[{n+1, 1}] = phi(phi_ty, 2);
      }
      // insert prefetched lds at the end of loop header
      builder_->SetInsertPoint(bbs_[phiA->get_incoming_block(0)]->getTerminator());
      for(unsigned m = 0; m < num_rep_m; m++)
        load_a(2*m, 0, 0, true);
      for(unsigned n = 0; n < num_rep_n; n+=2)
        load_b(n, 0, 0, true);
      // update accumulators
      builder_->SetInsertPoint(CurrBB);
      for(unsigned k = 0; k < num_rep_k; ++k){ // stride of instr in mat is 2
        int next_k = (k + 1) % num_rep_k;
        // prefetch A
        for(unsigned m = 0; m < num_rep_m; m++)
          load_a(2*m, 2*next_k, 1, true);
        // prefetch B
        for(unsigned n = 0; n < num_rep_n; n+=2)
          load_b(n, 2*next_k, 1, true);
        // tensor core ops
        for(unsigned m = 0; m < num_rep_m; m++)
        for(unsigned n = 0; n < num_rep_n; n++){
          call_mma(2*m, n, 2*k);
        }
      }
  }
  else{
    for (unsigned k = 0; k < num_rep_k; k++) {
      for (unsigned m = 0; m < num_rep_m; m++)
        load_a(2*m, 2*k, 0, /*is_prefetch*/false);
      for (unsigned n = 0; n < num_rep_n; n+=2)
        load_b(n,   2*k, 0, /*is_prefetch*/false);
      for (unsigned m = 0; m < num_rep_m; m++)
      for (unsigned n = 0; n < num_rep_n; n++)
        call_mma(2*m, n, 2*k);
    }
  }
  // write back
  unsigned i = 0;
  for(indices_t idx: idxs_.at(C)){
    std::vector<Value*> key(idx.size() - 2);
    std::copy(idx.begin() + 2, idx.end(), key.begin());
    if(i >= fcs.at(key).size())
      i = 0;
    vals_[C][idx] = fcs.at(key)[i++];
  };

}

/**
 * \brief Code Generation for FMA-based `dot` (FP32, FP64, Default)
 */
void generator::visit_fmadot(ir::dot_inst* C, ir::value* A, ir::value* B, ir::value* D, unsigned NK, Type *c_ty, Function *f_mul_add) {
  auto shape_c = C->get_type()->get_block_shapes();
  auto shape_a = A->get_type()->get_block_shapes();
  auto shape_b = B->get_type()->get_block_shapes();
  auto ord_a = layouts_->get(A)->get_order();
  auto ord_b = layouts_->get(B)->get_order();
  analysis::scanline_layout* layout_c = layouts_->get(C)->to_scanline();
  analysis::shared_layout* layout_a = (analysis::shared_layout*)layouts_->get(C->get_operand(0));
  analysis::shared_layout* layout_b = (analysis::shared_layout*)layouts_->get(C->get_operand(1));
  bool is_a_row = ord_a[0] == 1;
  bool is_b_row = ord_b[0] == 1;
  std::string a_trans = is_a_row ? "" : ".trans";
  std::string b_trans = is_b_row ? ".trans" : "";
  int stride_a_m = is_a_row ? shape_a[1] : 1;
  int stride_a_k = is_a_row ? 1 : shape_a[0];
  int stride_b_n = is_b_row ? 1 : shape_b[0];
  int stride_b_k = is_b_row ? shape_b[1] : 1;
  int stride_a0 = is_a_row ? stride_a_k : stride_a_m;
  int stride_a1 = is_a_row ? stride_a_m : stride_a_k;
  int stride_b0 = is_b_row ? stride_b_n : stride_b_k;
  int stride_b1 = is_b_row ? stride_b_k : stride_b_n;
  int lda = is_a_row ? stride_a_m : stride_a_k;
  int ldb = is_b_row ? stride_b_k : stride_b_n;
  int per_phase_a = swizzle_->get_per_phase(layout_a);
  int max_phase_a = swizzle_->get_max_phase(layout_a);
  int per_phase_b = swizzle_->get_per_phase(layout_b);
  int max_phase_b = swizzle_->get_max_phase(layout_b);
  int num_ptr_a   = 8;
  int num_ptr_b   = 8;
  int vec_a = 2;
  int vec_b = 4;
  distributed_axis ax_m = axes_.at(a_axes_->get(C, 0));
  distributed_axis ax_n = axes_.at(a_axes_->get(C, 1));
//  Value* thread = tgt_->get_local_id(mod_, *builder_, 0);

  Value* off_a0 = is_a_row ? i32(0) : mul(ax_m.thread_id, i32(ax_m.contiguous));
  Value* off_a1 = is_a_row ? mul(ax_m.thread_id, i32(ax_m.contiguous)): i32(0);
  std::vector<Value*> off_a(num_ptr_a);
  for(int i = 0; i < num_ptr_a; i++){
//    Value* off_a0i = add(off_a0, i32(is_a_row ? vec_a : layout_c->mts(0)*vec_a));
//    off_a0i = exact_udiv(off_a0i, i32(vec_a));
//    off_a0i = xor_(off_a0i, phase_a);
//    off_a0i = mul(off_a0i, i32(vec_a));
    off_a[i] = add(mul(off_a0, i32(stride_a0)), mul(off_a1, i32(stride_a1)));
  }
  Value* off_b0 = is_b_row ? mul(ax_n.thread_id, i32(ax_n.contiguous)): i32(0);
  Value* off_b1 = is_b_row ? i32(0) : mul(ax_n.thread_id, i32(ax_n.contiguous));
  std::vector<Value*> off_b(num_ptr_b);
  for(int i = 0; i < num_ptr_b; i++){
//    Value* off_b0i = add(off_b0, i32(is_b_row ? layout_c->mts(1)*vec_b : vec_b));
//    off_b0i = exact_udiv(off_b0i, i32(vec_b));
//    off_b0i = xor_(off_b0i, phase_b);
//    off_b0i = mul(off_b0i, i32(vec_b));
    off_b[i] = add(mul(off_b0, i32(stride_b0)), mul(off_b1, i32(stride_b1)));
  }
  std::vector<Value*> ptrs_a(num_ptr_a);
  for(int i = 0; i < num_ptr_a; i++)
    ptrs_a[i] = gep(shmems_[A], off_a[i]);
  std::vector<Value*> ptrs_b(num_ptr_b);
  for(int i = 0; i < num_ptr_b; i++)
    ptrs_b[i] = gep(shmems_[B], off_b[i]);

  std::map<indices_t, Value*> ret = vals_[D];
  std::map<std::pair<int, int>, Value*> has, hbs;
  auto ord = layout_c->get_order();
  for(unsigned k = 0; k < NK; k++){
    int z = 0;
    for(unsigned i = 0; i < shape_c[ord[1]]; i += layout_c->shape_per_cta(ord[1]))
    for(unsigned j = 0; j < shape_c[ord[0]]; j += layout_c->shape_per_cta(ord[0]))
    for(unsigned ii = 0; ii < layout_c->nts(ord[1]); ii++)
    for(unsigned jj = 0; jj < layout_c->nts(ord[0]); jj++){
      unsigned m = (ord[0] == 1) ? i : j;
      unsigned n = (ord[0] == 1) ? j : i;
      unsigned mm = (ord[0] == 1) ? ii : jj;
      unsigned nn = (ord[0] == 1) ? jj : ii;
      if(has.find({m + mm, k}) == has.end()){
        Value* pa = gep(ptrs_a[0], i32((m + mm)*stride_a_m + k*stride_a_k));
        Value* va = load(pa);
        has[{m + mm, k}] = va;
      }
      if(hbs.find({n + nn, k}) == hbs.end()){
        Value* pb = gep(ptrs_b[0], i32((n + nn)*stride_b_n + k*stride_b_k));
        Value* vb = load(pb);
        hbs[{n + nn, k}] = vb;
      }
      ret[idxs_[C].at(z)] = call(f_mul_add, {has[{m+mm,k}], hbs[{n+nn, k}], ret[idxs_[C].at(z)]});
      z++;
    }
  }

  for(indices_t idx: idxs_.at(C)){
    vals_[C][idx] = ret[idx];
  }
}

/**
 * \brief Code Generation for `dot`
 * Dispatches to appropriate specialized function
 */
void generator::visit_dot_inst(ir::dot_inst* dot) {
  Function *fn = builder_->GetInsertBlock()->getParent();
  Module *module = fn->getParent();
  ir::value *A = dot->get_operand(0);
  ir::value *B = dot->get_operand(1);
  ir::value *D = dot->get_operand(2);
  Type *c_ty = cvt(D->get_type()->get_scalar_ty());
  Function *f_mul_add = Intrinsic::getDeclaration(module, Intrinsic::fmuladd, std::vector<llvm::Type*>{c_ty});
  auto A_shapes = A->get_type()->get_block_shapes();
  size_t red_axis = 1;
  unsigned NK = A_shapes[red_axis];
  bool is_outer = NK == 1;
  bool is_mma = layouts_->get(dot)->to_mma();
  if(!is_outer && is_mma && tgt_->as_nvidia()->sm() < 80)
    return visit_mma884(dot, A, B, D, NK);
  if(!is_outer && is_mma && tgt_->as_nvidia()->sm() >= 80)
    return visit_mma16816(dot, A, B, D, NK); // rename it as visit_mma_v2()?
  if (dot->get_type()->get_scalar_ty()->is_fp32_ty() &&
      A->get_type()->get_scalar_ty()->is_fp32_ty())
    return visit_fmadot(dot, A, B, D, NK, c_ty, f_mul_add);
  throw std::runtime_error("dot has invalid operand type");
}

void generator::visit_trans_inst(ir::trans_inst* trans) {
  throw std::runtime_error("not supported");
}

/**
 * \brief Code Generation for `sqrt`
 */
void generator::visit_sqrt_inst(ir::sqrt_inst* x) {
  for(indices_t idx: idxs_.at(x)){
    Value *val = vals_[x->get_operand(0)][idx];
    Value *ret = intrinsic(Intrinsic::sqrt, {val->getType()}, {val});
    vals_[x][idx] = ret;
  }
}

Value* generator::shared_off(const std::vector<unsigned>& shapes, const std::vector<int>& order, indices_t idx){
  // strides
  std::vector<Value*> strides(shapes.size(), builder_->getInt32(0));
  strides[order[0]] = builder_->getInt32(1);
  for(size_t i = 1; i < idx.size(); i++)
    strides[order[i]] = builder_->CreateMul(strides[order[i-1]], builder_->getInt32(shapes[order[i-1]]));
  // result
  Value *result = builder_->getInt32(0);
  for(size_t i = 0; i < idx.size(); i++)
    result = builder_->CreateAdd(result, builder_->CreateMul(idx[i], strides[i]));
  return result;
}

inline Value* generator::shfl_sync(Value* acc, int32_t i){
  Type* ty = acc->getType();
  std::string asm_str = "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;";
  InlineAsm *shfl = InlineAsm::get(FunctionType::get(ty, {ty, i32_ty}, false), asm_str, "=f,f,r", false);
  if(ty->getPrimitiveSizeInBits() <= 32)
    return call(shfl, {acc, i32(i)});
  acc = bit_cast(acc, vec_ty(f32_ty, 2));
  Value* acc0 = builder_->CreateExtractElement(acc, i32(0));
  Value* acc1 = builder_->CreateExtractElement(acc, i32(1));
  Value* ret = UndefValue::get(vec_ty(f32_ty, 2));
  ret = insert_elt(ret, shfl_sync(acc0, i), i32(0));
  ret = insert_elt(ret, shfl_sync(acc1, i), i32(1));
  return bit_cast(ret, ty);
}

/**
 * \brief Code Generation for `reduce` (ND case)
 */
void generator::visit_reducend_inst_fast(ir::reduce_inst* x, acc_fn_t do_acc, Value *neutral){
  ir::value *arg = x->get_operand(0);
  const auto with_index = x->with_index();
  unsigned axis = x->get_axis();
  analysis::distributed_layout* layout = dynamic_cast<analysis::distributed_layout*>(layouts_->get(arg));
  const auto &shapes = layout->get_shape();

  Type* sca_ty = cvt(arg->get_type()->get_scalar_ty());
  size_t n_bits = sca_ty->getPrimitiveSizeInBits();
  std::string n_bits_str = std::to_string(n_bits);
  std::string cst = (n_bits == 64) ? "l" : "r";

  FunctionType *st_shared_ty = FunctionType::get(void_ty, {i1_ty, ptr_ty(sca_ty, 3), sca_ty}, false);
  InlineAsm *st_shared = InlineAsm::get(st_shared_ty, "@$0 st.shared.b" + n_bits_str + " [$1], $2;", "b," + cst + "," + cst, true);
  FunctionType *ld_shared_ty = FunctionType::get(sca_ty, {i1_ty, ptr_ty(sca_ty, 3)}, false);
  InlineAsm *ld_shared = InlineAsm::get(ld_shared_ty, "@$1 ld.shared.b" + n_bits_str + " $0, [$2];", "=" + cst + ",b," + cst, true);

  Type *index_ty = IntegerType::get(*ctx_, 32);
  FunctionType *st_shared_index_ty =
      FunctionType::get(void_ty, {i1_ty, ptr_ty(index_ty, 3), index_ty}, false);
  InlineAsm *st_shared_index = InlineAsm::get(
      st_shared_index_ty, "@$0 st.shared.b32 [$1], $2;", "b,r,r", true);
  FunctionType *ld_shared_index_ty =
      FunctionType::get(index_ty, {i1_ty, ptr_ty(index_ty, 3)}, false);
  InlineAsm *ld_shared_index = InlineAsm::get(
      ld_shared_index_ty, "@$1 ld.shared.b32 $0, [$2];", "=r,b,r", true);

  Value* thread = tgt_->get_local_id(mod_, *builder_, 0);
  Value* warp = udiv(thread, i32(32));
  Value* lane = urem(thread, i32(32));

  unsigned shuffle_width = 0;
  unsigned warps_per_inner = 0;
  auto arg_vals = vals_.at(arg);
  std::vector<indices_t> arg_idxs = idxs_.at(arg);
  size_t n_elts = arg_idxs.size();
  unsigned col_per_thread = 0;
  Value* warp_j = nullptr;
  if (analysis::scanline_layout *scanline = layout->to_scanline()) {
    std::vector<int> order = layout->get_order();
    unsigned mts = scanline->mts(order[0]);
    shuffle_width = std::min<int>(mts, 32);
    warps_per_inner = std::max<int>(mts / 32, 1);
    col_per_thread = shapes[order[0]] / mts;
    warp_j = urem(warp, i32(warps_per_inner));
  } else if (layout->to_mma()) {
    shuffle_width = 4;
    warps_per_inner = layout->to_mma()->wpt(1);
    col_per_thread = axes_.at(a_axes_->get(arg, 1)).values.size();
    warp_j = axes_.at(a_axes_->get(arg, 1)).thread_id;
  }
  assert(warp_j != nullptr);

  // unsigned col_per_thread = 2 * shapes[order[0]] / layout->shape_per_cta(order[0]);
  //
  Value *base = cast_shared_layout_ptr(layouts_->get(layouts_->tmp(x)),
                                       cvt(x->get_type()->get_scalar_ty()));
  Value *index_base =
      with_index ? cast_shared_layout_ptr(layouts_->get(layouts_->tmp_index(x)),
                                          IntegerType::get(*ctx_, 32))
                 : nullptr;

  // preds
  Value* is_lane0 = icmp_eq(lane, i32(0));
  Value* is_warp0 = icmp_eq(warp, i32(0));
  Value* is_thread0 = icmp_eq(thread, i32(0));
  Value* lane_j = urem(lane, i32(shuffle_width));
  if(warps_per_inner > 1)
    add_barrier();
  // compute partial sum for each warp, and store to shared memory
  for(size_t i = 0; i < n_elts/col_per_thread; i++){
    std::pair<Value*, Value*> acc;
    // reduce within thread
    for(size_t j = 0; j < col_per_thread; j++){
      auto arg_idx = arg_idxs[i*col_per_thread + j];
      bool is_first = j == 0;
      do_acc(
          acc, [&]() -> Value * { return arg_vals[arg_idx]; },
          [&]() -> Value * { return arg_idx[axis]; }, is_first);
    }

    // reduce within warp
    for(int k = shuffle_width/2 ; k > 0; k >>= 1) {
      do_acc(
          acc, [&]() -> Value * { return shfl_sync(acc.first, k); },
          [&]() -> Value * { return shfl_sync(acc.second, k); }, false);
    }
    // store partial result to shared memory
    auto x_idxs = idxs_[x][i];
    Value* x_idx = x_idxs.empty() ? builder_->getInt32(0) : x_idxs[0];
    // single warp on the reduce dimension -- no need to use shmem
    if(warps_per_inner==1){
      vals_[x][idxs_[x][i]] = with_index ? acc.second : acc.first;
    }
    else{
      Value* st_off = add(mul(x_idx, i32(warps_per_inner)), warp_j);
      call(st_shared, {icmp_eq(lane_j, i32(0)), gep(base, st_off), acc.first});
      if (with_index) {
        call(st_shared_index,
            {icmp_eq(lane_j, i32(0)), gep(index_base, st_off), acc.second});
      }
    }
  }
  if(warps_per_inner==1)
    return;
  add_barrier();
  // at this point, partial accumulator synchronized in shared memory
  // Just need to reduce `warp_per_inner` numbers in shared memory
  for(size_t i = 0; i < n_elts/col_per_thread; i++){
    auto x_idxs = idxs_[x][i];
    Value* x_idx = x_idxs.empty() ? builder_->getInt32(0) : x_idxs[0];
    Value* ld_off = add(mul(x_idx, i32(warps_per_inner)), urem(lane_j, i32(warps_per_inner)));
    std::pair<Value*, Value*> acc;
    acc.first = call(ld_shared, {builder_->getInt1(true), gep(base, ld_off)});
    acc.second = with_index ? call(ld_shared_index, {builder_->getInt1(true),
                                                     gep(index_base, ld_off)})
                            : nullptr;
    for (int k = warps_per_inner / 2; k > 0; k >>= 1) {
      do_acc(
          acc, [&]() -> Value * { return shfl_sync(acc.first, k); },
          [&]() -> Value * { return shfl_sync(acc.second, k); }, false);
    }
    vals_[x][idxs_[x][i]] = with_index ? acc.second : acc.first;
  }
  // add_barrier();
}


void generator::visit_reducend_inst(ir::reduce_inst* x, acc_fn_t do_acc, Value *neutral) {
  ir::value *arg = x->get_operand(0);
  unsigned axis = x->get_axis();
  auto with_index = x->with_index();

  // reduce within thread
  // index-><current reduced value, current min/max index (optional)>
  std::map<indices_t, std::pair<Value*, Value*>> accs;
  for(indices_t idx: idxs_.at(arg)){
    indices_t pidx = idx;
    pidx[axis] = i32(0);
    bool is_first = accs.find(pidx) == accs.end();
    do_acc(
        accs[pidx], [&]() -> Value * { return vals_[arg][idx]; },
        [&]() -> Value * { return idx[axis]; }, is_first);
  };

  // reduce within blocks
  auto *data_layout = layouts_->get(layouts_->tmp(x));
  auto *data_ptr =
      cast_shared_layout_ptr(data_layout, cvt(x->get_type()->get_scalar_ty()));
  auto *index_ptr =
      with_index ? cast_shared_layout_ptr(layouts_->get(layouts_->tmp_index(x)),
                                          IntegerType::get(*ctx_, 32))
                 : data_ptr;

  auto shape  = data_layout->get_shape();
  auto order  = data_layout->get_order();
  Value *lane = axes_.at(a_axes_->get(arg, axis)).thread_id;
  for(auto& x: accs) {
    // current element being computed
    std::pair<Value *, Value *> acc = x.second;
    indices_t write_idx = x.first;
    write_idx[axis] = lane;
    // shared memory write  pointer
    Value *write_off = shared_off(shape, order, write_idx);
    Value *write_ptr = gep(data_ptr, write_off);
    Value *index_write_ptr = gep(index_ptr, write_off);
    // initialize shared memory
    add_barrier();
    store(acc.first, write_ptr);
    if (with_index) {
      store(acc.second, index_write_ptr);
    }
    // build result
    indices_t idx(write_idx.size(), i32(0));
    for(size_t i = shape[axis]/2; i > 0; i >>= 1){
      idx[axis] = i32(i);
      // read pointer
      Value *read_msk = icmp_ult(lane, i32(i));
      Value *read_off = select(read_msk, shared_off(shape, order, idx), i32(0));
      Value *read_ptr = gep(write_ptr, read_off);
      Value *index_read_ptr = gep(index_write_ptr, read_off);
      add_barrier();
      // update accumulator
      do_acc(
          acc, [&]() -> Value * { return load(read_ptr); },
          [&]() -> Value * { return load(index_read_ptr); }, false);
      add_barrier();
      store(acc.first, write_ptr);
      if (with_index) {
        store(acc.second, index_write_ptr);
      }
    }
  }
  add_barrier();

  // write back
  for(indices_t idx: idxs_.at(x)){
    indices_t read_idx = idx;
    read_idx.insert(read_idx.begin() + axis, i32(0));
    Value *read_off = shared_off(shape, order, read_idx);
    Value *read_ptr =
        with_index ? gep(index_ptr, read_off) : gep(data_ptr, read_off);
    vals_[x][idx] = load(read_ptr);
  };
}

/**
 * \brief Code Generation for `reduce` (generic case)
 */
void generator::visit_reduce_inst(ir::reduce_inst* x) {
  Type *ty = cvt(x->get_type()->get_scalar_ty());
  // accumulation function
  ir::reduce_inst::op_t op = x->get_op();
  auto do_acc_op = [&](Value *x, Value *y) -> Value* {
    switch(op){
    case ir::reduce_inst::ADD: return add(x, y);
    case ir::reduce_inst::SUB: return sub(x, y);
    case ir::reduce_inst::ARGUMAX: return icmp_uge(x, y);
    case ir::reduce_inst::ARGUMIN: return icmp_ule(x, y);
    case ir::reduce_inst::ARGMAX: return icmp_sge(x, y);
    case ir::reduce_inst::ARGMIN: return icmp_sle(x, y);
    case ir::reduce_inst::UMAX: return select(icmp_uge(x, y), x, y);
    case ir::reduce_inst::UMIN: return select(icmp_ule(x, y), x, y);
    case ir::reduce_inst::MAX: return select(icmp_sge(x, y), x, y);
    case ir::reduce_inst::MIN: return select(icmp_sle(x, y), x, y);
    case ir::reduce_inst::FADD: return fadd(x, y);
    case ir::reduce_inst::FSUB: return fsub(x, y);
    case ir::reduce_inst::ARGFMAX: return fcmp_oge(x, y);
    case ir::reduce_inst::ARGFMIN: return fcmp_ole(x, y);
    case ir::reduce_inst::FMAX: return max_num(x, y);
    case ir::reduce_inst::FMIN: return min_num(x, y);
    case ir::reduce_inst::XOR: return xor_(x, y);

    default: throw std::runtime_error("unreachable");
    }
  };

  auto do_acc = [&](std::pair<Value *, Value *> &acc,
                    std::function<Value *()> load_value_fn,
                    std::function<Value *()> load_index_fn,
                    bool is_first) -> void {
    auto *val = load_value_fn();
    if (x->with_index()) {
      auto *index = load_index_fn();
      if (is_first) {
        acc.first = val;
        acc.second = index;
      } else {
        Value *ret = do_acc_op(acc.first, val);
        acc.first = select(ret, acc.first, val);
        acc.second = select(ret, acc.second, index);
      }
    } else {
      acc.first = is_first ? val : do_acc_op(acc.first, val);
    }
  };

  // neutral element
  Value *neutral;
  switch(op) {
    case ir::reduce_inst::ADD: neutral = ConstantInt::get(ty, 0); break;
    case ir::reduce_inst::SUB: neutral = ConstantInt::get(ty, 0); break;
    case ir::reduce_inst::ARGUMAX: neutral = ConstantInt::get(ty, INT32_MIN); break;
    case ir::reduce_inst::ARGUMIN: neutral = ConstantInt::get(ty, INT32_MAX); break;
    case ir::reduce_inst::ARGMAX: neutral = ConstantInt::get(ty, INT32_MIN); break;
    case ir::reduce_inst::ARGMIN: neutral = ConstantInt::get(ty, INT32_MAX); break;
    case ir::reduce_inst::UMAX: neutral = ConstantInt::get(ty, 0); break;
    case ir::reduce_inst::UMIN: neutral = ConstantInt::get(ty, UINT32_MAX); break;
    case ir::reduce_inst::MAX: neutral = ConstantInt::get(ty, INT32_MIN); break;
    case ir::reduce_inst::MIN: neutral = ConstantInt::get(ty, INT32_MAX); break;
    case ir::reduce_inst::FADD: neutral = ConstantFP::get(ty, 0); break;
    case ir::reduce_inst::FSUB: neutral = ConstantFP::get(ty, 0); break;
    case ir::reduce_inst::ARGFMAX: neutral = ConstantFP::get(ty, -INFINITY); break;
    case ir::reduce_inst::ARGFMIN: neutral = ConstantFP::get(ty, INFINITY); break;
    case ir::reduce_inst::FMAX: neutral = ConstantFP::get(ty, -INFINITY); break;
    case ir::reduce_inst::FMIN: neutral = ConstantFP::get(ty, INFINITY); break;
    case ir::reduce_inst::XOR: neutral = ConstantInt::get(ty, 0); break;
    default: throw std::runtime_error("unreachable");
  }
  ir::value *arg = x->get_operand(0);
  bool is_coalesced_scanline = layouts_->is_coalesced_scanline(x);
  bool is_a100_mma = layouts_->is_a100_mma(x);
  if (is_coalesced_scanline || is_a100_mma)
    visit_reducend_inst_fast(x, do_acc, neutral);
  else
    visit_reducend_inst(x, do_acc, neutral);
}

/**
 * \brief Code Generation for `select`
 */
void generator::visit_select_inst(ir::select_inst* x) {
  for(indices_t idx: idxs_.at(x)){
    vals_[x][idx] = select(vals_[x->get_operand(0)][idx],
                           vals_[x->get_operand(1)][idx],
                           vals_[x->get_operand(2)][idx]);
  }
}



void generator::visit_layout_convert(ir::value *out, ir::value *in){
  ir::block_type::block_shapes_t shape = out->get_type()->get_block_shapes();
  // pointer to temporary shared memory
  Type *ty = cvt(out->get_type()->get_scalar_ty());

  // Orders
  analysis::distributed_layout* in_layout = dynamic_cast<analysis::distributed_layout*>(layouts_->get(in));
  analysis::distributed_layout* out_layout = dynamic_cast<analysis::distributed_layout*>(layouts_->get(out));
  Value *base;
  int off = alloc_->offset(layouts_->get(layouts_->tmp(out)));
   // std::cout << off << std::endl;
  base = gep(shmem_, i32(off));
  base = bit_cast(base, ptr_ty(ty, 3));
  std::vector<int> n_reps;
  for(int i = 0; i < shape.size(); i++){
    int in_per_cta = in_layout->shape_per_cta(i);
    int out_per_cta = out_layout->shape_per_cta(i);
    int max_per_cta = std::max(in_per_cta, out_per_cta);
    n_reps.push_back(shape[i]/max_per_cta);
  }
  std::vector<std::vector<Value*>> in_ax;
  std::vector<std::vector<Value*>> out_ax;
  for(int d = 0; d < shape.size(); d++){
    in_ax.push_back(axes_.at(a_axes_->get(in, d)).values);
    out_ax.push_back(axes_.at(a_axes_->get(out, d)).values);
  }
  auto in_ord =
      in_layout->to_mma() ? out_layout->get_order() : in_layout->get_order();
  auto out_ord =
      out_layout->to_mma() ? in_layout->get_order() : out_layout->get_order();
  // out_ord[0] == 0 or in_order[0] == 0 means the first dimension is
  // non-contiguous. in_vec can be greater than 0 only if both out_ord[0] and
  // and in_ord[0] are contiguous.
  int in_vec = out_ord[0] == 0  ? 1
               : in_ord[0] == 0 ? 1
                                : in_layout->contig_per_thread(in_ord[0]);
  int out_vec = out_ord[0] == 0 ? 1 : out_layout->contig_per_thread(out_ord[0]);
  int pad = std::max(in_vec, out_vec);
  Value *in_ld = i32(shape[in_ord[0]] + pad);
  Value *out_ld = i32(shape[out_ord[0]] + pad);
  for(int i = 0; i < n_reps[0]; i++)
  for(int j = 0; j < n_reps[1]; j++){
    int max_ii, max_jj;
    add_barrier();
    max_ii = in_ax[0].size()/n_reps[0];
    max_jj = in_ax[1].size()/n_reps[1];
    for(int ii = 0; ii < max_ii; ii++)
    for(int jj = 0; jj < max_jj; jj+=in_vec){
      // shared mem pointer
      indices_t offs = {in_ax[0][ii], in_ax[1][jj]};
      Value *off  = add(offs[out_ord[0]], mul(out_ld, offs[out_ord[1]]));
      Value *ptr = gep(base, off);
      // stash value to shared mem
      Value* vals = UndefValue::get(vec_ty(ty, in_vec));
      for(int jjj = 0; jjj < in_vec; jjj++){
        indices_t idxs = {in_ax[0][i*max_ii + ii],
                          in_ax[1][j*max_jj + jj + jjj]};
        Value* val = bit_cast(vals_[in][idxs], ty);
        vals = insert_elt(vals, val, jjj);
      }
      ptr = bit_cast(ptr, ptr_ty(vals->getType(), ptr->getType()->getPointerAddressSpace()));
      store(vals, ptr);
    }
    add_barrier();
    max_ii = out_ax[0].size()/n_reps[0];
    max_jj = out_ax[1].size()/n_reps[1];
    for(int ii = 0; ii < max_ii; ii++)
    for(int jj = 0; jj < max_jj; jj+=out_vec){
      // shared mem pointer
      indices_t offs = {out_ax[0][ii], out_ax[1][jj]};
      Value *off  = add(offs[out_ord[0]], mul(out_ld, offs[out_ord[1]]));
      Value *ptr = gep(base, off);
      ptr = bit_cast(ptr, ptr_ty(vec_ty(ty, out_vec), ptr->getType()->getPointerAddressSpace()));
      // load value from shared rem
      Value* vals = load(ptr);
      for(int jjj = 0; jjj < out_vec; jjj++){
        indices_t idxs = {out_ax[0][i*max_ii + ii],
                          out_ax[1][j*max_jj + jj + jjj]};
        vals_[out][idxs] = extract_elt(vals, jjj);
      }
    }

  }
}

void generator::visit_cvt_layout_inst(ir::cvt_layout_inst *rc) {
  visit_layout_convert(rc, rc->get_operand(0));
}

void generator::visit_masked_load_async_inst(ir::masked_load_async_inst* x){
  unsigned in_vec = 1;
  ir::value *arg = x->get_pointer_operand();
  analysis::shared_layout* out_layout = layouts_->get(x)->to_shared();
  analysis::scanline_layout* in_layout = layouts_->get(arg)->to_scanline();
  auto out_order = out_layout->get_order();
  auto in_order = in_layout->get_order();
  // tiles
  if(out_order == in_order)
    in_vec = in_layout->nts(in_order[0]);
  int out_vec = swizzle_->get_vec(out_layout);
  int min_vec = std::min<int>(out_vec, in_vec);
  int s = std::max<int>(out_vec / in_vec, 1);
  //
  int per_phase = swizzle_->get_per_phase(out_layout);
  int max_phase = swizzle_->get_max_phase(out_layout);
  //
  int in_ld = in_layout->get_shape()[in_order[0]] / in_layout->mts(in_order[0]);
  int n_shared_1 = std::max<int>(per_phase*max_phase / in_layout->mts(in_order[1]), 1);
  int n_shared_0 = std::max<int>(in_vec    / out_vec, 1);
  auto shapes = x->get_type()->get_block_shapes();
  BasicBlock* CurrBB = builder_->GetInsertBlock();
  BasicBlock* FirstBB = &CurrBB->getParent()->getEntryBlock();
  std::map<std::pair<int, int>, Value*> tmp;
  std::vector<std::pair<Value*, int>> shared;
  for(int i = 0; i < idxs_.at(arg).size(); i++){
    unsigned id = i / min_vec;
    // input ptr info
    int id_0 = id % (in_ld/min_vec);
    int id_1 = id / (in_ld/min_vec);
    int off_0 = id_0 / n_shared_0 * n_shared_0 * in_layout->mts(in_order[0]);
    int off_1 = id_1 / n_shared_1 * n_shared_1 * in_layout->mts(in_order[1]);
    int off = (off_1*shapes[in_order[0]] + off_0);
    std::pair<int, int> key = {id_1  % n_shared_1, id_0 % n_shared_0};
    if(tmp.find(key) == tmp.end()){
      if(CurrBB != FirstBB)
        builder_->SetInsertPoint(FirstBB->getTerminator());
      indices_t idx = idxs_.at(arg).at(key.first*in_ld);
      Value* phase = udiv(idx[in_order[1]], i32(per_phase));
      phase = urem(phase, i32(max_phase));
      Value* off_1 = mul(idx[in_order[1]], i32(shapes[in_order[0]]));
      Value* off_0  = add(idx[in_order[0]], i32(key.second*out_vec));
      off_0 = udiv(off_0, i32(min_vec));
      off_0 = add(mul(xor_(udiv(off_0, i32(s)), phase),i32(s)), urem(off_0, i32(s)));
      off_0 = mul(off_0 , i32(min_vec));
      Value* off = add(off_0, off_1);
      if(CurrBB != FirstBB)
        builder_->SetInsertPoint(CurrBB);
      tmp[key] = gep(shmems_[x], {off});
    }
    shared.push_back({tmp[key], off});
  }
  size_t dtsize = x->get_type()->get_scalar_ty()->get_primitive_size_in_bits() / 8;
  for(size_t i = 0; i < idxs_.at(arg).size(); i += in_vec){
    auto idx = idxs_[arg][i];
    // input ptr info
    Value *ptr = vals_[arg][idx];
    size_t in_off = 0;
    GetElementPtrInst *in_gep = dyn_cast<GetElementPtrInst>(vals_[arg][idx]);
    if(in_gep){
      ConstantInt* cst = dyn_cast<ConstantInt>(in_gep->idx_begin());
      in_off = cst ? cst->getValue().getSExtValue()*dtsize : 0;
      ptr= cst ? in_gep->getPointerOperand() : in_gep;
    }
    // output ptr info
    Value* out_base = shared[i].first;
    int out_off = shared[i].second*dtsize;
    // asm
    std::string mod = (in_vec*dtsize == 16) ? ".cg" : ".ca";
//    Value* false_value = vals_[x->get_false_value_operand()][idx];
//    bool is_zero_false_value = false;
//    if(Constant* cst = dyn_cast<Constant>(false_value))
//      is_zero_false_value = cst->isZeroValue();
    Value* src_size = builder_->CreateSelect(vals_[x->get_mask_operand()][idx], i32(in_vec*dtsize), i32(0));
    std::string asm_str = "cp.async" + mod + ".shared.global [$0 + " + std::to_string(out_off) + "], [$1 + " + std::to_string(in_off) + "], " + std::to_string(in_vec*dtsize) + ", $2;";
    FunctionType *ty = FunctionType::get(void_ty, {out_base->getType(), ptr->getType(), builder_->getInt32Ty()}, false);
    InlineAsm *iasm = InlineAsm::get(ty, asm_str, "r,l,r", true);
    call(iasm, {out_base, ptr, src_size});
  }

  std::string asm_str = "cp.async.commit_group;";
  InlineAsm *iasm = InlineAsm::get(FunctionType::get(void_ty, {}), asm_str, "", true);
  call(iasm);
}

void generator::visit_copy_to_shared_inst(ir::copy_to_shared_inst* cts) {
  unsigned in_vec = 1;
  ir::value *arg = cts->get_operand(0);
  analysis::shared_layout* out_layout = layouts_->get(cts)->to_shared();
  analysis::distributed_layout* in_layout = dynamic_cast<analysis::distributed_layout*>(layouts_->get(arg));
  auto out_order = out_layout->get_order();
  auto in_order = in_layout->get_order();
  // tiles
  if(out_order == in_order)
    in_vec = in_layout->contig_per_thread(in_order[0]);
  int out_vec = swizzle_->get_vec(out_layout);
  int min_vec = std::min<int>(out_vec, in_vec);
  int s = std::max<int>(out_vec / in_vec, 1);
  //
  int per_phase = swizzle_->get_per_phase(out_layout);
  int max_phase = swizzle_->get_max_phase(out_layout);
  //
  int mts_0 = in_layout->shape_per_cta(in_order[0]) / in_layout->contig_per_thread(in_order[0]);
  int mts_1 = in_layout->shape_per_cta(in_order[1]) / in_layout->contig_per_thread(in_order[1]);
  if(in_layout->to_mma()){
    mts_0 = 4 * in_layout->to_mma()->wpt(in_order[0]);
    mts_1 = 8 * in_layout->to_mma()->wpt(in_order[1]);
    per_phase = 1;
    max_phase = 8;
  }

  int in_ld = in_layout->get_shape()[in_order[0]] / mts_0;
  int n_shared_0 = std::max<int>(in_vec    / out_vec, 1);
  int n_shared_1 = std::max<int>(per_phase*max_phase / mts_1, 1);
  if(in_layout->to_mma()){
    n_shared_0 = 8;
    n_shared_1 = 1;
  }

  BasicBlock* CurrBB = builder_->GetInsertBlock();
  BasicBlock* FirstBB = &CurrBB->getParent()->getEntryBlock();
  auto shapes = cts->get_type()->get_block_shapes();


  // store to shared
  Value *current = nullptr;
  std::map<std::pair<int, int>, Value*> ptrs;
  for(int i = 0; i < idxs_.at(arg).size(); i++){
    auto idx = idxs_[arg][i];
    Value *in_value = vals_[arg][idx];
    if(i % min_vec == 0)
      current = UndefValue::get(vec_ty(in_value->getType(), min_vec));
    current = insert_elt(current, in_value, i % min_vec);
    if(i % min_vec == min_vec - 1){
      unsigned id = i / min_vec;
      // input ptr info
      int id_0 = id % (in_ld/min_vec);
      int id_1 = id / (in_ld/min_vec);
      // std::cout << id_0 << " " << id_1 << " " << in_ld << " " << std::endl;
      std::pair<int, int> key = {id_1  % n_shared_1, id_0 % n_shared_0};
      if(ptrs.find(key) == ptrs.end()){
        if(FirstBB->getTerminator())
            builder_->SetInsertPoint(FirstBB->getTerminator());
        else
            builder_->SetInsertPoint(FirstBB);
        indices_t idx = idxs_.at(arg).at(key.first*in_ld);
        Value* phase = udiv(idx[in_order[1]], i32(per_phase));
        phase = urem(phase, i32(max_phase));
        Value* off_1 = mul(idx[in_order[1]], i32(shapes[in_order[0]]));
        Value* off_0  = add(idx[in_order[0]], i32(key.second*out_vec));
        off_0 = udiv(off_0, i32(min_vec));
        off_0 = add(mul(xor_(udiv(off_0, i32(s)), phase),i32(s)), urem(off_0, i32(s)));
        off_0 = mul(off_0 , i32(min_vec));
        Value* off = add(off_0, off_1);
        builder_->SetInsertPoint(CurrBB);
        ptrs[key] = gep(shmems_.at(cts), {off});
      }
      int off_0 = id_0 / n_shared_0 * n_shared_0 * mts_0;
      int off_1 = id_1 / n_shared_1 * n_shared_1 * mts_1;
      if(in_layout->to_mma()){
        off_0 = id_0/n_shared_0*n_shared_0*8;
        off_1 = id_1/n_shared_1*n_shared_1*8;
      }
      int off = (off_1*shapes[in_order[0]] + off_0);
      Value* ptr = gep(ptrs[key], {i32(off)});
      ptr = bit_cast(ptr, current->getType()->getPointerTo(3));
      // asm
      store(current, ptr);
    }
  };
}

void generator::visit_copy_from_shared_inst(ir::copy_from_shared_inst*) {
  throw std::runtime_error("TODO");
}

Instruction* generator::add_barrier() {
  Module *module = builder_->GetInsertBlock()->getModule();
  return tgt_->add_barrier(module, *builder_);
}

void generator::visit_barrier_inst(ir::barrier_inst*) {
  add_barrier();
}

void generator::visit_clock_inst(ir::clock_inst* clock){
  InlineAsm *iasm = InlineAsm::get(FunctionType::get(builder_->getInt64Ty(), {}), "mov.u64 $0, %clock64;", "=l", true);
  vals_[clock][{}] = call(iasm);
}

void generator::visit_globaltimer_inst(ir::globaltimer_inst* timer){
  InlineAsm *iasm = InlineAsm::get(FunctionType::get(builder_->getInt64Ty(), {}), "mov.u64 $0, %globaltimer;", "=l", true);
  vals_[timer][{}] = call(iasm);
}



void generator::visit_prefetch_s_inst(ir::prefetch_s_inst *i) {
  ir::value *v = i->get_operand(0);
  int inc = i->get_inc();
  if (inc == 0) {
    // If dot has not been visitied, do nothing.
  } else {
    // If dot has been visitied, insert prefetched lds
    assert(inc == 1);
    assert(prefetch_latch_to_bb_.find(v) != prefetch_latch_to_bb_.end() &&
           "dot hasn't be visited");
    // sink lds & extract element
    // move lds & all uses to current location
    std::stack<Value*> work_stack;
    for (Value *value : prefetch_latch_to_bb_[v])
      work_stack.push(value);
    std::vector<Instruction*> dead_instrs;
    while (!work_stack.empty()) {
      Value *m = work_stack.top();
      work_stack.pop();

      for (auto u : m->users())
        work_stack.push(u);

      assert(isa<Instruction>(m));
      auto m_instr = static_cast<Instruction*>(m);

      m_instr->removeFromParent();
      m_instr->insertAfter(&*std::prev(builder_->GetInsertBlock()->end()));
      assert(m_instr->getParent() == &*builder_->GetInsertBlock());
      builder_->SetInsertPoint(m_instr->getParent());
    }
  }
}

void generator::visit_async_wait_inst(ir::async_wait_inst* i) {
  std::string asm_str = "cp.async.wait_group " + std::to_string(i->get_N()) + ";";
  InlineAsm *iasm = InlineAsm::get(FunctionType::get(void_ty, {}), asm_str, "", true);
  call(iasm);
}

/**
 * \brief Code Generation for `extern_elementwise`
 */
void generator::visit_extern_elementwise_inst(ir::extern_elementwise_inst *i) {
  std::vector<Type *> operand_types;
  for (size_t j = 0; j < i->get_num_operands(); j++) {
    operand_types.push_back(
        cvt(i->get_operand(j)->get_type()->get_scalar_ty()));
  }
  Type *ret_type = cvt(i->get_type()->get_scalar_ty());
  FunctionType *FT =
      FunctionType::get(ret_type, std::move(operand_types), false);
  Function *F = llvm::cast<llvm::Function>(
      mod_->getOrInsertFunction(i->get_symbol_name(), FT).getCallee());
  for (auto idx : idxs_.at(i)) {
    std::vector<llvm::Value *> args;
    for (size_t j = 0; j < i->get_num_operands(); j++) {
      args.emplace_back(vals_[i->get_operand(j)][idx]);
    }
    vals_[i][idx] = call(F, std::move(args));
  }
  add_extern_lib(i->get_lib_name(), i->get_lib_path());
}

//void generator::visit_make_range_dyn(ir::make_range_dyn* x) {
//  for(indices_t idx: idxs_.at(x)){
//    assert(idx.size() == 1);
//    if(idx[0] == i32(0))
//      vals_[x][idx] = idx[0];
//    else{
//      BinaryOperator *bin_add = dyn_cast<BinaryOperator>(idx[0]);
//      assert(bin_add);
//      vals_[x][idx] = bin_add->getOperand(0);
//    }
//  }
//}

//void generator::visit_make_range_sta(ir::make_range_sta* x) {
//  for(indices_t idx: idxs_.at(x)){
//    assert(idx.size() == 1);
//    if(idx[0] == i32(0)){
//      vals_[x][idx] = idx[0];
//    }
//    else{
//      BinaryOperator *bin_add = dyn_cast<BinaryOperator>(idx[0]);
//      assert(bin_add);
//      Value *cst = bin_add->getOperand(1);
//      assert(isa<Constant>(cst));
//      vals_[x][idx] = cst;
//    }
//  };
//}

void generator::visit_make_range(ir::make_range* x) {
  for(indices_t idx: idxs_.at(x)){
    Value* start = ConstantInt::get(idx[0]->getType(), x->get_first()->get_value());
    vals_[x][idx] = add(start, idx[0]);
  }
}

void generator::visit_undef_value(ir::undef_value *x) {
  ir::type* sca_ty = x->get_type()->get_scalar_ty();
  Type* ty = cvt(sca_ty);
  for(indices_t idx: idxs_.at(x))
    vals_[x][idx] = llvm::UndefValue::get(ty);
}

void generator::visit_constant_int(ir::constant_int *x){
  Type *ty = cvt(x->get_type()->get_scalar_ty());
  for(indices_t idx: idxs_.at(x))
    vals_[x][idx] = ConstantInt::get(ty, x->get_value());
}

void generator::visit_constant_fp(ir::constant_fp *x){
  Type *ty = cvt(x->get_type()->get_scalar_ty());
  for(indices_t idx: idxs_.at(x)) {
    // manually select bf16 constant
    if (x->get_type()->get_scalar_ty()->is_bf16_ty()) {
      // highest 16 bits of fp32
      float fp32_value = x->get_value();
      uint16_t bf16_raw = (*reinterpret_cast<uint32_t*>(&fp32_value)
                            & 0xffff0000) >> 16;
      std::stringstream const_str;
      const_str << "0x" << std::hex << bf16_raw << "U"; // unsigned
      InlineAsm *bf16_const = InlineAsm::get(FunctionType::get(bf16_ty, {}, false),
                                             " mov.b16 $0, " + const_str.str() + ";",
                                             "=h", false);
      vals_[x][idx] = builder_->CreateCall(bf16_const, {});
    } else
      vals_[x][idx] = ConstantFP::get(ty, x->get_value());
  }
}

void generator::visit_alloc_const(ir::alloc_const *alloc) {
  unsigned size = ((ir::constant_int*)alloc->get_operand(0))->get_value();
  Type *element_ty = cvt(alloc->get_type()->get_pointer_element_ty());
  Type *array_ty = llvm::ArrayType::get(element_ty, size);
  Value *array = new llvm::GlobalVariable(*mod_, array_ty, false, llvm::GlobalVariable::ExternalLinkage,
                                            nullptr, alloc->get_name(), nullptr, llvm::GlobalVariable::NotThreadLocal, 4);
  vals_[alloc][{}] = bit_cast(array, element_ty->getPointerTo(4));
}


void generator::forward_declare(ir::function* fn){
  FunctionType *fn_ty = (FunctionType*)cvt(fn->get_fn_type());
  if(!tgt_->is_gpu()){
    Type *fn_ret_ty = fn_ty->getReturnType();
    std::vector<Type*> fn_args_ty;
    for(unsigned i = 0; i < fn_ty->getNumParams(); i++)
      fn_args_ty.push_back(fn_ty->getParamType(i));
    fn_args_ty.push_back(i32_ty);
    fn_args_ty.push_back(i32_ty);
    fn_args_ty.push_back(i32_ty);
    fn_ty = FunctionType::get(fn_ret_ty, fn_args_ty, false);
  }
  Function *ret = Function::Create(fn_ty, Function::ExternalLinkage, fn->get_name(), mod_);
  fns_[fn] = ret;
}

Value *generator::cast_shared_layout_ptr(analysis::data_layout *layout,
                                         Type *ty) {
  unsigned addr_space = shmem_->getType()->getPointerAddressSpace();
  Value *base = bit_cast(shared_ptr_.at(layout), ptr_ty(ty, addr_space));
  return base;
}

void addAttribute(Function* fn, unsigned id, Attribute attr) {
#if LLVM_VERSION_MAJOR >= 15
  fn->addAttributeAtIndex(id, attr);
#else
  fn->addAttribute(id, attr);
#endif
}

void generator::visit_function(ir::function* fn) {
  idxs_.clear();
  vals_.clear();
  seen_.clear();
  LLVMContext &ctx = builder_->getContext();

  Function* ret = fns_[fn];


  // set attributes
  for(auto attr_pair: fn->attrs()){
    unsigned id = attr_pair.first;
    for(ir::attribute attr: attr_pair.second)
    if(attr.is_llvm_attr()){
      llvm::Attribute llattr = cvt(attr);
      if(llattr.getKindAsEnum() != llvm::Attribute::None)
        addAttribute(ret, id, llattr);
    }
  }
  // set metadata
  if(tgt_->is_gpu()){
      tgt_->set_kernel(*builder_, ctx, mod_, ret);
      Metadata *md_args[] = {
        ValueAsMetadata::get(ret),
        MDString::get(ctx, "maxntidx"),
        ValueAsMetadata::get(i32(num_warps_*32))
      };
      mod_->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(MDNode::get(ctx, md_args));
  }
  // set arguments
  for(unsigned i = 0; i < fn->args().size(); i++)
    vals_[fn->args()[i]][{}] = &*(ret->arg_begin() + i);
  // create blocks
  auto blocks = ir::cfg::reverse_post_order(fn);
  for(ir::basic_block *block: blocks) {
    BasicBlock *dst_block = BasicBlock::Create(ctx, block->get_name(), ret);
    bbs_[block] = dst_block;
  }
  builder_->SetInsertPoint(bbs_[fn->blocks()[0]]);
  // create policies
  if(tgt_->as_nvidia()->sm() >= 80)
  for(ir::load_inst::EVICTION_POLICY evict: {ir::load_inst::EVICT_FIRST, ir::load_inst::EVICT_LAST}){
    std::string policy = (evict == ir::load_inst::EVICT_FIRST) ? "evict_first" : "evict_last";
    std::string asm_str = "createpolicy.fractional.L2::" + policy + ".b64 $0, 1.0;";
    InlineAsm* iasm = InlineAsm::get(FunctionType::get(i64_ty, {}), asm_str, "=l", false);
    policies_[evict] = call(iasm);
  }
  // initialize layouts
  for(auto x: layouts_->get_all()){
    visit_layout(x.second);
  }
  // generate LLVM-IR code
  for(ir::basic_block *block: blocks)
    visit_basic_block(block);
  // finalize
  finalize_function(fn);
}



void generator::visit_layout_mma(analysis::mma_layout* layout) {
  ir::value *a = nullptr;
  ir::value *b = nullptr;
  for(ir::value* v: layout->get_values())
    if(ir::dot_inst* dot = dynamic_cast<ir::dot_inst*>(v)){
      a = dot->get_operand(0);
      b = dot->get_operand(1);
    }
  analysis::data_layout* layout_a = layouts_->get(a);
  analysis::data_layout* layout_b = layouts_->get(b);

  const auto& shape = layout->get_shape();
  Value *_1 = i32(1);
  Value *_2 = i32(2);
  Value *_3 = i32(3);
  Value *_4 = i32(4);
  Value *_8 = i32(8);
  Value *_16 = i32(16);
  Value *_32 = i32(32);
  int cc = tgt_->as_nvidia()->sm();
  std::vector<Value*> idx_m;
  std::vector<Value*> idx_n;
  std::vector<Value*> idx_z;
  //
  Value* thread = tgt_->get_local_id(mod_, *builder_, 0);
  Value *lane = urem(thread, _32);
  Value *warp = udiv(thread, _32);
  /* lane offset */
  if(cc < 80){
    auto ord_a = layout_a->get_order();
    auto ord_b = layout_b->get_order();
    bool is_a_row = ord_a[0] != 0;
    bool is_b_row = ord_b[0] != 0;
    /* warp offset */
    Value *warp_0 = urem(warp, i32(layout->wpt(0)));
    Value *warp_12 = udiv(warp, i32(layout->wpt(0)));
    Value *warp_1 = urem(warp_12, i32(layout->wpt(1)));
    Value *off_warp_m = mul(warp_0, i32(layout->spw(0)));
    Value *off_warp_n = mul(warp_1, i32(layout->spw(1)));
    // Quad offset
    Value *off_quad_m = mul(udiv(and_(lane, _16), _4), i32(layout->fpw(0)));
    Value *off_quad_n = mul(udiv(and_(lane, _16), _4), i32(layout->fpw(1)));
    // Pair offset
    Value *off_pair_m = udiv(urem(lane, _16), _4);
    off_pair_m = urem(off_pair_m, i32(layout->fpw(0)));
    off_pair_m = mul(off_pair_m, i32(4));
    Value *off_pair_n = udiv(urem(lane, _16), _4);
    off_pair_n = udiv(off_pair_n, i32(layout->fpw(0)));
    off_pair_n = urem(off_pair_n, i32(layout->fpw(1)));
    off_pair_n = mul(off_pair_n, i32(4));
    // scale
    off_pair_m = mul(off_pair_m, i32(layout->rep(0)/2));
    off_quad_m = mul(off_quad_m, i32(layout->rep(0)/2));
    off_pair_n = mul(off_pair_n, i32(layout->rep(1)/2));
    off_quad_n = mul(off_quad_n, i32(layout->rep(1)/2));
    // Quad pair offset
    Value *off_lane_m = add(off_pair_m, off_quad_m);
    Value *off_lane_n = add(off_pair_n, off_quad_n);
    // a offset
    offset_a_m_[layout] = add(off_warp_m, off_lane_m);
    offset_a_k_[layout] = and_(lane, _3);
    // b offsets
    offset_b_n_[layout] = add(off_warp_n, off_lane_n);
    offset_b_k_[layout] = and_(lane, _3);
    // i indices
    Value *offset_c_m = add(and_(lane, _1), offset_a_m_[layout]);
    for(unsigned m = 0; m < shape[0]; m+=layout->shape_per_cta(0))
    for(unsigned mm = 0; mm < layout->rep(0); mm++)
      idx_m.push_back(add(offset_c_m, i32(m + mm*2)));
    // j indices
    Value *offset_c_n = add(and_(lane, _2), add(off_warp_n, off_pair_n));
    for(unsigned n = 0; n < shape[1]; n+=layout->shape_per_cta(1))
    for(unsigned nn = 0; nn < layout->rep(1); nn++){
      idx_n.push_back(add(offset_c_n, i32(n + nn/2*4 + (nn%2)*2*layout->fpw(1)*layout->rep(1))));
      idx_n.push_back(add(offset_c_n, i32(n + nn/2*4 + (nn%2)*2*layout->fpw(1)*layout->rep(1) + 1)));
    }
    if(is_a_row){
      offset_a_m_[layout] = add(offset_a_m_[layout], urem(thread, i32(4)));
      offset_a_k_[layout] = i32(0);
    }
    if(!is_b_row){
      offset_b_n_[layout] = add(offset_b_n_[layout], urem(thread, i32(4)));
      offset_b_k_[layout] = i32(0);
    }
    /* axes */
    axes_[layout->get_axis(0)] = distributed_axis{1, idx_m, warp_0};
    axes_[layout->get_axis(1)] = distributed_axis{1, idx_n, warp_1};
  }
  else{
    /* warp offset */
    Value *warp_0 = urem(warp, i32(layout->wpt(0)));
    Value *warp_1 = urem(udiv(warp, i32(layout->wpt(0))), i32(layout->wpt(1)));
    Value *off_warp_m = mul(warp_0, i32(layout->spw(0)));
    Value *off_warp_n = mul(warp_1, i32(layout->spw(1)));
    Value *off_lane_m = urem(lane, _16);
    Value *off_lane_n = urem(lane, _8);
    /* offsets */
    // a offset
    offset_a_m_[layout] = add(off_warp_m, off_lane_m);
    offset_a_k_[layout] = i32(0);
    // b offsets
    offset_b_n_[layout] = add(off_warp_n, off_lane_n);
    offset_b_k_[layout] = i32(0);
    // c offset
    Value *off_c_m = add(udiv(lane, _4), off_warp_m);
    Value *off_c_n = add(mul(_2, urem(lane, _4)), off_warp_n);
    for(unsigned m = 0; m < shape[0]; m+=layout->shape_per_cta(0)){
      idx_m.push_back(add(off_c_m, i32(m)));
      idx_m.push_back(add(off_c_m, i32(m + 8)));
    }
    for(unsigned n = 0; n < shape[1]; n+=layout->shape_per_cta(1)){
      idx_n.push_back(add(off_c_n, i32(n)));
      idx_n.push_back(add(off_c_n, i32(n + 1)));
    }
    /* axes */
    axes_[layout->get_axis(0)] = distributed_axis{1, idx_m, warp_0};
    axes_[layout->get_axis(1)] = distributed_axis{1, idx_n, warp_1};
  }
}

void generator::visit_layout_scanline(analysis::scanline_layout* layout) {
  Value* thread_id = tgt_->get_local_id(mod_, *builder_, 0);
  auto order = layout->get_order();
  const auto& shape = layout->get_shape();
  // Delinearize
  size_t dim = shape.size();
  std::vector<Value*> thread_ids(dim);
  for(unsigned k = 0; k < dim - 1; k++){
    Constant *dim_k = i32(layout->mts(order[k]));
    Value *rem = urem(thread_id, dim_k);
    thread_id = udiv(thread_id, dim_k);
    thread_ids[order[k]] = rem;
  }
  Constant *dim_k = i32(layout->mts(order[dim - 1]));
  thread_ids[order[dim - 1]] = urem(thread_id, dim_k);

  // Create axes
  for(unsigned k = 0; k < dim; k++) {
    int nts = layout->nts(k);
    int mts = layout->mts(k);
    std::string str_k = std::to_string(k);
    Value *contiguous_k = i32(nts);
    Value *scaled_thread_ids = mul(thread_ids[k], contiguous_k);
    unsigned per_cta  = layout->shape_per_cta(k);
    unsigned per_thread = nts * shape[k] / per_cta;
    std::vector<Value*> idx_list(per_thread);
    for(unsigned n = 0 ; n < per_thread; n++){
      unsigned offset = n / nts * per_cta + n % nts;
      idx_list[n] = add(scaled_thread_ids, i32(offset), "idx_" + str_k + "_" + std::to_string(n));
    }
    axes_[layout->get_axis(k)] = distributed_axis{nts, idx_list, thread_ids[k]};
  }
}

void generator::visit_layout_shared(analysis::shared_layout* layout) {
  Type* ty = cvt(layout->get_type());
  PointerType *ptr_ty = ty->getPointerTo(shmem_->getType()->getPointerAddressSpace());
  if (layout->get_N_buffer()) {
    // create pointers
    shared_pre_ptr_[layout] = gep(shmem_, i32(alloc_->offset(layout)));
    shared_pre_ptr_[layout] = bit_cast(shared_pre_ptr_[layout], ptr_ty);

    BasicBlock *current = builder_->GetInsertBlock();

    auto info = *layout->get_N_buffer();
    ir::phi_node *phi = info.phi;
    BasicBlock *parent = bbs_.at(phi->get_parent());
    if(parent->empty())
      builder_->SetInsertPoint(parent);
    else if (const Instruction *first_non_phi = &*parent->getFirstNonPHI()) {
      builder_->SetInsertPoint(&*parent->getFirstNonPHI());
    } else
      builder_->SetInsertPoint(parent);

    // create smem_idx
    read_smem_idx_[layout] = phi(i32_ty, 2);
    write_smem_idx_[layout] = phi(i32_ty, 2);

    // create pointers
    // ptr of the current iteration
    shared_ptr_[layout] = phi(ptr_ty, 2);
    // ptr of the next iteration
    shared_next_ptr_[layout] = phi(ptr_ty, 2);

    builder_->SetInsertPoint(current);
  } else if(layout->get_double_buffer()) {
    BasicBlock *current = builder_->GetInsertBlock();
    auto info = *layout->get_double_buffer();
    ir::phi_node *phi = info.phi;
    BasicBlock *parent = bbs_.at(phi->get_parent());
    if(parent->empty())
      builder_->SetInsertPoint(parent);
    else
      builder_->SetInsertPoint(&*parent->getFirstNonPHI());
    // create pointers
    shared_ptr_[layout] = phi(ptr_ty, 2);
    shared_pre_ptr_[layout] = gep(shmem_, i32(alloc_->offset(layout)));
    shared_pre_ptr_[layout] = bit_cast(shared_pre_ptr_[layout], shared_ptr_[layout]->getType());
    shared_off_[layout] = phi(i32_ty, 2);
    shared_next_ptr_[layout] = gep(shared_ptr_[layout], shared_off_[layout], "next_ptr");
    builder_->SetInsertPoint(current);
  } else{
    size_t offset = alloc_->offset(layout);
    shared_ptr_[layout] = gep(shmem_, i32(offset));
    shared_ptr_[layout] = bit_cast(shared_ptr_[layout], ptr_ty);
  }
}

void generator::visit_basic_block(ir::basic_block * block) {

  BasicBlock *parent = bbs_[block];
  builder_->SetInsertPoint(parent);
  for(ir::instruction *i: block->get_inst_list()){
    visit_value(i);
    // std::cout << "done" << std::endl;
  }
  // Update ir bb -> llvm bb mapping
  bbs_[block] = builder_->GetInsertBlock();
}

void generator::visit_argument(ir::argument* arg) {

}

void generator::init_idx(ir::value *v) {
  idxs_[v].clear();
  if(!v->get_type()->is_block_ty()){
    idxs_[v].push_back({});
    return;
  }
  if(layouts_->get(v)->to_shared())
    return;
  const auto &shapes = v->get_type()->get_block_shapes();
  size_t rank = shapes.size();
  std::vector<distributed_axis> axes(rank);
  std::vector<int> ord(rank);
  // compute axes
  // std::cout << "axes" << std::endl;
  for(size_t d = 0; d < shapes.size(); d++){
    // std::cout << d << " " << shapes[d] << std::endl;
    // std::cout << a_axes_->get(v, d) << std::endl;
    if(shapes[d] > 1){
      unsigned x = a_axes_->get(v, d);
      axes[d] = axes_.at(x);
    }
    else{
      axes[d].contiguous = 1;
      axes[d].values = {i32(0)};
    }
  }
  // std::cout << "axes ok" << std::endl;
  // compute order
  analysis::data_layout* layout = layouts_->get(v);
  std::iota(ord.begin(), ord.end(), 0);
  auto cmp = [&](int x, int y) {
    unsigned axx = a_axes_->get(v, x);
    unsigned axy = a_axes_->get(v, y);
    size_t posx = layout->find_axis(axx);
    size_t posy = layout->find_axis(axy);
    if(posx < rank && posy < rank)
      return layout->get_order(posx) < layout->get_order(posy);
    return false;
  };
  std::sort(ord.begin(), ord.end(), cmp);
  ords_[v] = ord;
  // indices
  if(axes.size() == 1)
    for(Value* x0: axes[ord[0]].values){
      idxs_[v].push_back({x0});
    }
  if(axes.size() == 2)
    for(Value* x1: axes[ord[1]].values)
    for(Value* x0: axes[ord[0]].values){
      indices_t idx(2);
      idx[ord[0]] = x0;
      idx[ord[1]] = x1;
      idxs_[v].push_back(idx);
    }
  if(axes.size() == 3)
    for(Value* x2: axes[ord[2]].values)
    for(Value* x1: axes[ord[1]].values)
    for(Value* x0: axes[ord[0]].values){
      indices_t idx(3);
      idx[ord[0]] = x0;
      idx[ord[1]] = x1;
      idx[ord[2]] = x2;
      idxs_[v].push_back(idx);
    }
}

void generator::finalize_shared_layout(analysis::shared_layout *shared) {
  if (auto n_buffer = shared->get_N_buffer()) {
    // if (*_smem_idx == #stages-1) {
    //   *_smem_idx = 0;
    // } else *_smem_idx++;
    auto finalize_smem_idx = [&](auto &smem_idx, int init_stage) {
      // insert point
      Value *idx = smem_idx[shared];
      builder_->SetInsertPoint(bbs_.at(n_buffer->phi->get_parent())->getTerminator());
      Value *cond = icmp_eq(idx, i32(shared->get_num_stages()-1));
      PHINode *_ret = phi(i32_ty, 2);
      Instruction *then_term = nullptr;
      Instruction *else_term = nullptr;
      Instruction *dummy = builder_->CreateRet(nullptr);
      llvm::SplitBlockAndInsertIfThenElse(cond, _ret, &then_term, &else_term, nullptr);
      dummy->removeFromParent();
      builder_->SetInsertPoint(then_term);
      Value *zero_smem_idx = i32(0);
      builder_->SetInsertPoint(else_term);
      Value *inc_smem_idx = add(idx, i32(1));
      builder_->SetInsertPoint(_ret->getParent());
      _ret->addIncoming(zero_smem_idx, then_term->getParent());
      _ret->addIncoming(inc_smem_idx, else_term->getParent());
      // update ir::bb -> llvm::bb mapping
      bbs_.at(n_buffer->phi->get_parent()) = builder_->GetInsertBlock();
      // idx = init_stage;
      // loop: ...
      if (auto idx_phi = llvm::dyn_cast<PHINode>(smem_idx[shared])) {
        idx_phi->addIncoming(i32(init_stage), bbs_.at(n_buffer->phi->get_incoming_block(0)));
        idx_phi->addIncoming(_ret, bbs_.at(n_buffer->phi->get_incoming_block(1)));
      } else
        throw std::runtime_error("Should be PHINode");
    };

    // read_smem_idx is used by next_ptr to compute the next iteration value, so init value is 2
    finalize_smem_idx(read_smem_idx_, 2);
    finalize_smem_idx(write_smem_idx_, shared->get_num_stages()-1);

    // finalize pointers
    ir::phi_node *pn = n_buffer->phi;
    BasicBlock *header = bbs_.at(pn->get_incoming_block(0));
    BasicBlock *loop = bbs_.at(pn->get_incoming_block(1));
    // %curr_ptr = phi %shared_pre_ptr, %next_ptr
    // %next_ptr = phi %shared_pre_ptr[+1], (gep(%pre_ptr, read_smem_idx*per_stage_size))
    if (auto curr_ptr = dyn_cast<PHINode>(shared_ptr_[shared])) {
      curr_ptr->addIncoming(shared_pre_ptr_[shared], header);
      curr_ptr->addIncoming(shared_next_ptr_[shared], loop);
    } else
      throw std::runtime_error("Should be PHINode");

    BasicBlock *current = builder_->GetInsertBlock();
    builder_->SetInsertPoint(header->getTerminator());
    Value *next_ptr_header = gep(shared_pre_ptr_[shared], i32(shared->get_per_stage_elements()));
    builder_->SetInsertPoint(current->getTerminator());

    assert(isa<PHINode>(shared_next_ptr_[shared]));
    static_cast<PHINode*>(shared_next_ptr_[shared])->addIncoming(next_ptr_header, header);

    Value *lds_offset = mul(read_smem_idx_[shared], i32(shared->get_per_stage_elements()));
    Value *next_ptr = gep(shared_pre_ptr_[shared], lds_offset);
    static_cast<PHINode*>(shared_next_ptr_[shared])->addIncoming(next_ptr, loop);
  } else if(shared->get_double_buffer()) {
    auto info = *shared->get_double_buffer();
    ir::phi_node *phi = info.phi;
    PHINode *ptr = (PHINode*)shmems_[phi];
    PHINode *offset = (PHINode*)shoffs_[phi];
    for(unsigned n = 0; n < phi->get_num_incoming(); n++){
      ir::basic_block* inc_block = phi->get_incoming_block(n);
      ir::value* inc_val = phi->get_incoming_value(n);
      BasicBlock *llvm_inc_block = bbs_.at(inc_block);
      if(inc_val == info.latch){
        builder_->SetInsertPoint(llvm_inc_block->getTerminator());
        Value *next_offset = neg(offset);
        offset->addIncoming(next_offset, llvm_inc_block);
      }
      else {
        unsigned num_bytes = shared->get_type()->get_primitive_size_in_bits() / 8;
        offset->addIncoming(i32(shared->get_size() / (2*num_bytes)), llvm_inc_block);
      }
      ptr->addIncoming(shmems_[inc_val], llvm_inc_block);
    }
  }
}

void generator::finalize_function(ir::function *fn) {
  // finalize double-buffering
  for(const auto& x: layouts_->get_all())
  if(auto *shared = dynamic_cast<analysis::shared_layout*>(x.second))
    finalize_shared_layout(shared);
  // finalize phi
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *inst: block->get_inst_list())
    if(auto *phi = dynamic_cast<ir::phi_node*>(inst))
      finalize_phi_node(phi);
  for(auto& x: lazy_phi_incs_)
    std::get<0>(x)->addIncoming(std::get<1>(x), bbs_[std::get<2>(x)]);
}

void generator::finalize_phi_node(ir::phi_node *x) {
  if(shmems_.find(x) != shmems_.end())
    return;
  for(unsigned n = 0; n < x->get_num_incoming(); n++){
    ir::basic_block *_block = x->get_incoming_block(n);
    BasicBlock *block = bbs_.at(_block);
    for(indices_t idx: idxs_.at(x)){
      PHINode *phi = (PHINode*)vals_[x][idx];
      Value *inc = vals_[x->get_incoming_value(n)][idx];
      // x->print(std::cout);
      phi->addIncoming(inc, block);
    }
  }
}

void generator::packed_type(ir::value* i){
  Type* dtype = cvt(i->get_type()->get_tile_element_ty());
  auto* layout = dynamic_cast<analysis::scanline_layout*>(layouts_->get(i));
  assert(layout);
}

void generator::visit(ir::module &src, llvm::Module &dst) {
  mod_ = &dst;
  ctx_ = &dst.getContext();
  builder_ = new Builder(*ctx_);
  // allocate shared memory
  if(tgt_->is_gpu())
  if(unsigned alloc_size = alloc_->allocated_size()){
    Type *int_8_ty = Type::getInt8Ty(*ctx_);
    Type *int_32_ty = Type::getInt32Ty(*ctx_);
    ArrayType *array_ty = ArrayType::get(int_32_ty, 0);
    Type *ptr_ty = ptr_ty(int_8_ty, 3);
    GlobalVariable *sh_mem_array =
      new GlobalVariable(*mod_, array_ty, false, GlobalVariable::ExternalLinkage,
                         nullptr, "__shared_ptr", nullptr, GlobalVariable::NotThreadLocal, 3);
    shmem_ = bit_cast(sh_mem_array, ptr_ty);
  }
  // instantiate device functions
//  for(ir::function *fn: src.get_function_list())
//  for(ir::basic_block *bb: fn->blocks())
//  for(ir::instruction *i: bb->get_inst_list())
//  if(auto *call = dynamic_cast<ir::call_inst*>(i)){
//    std::cout << "call??" << std::endl;
//  }
  // visit functions
  for(ir::function *fn: src.get_function_list())
    forward_declare(fn);
  for(ir::function *fn: src.get_function_list())
    visit_function(fn);
}

void generator::add_extern_lib(const std::string &lib_name,
                               const std::string &lib_path) {
  if (extern_lib_map_.count(lib_name) == 0) {
    extern_lib_map_[lib_name] = create_extern_lib(lib_name, lib_path);
  } else if (extern_lib_map_.at(lib_name)->path() != lib_path) {
    throw std::runtime_error("A library has multiple paths (1) " + lib_path +
                             " (2) " + extern_lib_map_.at(lib_name)->path());
  }
}

}  // namespace codegen
}  // namespace triton
