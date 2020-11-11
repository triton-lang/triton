#include <numeric>
#include "triton/codegen/selection/generator.h"
#include "triton/codegen/selection/machine_layout.h"
#include "triton/codegen/selection/machine_value.h"
#include "triton/codegen/target.h"
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/transform/coalesce.h"
#include "triton/ir/context.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/type.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/InlineAsm.h"

namespace triton{
namespace codegen{

using namespace llvm;

// Function for extended Euclidean Algorithm
inline int gcd_impl(int a, int b, int *x, int *y)
{
    // Base Case
    if (a == 0)
    {
        *x = 0;
        *y = 1;
        return b;
    }
    int x1, y1; // To store results of recursive call
    int gcd = gcd_impl(b%a, a, &x1, &y1);
    // Update x and y using results of
    // recursive call
    *x = y1 - (b/a) * x1;
    *y = x1;
    return gcd;
}

inline int gcd(int a, int b) {
  int x, y;
  return gcd_impl(a, b, &x, &y);
}


llvm::Instruction::BinaryOps llvm_op(ir::binary_op_t op) {
  using llop = llvm::Instruction::BinaryOps;
  using ttop = ir::binary_op_t;
  switch(op) {
    case ttop::Add: return llop::Add;
    case ttop::FAdd: return llop::FAdd;
    case ttop::Sub: return llop::Sub;
    case ttop::FSub: return llop::FSub;
    case ttop::Mul: return llop::Mul;
    case ttop::FMul: return llop::FMul;
    case ttop::UDiv: return llop::UDiv;
    case ttop::SDiv: return llop::SDiv;
    case ttop::FDiv: return llop::FDiv;
    case ttop::URem: return llop::URem;
    case ttop::SRem: return llop::SRem;
    case ttop::FRem: return llop::FRem;
    case ttop::Shl: return llop::Shl;
    case ttop::LShr: return llop::LShr;
    case ttop::AShr: return llop::AShr;
    case ttop::And: return llop::And;
    case ttop::Or: return llop::Or;
    case ttop::Xor: return llop::Xor;
  }
  throw std::runtime_error("unknown operator");
}

llvm::Instruction::CastOps llvm_op(ir::cast_op_t op) {
  using llop = llvm::Instruction::CastOps;
  using ttop = ir::cast_op_t;
  switch(op){
  case ttop::Trunc: return llop::Trunc;
  case ttop::ZExt: return llop::ZExt;
  case ttop::SExt: return llop::SExt;
  case ttop::FPTrunc: return llop::FPTrunc;
  case ttop::FPExt: return llop::FPExt;
  case ttop::UIToFP: return llop::UIToFP;
  case ttop::SIToFP: return llop::SIToFP;
  case ttop::FPToUI: return llop::FPToUI;
  case ttop::FPToSI: return llop::FPToSI;
  case ttop::PtrToInt: return llop::PtrToInt;
  case ttop::IntToPtr: return llop::IntToPtr;
  case ttop::BitCast: return llop::BitCast;
  case ttop::AddrSpaceCast: return llop::AddrSpaceCast;
  }
  throw std::runtime_error("unknown operator");
}

llvm::CmpInst::Predicate llvm_pred(ir::cmp_pred_t pred) {
  using llop = llvm::CmpInst::Predicate;
  using ttop = ir::cmp_pred_t;
  switch(pred){
    case ttop::FIRST_FCMP_PREDICATE: return llop::FIRST_FCMP_PREDICATE;
    case ttop::FCMP_FALSE: return llop::FCMP_FALSE;
    case ttop::FCMP_OEQ: return llop::FCMP_OEQ;
    case ttop::FCMP_OGT: return llop::FCMP_OGT;
    case ttop::FCMP_OGE: return llop::FCMP_OGE;
    case ttop::FCMP_OLT: return llop::FCMP_OLT;
    case ttop::FCMP_OLE: return llop::FCMP_OLE;
    case ttop::FCMP_ONE: return llop::FCMP_ONE;
    case ttop::FCMP_ORD: return llop::FCMP_ORD;
    case ttop::FCMP_UNO: return llop::FCMP_UNO;
    case ttop::FCMP_UEQ: return llop::FCMP_UEQ;
    case ttop::FCMP_UGT: return llop::FCMP_UGT;
    case ttop::FCMP_UGE: return llop::FCMP_UGE;
    case ttop::FCMP_ULT: return llop::FCMP_ULT;
    case ttop::FCMP_ULE: return llop::FCMP_ULE;
    case ttop::FCMP_UNE: return llop::FCMP_UNE;
    case ttop::FCMP_TRUE: return llop::FCMP_TRUE;
    case ttop::LAST_FCMP_PREDICATE: return llop::LAST_FCMP_PREDICATE;
    case ttop::FIRST_ICMP_PREDICATE: return llop::FIRST_ICMP_PREDICATE;
    case ttop::ICMP_EQ: return llop::ICMP_EQ;
    case ttop::ICMP_NE: return llop::ICMP_NE;
    case ttop::ICMP_UGT: return llop::ICMP_UGT;
    case ttop::ICMP_UGE: return llop::ICMP_UGE;
    case ttop::ICMP_ULT: return llop::ICMP_ULT;
    case ttop::ICMP_ULE: return llop::ICMP_ULE;
    case ttop::ICMP_SGT: return llop::ICMP_SGT;
    case ttop::ICMP_SGE: return llop::ICMP_SGE;
    case ttop::ICMP_SLT: return llop::ICMP_SLT;
    case ttop::ICMP_SLE: return llop::ICMP_SLE;
    case ttop::LAST_ICMP_PREDICATE: return llop::LAST_ICMP_PREDICATE;
  }
  throw std::runtime_error("unknown operator");
}


inline Type *llvm_type(ir::type *ty, LLVMContext &ctx) {
  // function
  if(auto* tt = dynamic_cast<ir::function_type*>(ty)){
    Type *return_ty = llvm_type(tt->get_return_ty(), ctx);
    std::vector<Type*> param_tys;
    std::transform(tt->params_begin(), tt->params_end(), std::back_inserter(param_tys),
                   [&ctx](ir::type* t){ return llvm_type(t, ctx);});
    return FunctionType::get(return_ty, param_tys, false);
  }
  // pointer
  if(ty->is_pointer_ty()){
    Type *elt_ty = llvm_type(ty->get_pointer_element_ty(), ctx);
    unsigned addr_space = ty->get_pointer_address_space();
    return PointerType::get(elt_ty, addr_space);
  }
  // integer
  if(ty->is_integer_ty()){
    unsigned bitwidth = ty->get_integer_bitwidth();
    return IntegerType::get(ctx, bitwidth);
  }
  // primitive types
  switch(ty->get_type_id()){
    case ir::type::VoidTyID:      return Type::getVoidTy(ctx);
    case ir::type::HalfTyID:      return Type::getHalfTy(ctx);
    case ir::type::FloatTyID:     return Type::getFloatTy(ctx);
    case ir::type::DoubleTyID:    return Type::getDoubleTy(ctx);
    case ir::type::X86_FP80TyID:  return Type::getX86_FP80Ty(ctx);
    case ir::type::PPC_FP128TyID: return Type::getPPC_FP128Ty(ctx);
    case ir::type::LabelTyID:     return Type::getLabelTy(ctx);
    case ir::type::MetadataTyID:  return Type::getMetadataTy(ctx);
    case ir::type::TokenTyID:     return Type::getTokenTy(ctx);
    default: break;
  }
  // unknown type
  throw std::runtime_error("unknown conversion from ir::type to Type");
}


inline llvm::Attribute llvm_attr(llvm::LLVMContext& ctx, ir::attribute attr) {
  switch(attr.get_kind()){
    case ir::noalias: return llvm::Attribute::get(ctx, llvm::Attribute::NoAlias);
    case ir::readonly: return llvm::Attribute::get(ctx, llvm::Attribute::ReadOnly);
    case ir::writeonly: return llvm::Attribute::get(ctx, llvm::Attribute::WriteOnly);
    case ir::aligned: return llvm::Attribute::get(ctx, llvm::Attribute::Alignment, attr.get_value());
    case ir::retune: return llvm::Attribute::get(ctx, llvm::Attribute::None);
    default: throw std::runtime_error("cannot convert ir::attribute_t to llvm::Attribute");
  }
}

inline bool is_trans(ir::value *v) {
  if(dynamic_cast<ir::trans_inst *>(v)) {
    return true;
  }
  if(auto *phi = dynamic_cast<ir::instruction *>(v)) {
    bool result = true;
    for(ir::value *op: phi->ops())
      result = result && is_trans(op);
    return result;
  }
  return false;
}




generator::generator(analysis::axes *a_axes,
                    analysis::layouts *layouts,
                    analysis::align *alignment,
                    analysis::allocation *alloc,
                     target *tgt,
                    unsigned num_warps)
  : a_axes_(a_axes), layouts_(layouts), alignment_(alignment), alloc_(alloc),
    tgt_(tgt), num_warps_(num_warps) {

}


void generator::visit_value(ir::value* v) {
  if(!seen_.insert(v).second)
    return;
  // create machine tile
  if(v->get_type()->is_tile_ty()){
    tmap_[v] = machine_layouts_.at(layouts_->get(v))->create(v);
  }
  // visit operands
  BasicBlock *current = builder_->GetInsertBlock();
  auto *inst = dynamic_cast<ir::instruction*>(v);
  if(inst)
    for(ir::value *op: inst->ops()){
      if(dynamic_cast<ir::constant*>(op) || !dynamic_cast<ir::phi_node*>(v))
        visit_value(op);
    }
  // change insert point for phi node
  builder_->SetInsertPoint(current);
  auto *phi = dynamic_cast<ir::phi_node*>(v);
  if(phi && !current->empty() && current->getFirstNonPHI())
    builder_->SetInsertPoint(&*current->getFirstNonPHI());
  // visit user
  if(auto *usr = dynamic_cast<ir::user*>(v))
    usr->accept(this);
  // revert insert point
  if(phi && !current->empty() && current->getFirstNonPHI())
    builder_->SetInsertPoint(current);
}

void generator::visit_phi_node(ir::phi_node* phi) {
  Type *ty = llvm_type(phi->get_type()->get_scalar_ty(), *ctx_);
  unsigned num_ops = phi->get_num_operands();
  for_each(phi, [&](indices_t idx){
    set_value(phi, idx, builder_->CreatePHI(ty, num_ops));
  });
}

void generator::visit_binary_operator(ir::binary_operator*binop) {
  for_each(binop, [&](indices_t idx){
    Value *lhs = get_value(binop->get_operand(0), idx);
    Value *rhs = get_value(binop->get_operand(1), idx);
    Value *ret = builder_->CreateBinOp(llvm_op(binop->get_op()), lhs, rhs);
    set_value(binop, idx, ret);
  });
}

void generator::visit_getelementptr_inst(ir::getelementptr_inst* gep) {
  for_each(gep, [&](indices_t idx){
    Value *ptr = get_value(gep->get_operand(0), idx);
    std::vector<Value*> idx_vals;
    std::transform(gep->idx_begin(), gep->idx_end(), std::back_inserter(idx_vals),
                   [&](ir::value* x){ return get_value(x, idx);});
    Type *source_ty = llvm_type(gep->get_source_elt_ty()->get_scalar_ty(), *ctx_);
    Value *ret = builder_->CreateGEP(source_ty, ptr, idx_vals);
    set_value(gep, idx, ret);
  });
}

void generator::visit_icmp_inst(ir::icmp_inst* icmp) {
  for_each(icmp, [&](indices_t idx){
    ir::cmp_pred_t pred = icmp->get_pred();
    Value *lhs = get_value(icmp->get_operand(0), idx);
    Value *rhs = get_value(icmp->get_operand(1), idx);
    Value *ret = builder_->CreateICmp(llvm_pred(pred), lhs, rhs);
    set_value(icmp, idx, ret);
  });
}

void generator::visit_fcmp_inst(ir::fcmp_inst* fcmp) {
  for_each(fcmp, [&](indices_t idx){
    ir::cmp_pred_t pred = fcmp->get_pred();
    Value *lhs = get_value(fcmp->get_operand(0), idx);
    Value *rhs = get_value(fcmp->get_operand(1), idx);
    Value *ret = builder_->CreateFCmp(llvm_pred(pred), lhs, rhs);
    set_value(fcmp, idx, ret);
  });
}

void generator::visit_cast_inst(ir::cast_inst* cast) {
  for_each(cast, [&](indices_t idx){
    Value *arg = get_value(cast->get_operand(0), idx);
    Type *dst_ty = llvm_type(cast->get_type()->get_scalar_ty(), *ctx_);
    Value *ret = builder_->CreateCast(llvm_op(cast->get_op()), arg, dst_ty);
    set_value(cast, idx, ret);
  });
}

void generator::visit_return_inst(ir::return_inst* rr) {
  ir::value *ret_val = rr->get_return_value();
  builder_->CreateRet(ret_val ? vmap_.at(ret_val) : nullptr);
}

void generator::visit_cond_branch_inst(ir::cond_branch_inst* br) {
  BasicBlock *true_dest  = (BasicBlock*)vmap_.at(br->get_true_dest());
  BasicBlock *false_dest = (BasicBlock*)vmap_.at(br->get_false_dest());
  Value *cond = vmap_.at(br->get_cond());
  builder_->CreateCondBr(cond, true_dest, false_dest);
}

void generator::visit_uncond_branch_inst(ir::uncond_branch_inst* br) {
  BasicBlock *dest = (BasicBlock*)vmap_.at(br->get_dest());
  builder_->CreateBr(dest);
}


void generator::visit_unmasked_load_inst(ir::unmasked_load_inst* x) {
  if(!x->get_type()->is_tile_ty()){
    Value *ptr = get_value(x->get_pointer_operand(), {});
    set_value(x, {}, builder_->CreateLoad(ptr));
    return;
  }
  // find vector size
  ir::value *ptr = x->get_pointer_operand();
  size_t ld = layouts_->get(ptr)->get_order(0);
  unsigned alignment = std::max<int>(alignment_->get(ptr, ld), 1);


  // vector loads
  std::map<unsigned, Value*> packets;
  for_each(x, [&](indices_t idx){
    distributed_tile* result = (distributed_tile*)tmap_.at(x);
    // vector size
    unsigned contiguous = 1;
    if(ld < x->get_type()->get_tile_rank())
      contiguous = result->axis(ld).contiguous;
    unsigned vector_size = gcd(contiguous, alignment);

    unsigned linear = result->get_linear_index(idx);
    unsigned id = linear / vector_size;
    if(linear % vector_size == 0) {
      distributed_tile *pointers = (distributed_tile*)tmap_.at(ptr);
      Value *ptr = pointers->get_value(idx);
      ptr = builder_->CreateBitCast(ptr, PointerType::get(VectorType::get(result->get_ty(), vector_size),
                                                        ptr->getType()->getPointerAddressSpace()));
      packets[id] = builder_->CreateLoad(ptr);
    }
  });

  // extract result element
  for_each(x, [&](indices_t idx){
    distributed_tile* result = (distributed_tile*)tmap_.at(x);
    // vector size
    unsigned contiguous = 1;
    if(ld < x->get_type()->get_tile_rank())
      contiguous = result->axis(ld).contiguous;
    unsigned vector_size = gcd(contiguous, alignment);
    unsigned linear = result->get_linear_index(idx);
    unsigned id = linear / vector_size;
    set_value(x, idx, builder_->CreateExtractElement(packets.at(id), linear % vector_size));
  });
}

void generator::visit_masked_load_inst(ir::masked_load_inst* x) {
  if(!x->get_type()->is_tile_ty()){
    Value *ptr = vmap_.at(x->get_pointer_operand());
    Value *mask = vmap_.at(x->get_mask_operand());
    BasicBlock *current_bb = builder_->GetInsertBlock();
    Function *parent = builder_->GetInsertBlock()->getParent();
    BasicBlock *mask_then_bb = BasicBlock::Create(*ctx_, "mask_then", parent);
    BasicBlock *mask_done_bb = BasicBlock::Create(*ctx_, "mask_done", parent);
    builder_->CreateCondBr(mask, mask_then_bb, mask_done_bb);
    builder_->SetInsertPoint(mask_then_bb);
    Value *result_then = builder_->CreateLoad(ptr);
    builder_->CreateBr(mask_done_bb);
    builder_->SetInsertPoint(mask_done_bb);
    Value *result = nullptr;
    if(x->get_false_value_operand()){
      Value *result_false = vmap_.at(x->get_false_value_operand());
      result = builder_->CreatePHI(result_then->getType(), 2);
      ((PHINode*)result)->addIncoming(result_then, mask_then_bb);
      ((PHINode*)result)->addIncoming(result_false, current_bb);
    }
    else
      result = result_then;
    vmap_[x] = result;
    return;
  }
  // find vector size
  ir::value *ptr = x->get_pointer_operand();
  auto order = layouts_->get(ptr)->get_order();
  size_t ld;
  for(size_t i = 0; i < order.size(); i++){
    ld = order[i];
    if(ld < x->get_type()->get_tile_rank())
      break;
  }
  //size_t ld = layouts_->get(ptr)->get_order(0);
  unsigned alignment = alignment_->get(ptr, ld);
  distributed_tile *pointers = (distributed_tile*)tmap_.at(ptr);
  distributed_tile *masks = (distributed_tile*)tmap_.at(x->get_mask_operand());
  distributed_tile *false_values = (distributed_tile*)tmap_.at(x->get_false_value_operand());
  std::map<unsigned, Value*> packets;
  for_each(x, [&](indices_t idx){
    distributed_tile* result = (distributed_tile*)tmap_.at(x);
    unsigned vector_size = gcd(result->axis(ld).contiguous, alignment);
    unsigned linear = result->get_linear_index(idx);
    unsigned id = linear / vector_size;
    if(linear % vector_size == 0) {
      Value *ptr = pointers->get_value(idx);
      ptr = builder_->CreateBitCast(ptr, PointerType::get(VectorType::get(result->get_ty(), vector_size),
                                                        ptr->getType()->getPointerAddressSpace()));

      Value *mask = masks->get_value(idx);
      BasicBlock *current_bb = builder_->GetInsertBlock();
      Function *parent = builder_->GetInsertBlock()->getParent();
      BasicBlock *mask_then_bb = BasicBlock::Create(*ctx_, "mask_then", parent);
      BasicBlock *mask_done_bb = BasicBlock::Create(*ctx_, "mask_done", parent);
      builder_->CreateCondBr(mask, mask_then_bb, mask_done_bb);
      builder_->SetInsertPoint(mask_then_bb);
      Value *result_then = builder_->CreateLoad(ptr);
      builder_->CreateBr(mask_done_bb);
      builder_->SetInsertPoint(mask_done_bb);
      Value *current_result = nullptr;
      if(false_values){
        current_result = builder_->CreatePHI(result_then->getType(), 2);
        ((PHINode*)current_result)->addIncoming(result_then, mask_then_bb);
        Value *result_false = false_values->get_value(idx);
        if(result_then->getType()->isVectorTy())
          result_false = builder_->CreateVectorSplat(vector_size, result_false);
        ((PHINode*)current_result)->addIncoming(result_false, current_bb);
      }
      else
        current_result = result_then;

//      ConstantInt *cst = nullptr;
//      if(GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(ptr))
//        if(gep->getNumIndices() == 1)
//          cst = dyn_cast<ConstantInt>(gep->idx_begin());
//          llvm::Value* mask = masks->get_value(idx);
//          std::string offset = "";
//          if(cst)
//            offset = " + " + std::to_string(cst->getValue().getSExtValue()*2*vector_size);
//          Type *fp16x2_ty = VectorType::get(builder_->getHalfTy(), 2);
//          Type *fp16x2_pack4_ty = StructType::get(*ctx_, {fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty});
//          FunctionType *ty = FunctionType::get(fp16x2_pack4_ty, {mask->getType(), ptr->getType()}, false);
//          std::string asm_str = "@$0 ld.global.nc.v4.b32 {$1, $2, $3, $4}, [$5" + offset + "];";
//          if(false_values)
//            asm_str += "\n\t@!$0 mov.v4.b32 {$1, $2, $3, $4}, {0, 0, 0, 0};";
//          InlineAsm *iasm = InlineAsm::get(ty, asm_str, "b,=r,=r,=r,=r,l", true);
//          Value *current_result = builder_->CreateCall(iasm, {mask, ptr});

      packets[id] = current_result;
    }
  });
  // extract result element
  for_each(x, [&](indices_t idx){
    distributed_tile* result = (distributed_tile*)tmap_.at(x);
    unsigned vector_size = gcd(result->axis(ld).contiguous, alignment);
    unsigned linear = result->get_linear_index(idx);
    unsigned id = linear / vector_size;
//        Value *tmp = builder_->CreateExtractValue(packets.at(id), {(linear % vector_size) / 2});
//        Value *res = builder_->CreateExtractElement(tmp, (linear % vector_size) % 2);
//        result->set_value(idx, res);
    result->set_value(idx, builder_->CreateExtractElement(packets.at(id), linear % vector_size));
  });
}

void generator::visit_unmasked_store_inst(ir::unmasked_store_inst* st) {
  for_each(st->get_pointer_operand(), [&](indices_t idx){
    Value *ptr = get_value(st->get_pointer_operand(), idx);
    Value *val = get_value(st->get_value_operand(), idx);
     builder_->CreateStore(val, ptr);
  });
}



void generator::visit_masked_store_inst(ir::masked_store_inst* st) {
  distributed_tile* ptrs = (distributed_tile*)tmap_.at(st->get_pointer_operand());
  distributed_tile* masks = (distributed_tile*)tmap_.at(st->get_mask_operand());
  // vector size
  int vector_size = 1;
  int ld = ptrs->get_order()[0];
  unsigned alignment = alignment_->get(st->get_pointer_operand(), ld);
  vector_size = gcd(ptrs->axis(ld).contiguous, alignment);
  // create packets
  std::map<unsigned, Value*> packets;
  ir::value *arg = st->get_value_operand();
  for_each(arg, [&](indices_t idx){
    distributed_tile* in = (distributed_tile*)tmap_.at(arg);
    unsigned linear = in->get_linear_index(idx);
    unsigned id = linear / vector_size;
    Value *in_value = in->get_value(idx);
    if(linear % vector_size == 0)
      packets[id] = UndefValue::get(VectorType::get(in_value->getType(), vector_size));
    packets[id] = builder_->CreateInsertElement(packets.at(id), in_value, linear % vector_size);
  });
  // write-back packets
  for_each(arg, [&](indices_t idx){
    distributed_tile* in = (distributed_tile*)tmap_.at(arg);
    unsigned linear = in->get_linear_index(idx);
    unsigned id = linear / vector_size;
    if(linear % vector_size == 0){
      // fetch tile elements
      Value *elt = packets[id];
      Value *ptr = ptrs->get_value(idx);
      Value *pred = masks->get_value(idx);
      // type information
      Type *ty = elt->getType();
      unsigned nbits = ty->getScalarSizeInBits();
      unsigned nbytes = nbits / 8;
      // extract pointer offset
      std::string offset = "";
      if(GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(ptr))
      if(gep->getNumIndices() == 1)
      if(ConstantInt *cst = dyn_cast<ConstantInt>(gep->idx_begin())){
        offset = " + " + std::to_string(cst->getValue().getSExtValue()*nbytes);
        ptr = gep->getPointerOperand();
      }
      ptr = builder_->CreateBitCast(ptr, ty->getPointerTo(1));
      if(tgt_->is_gpu()){
        // asm argument type
        std::vector<Type*> arg_ty = {pred->getType(), ptr->getType()};
        for(int v = 0; v < vector_size; v++)
          arg_ty.push_back(ty->getScalarType());
        // asm function type
        FunctionType *fn_ty = FunctionType::get(builder_->getVoidTy(), arg_ty, false);
        // asm string
        std::string asm_str;
        asm_str += "@$0 st.global";
        if(vector_size > 1)
          asm_str += ".v" + std::to_string(vector_size);
        asm_str += ".b" + std::to_string(nbits) + " [$1" + offset + "],";
        if(vector_size > 1)
          asm_str += "{";
        for(int v = 0; v < vector_size; v++){
          if(v > 0)
            asm_str += ", ";
          asm_str += "$" + std::to_string(2 + v);
        }
        if(vector_size > 1)
          asm_str += "}";
        asm_str += ";";
        // asm constraint
        std::string constraint = "b,l";
        for(int v = 0; v < vector_size; v++){
          constraint += ",";
          constraint += (nbits == 32 ? "r" : "h");
        }
        // create inline asm
        InlineAsm *iasm = InlineAsm::get(fn_ty, asm_str, constraint, true);
        // call asm
        std::vector<Value*> args = {pred, ptr};
        for(int v = 0; v < vector_size; v++)
          args.push_back(builder_->CreateExtractElement(elt, builder_->getInt32(v)));
        builder_->CreateCall(iasm, args);
      }
      else{
        builder_->CreateMaskedStore(elt, ptr, alignment, builder_->CreateVectorSplat(vector_size, pred));
      }

    }
  });
}


void generator::visit_reshape_inst(ir::reshape_inst* reshape) {
  for_each(reshape, [&](indices_t out_idx){
    distributed_tile* result = (distributed_tile*)tmap_.at(reshape);
    unsigned pos = result->get_linear_index(out_idx);
    ir::value* in = reshape->get_operand(0);
    distributed_tile *in_tile = (distributed_tile*)tmap_.at(in);
    indices_t in_idx = in_tile->get_ordered_indices(pos);
    set_value(reshape, out_idx, get_value(in, in_idx));
  });
}

void generator::visit_splat_inst(ir::splat_inst* splat) {
  Value *in = get_value(splat->get_operand(0), {});
  for_each(splat, [&](indices_t idx){
    set_value(splat, idx, in);
  });
}

void generator::visit_broadcast_inst(ir::broadcast_inst* bcast) {
  ir::value* in = bcast->get_operand(0);
  const auto& in_shapes = in->get_type()->get_tile_shapes();
  distributed_tile *in_tile = (distributed_tile*)tmap_.at(in);
  for_each(bcast, [&](indices_t out_idx){
    indices_t in_idx = out_idx;
    for(size_t k = 0; k < in_idx.size(); k++){
      if(in_shapes[k] == 1)
        in_idx[k] = builder_->getInt32(0);
    }
    set_value(bcast, out_idx, in_tile->get_value(in_idx));
  });
}

void generator::visit_downcast_inst(ir::downcast_inst* x) {
  vmap_[x] = tmap_[x->get_operand(0)]->get_value({builder_->getInt32(0)});
}

void generator::visit_get_program_id_inst(ir::get_program_id_inst* pid) {
  Module *module = builder_->GetInsertBlock()->getModule();
  Value *ret = tgt_->get_block_id(module, *builder_, pid->get_axis());
  vmap_[pid] = ret;
}

void generator::visit_get_num_program_inst(ir::get_num_program_inst* np) {
  Module *module = builder_->GetInsertBlock()->getModule();
  Value *ret = tgt_->get_num_blocks(module, *builder_, np->get_axis());
  vmap_[np] = ret;
}

void generator::visit_exp_inst(ir::exp_inst* x){
  distributed_tile *arg = (distributed_tile*)tmap_.at(x->get_operand(0));
//  Function *fn = builder_->GetInsertBlock()->getParent();
//  Module *module = fn->getParent();
//  Type *ty = llvm_type(x->get_type()->get_scalar_ty(), *ctx_);
//  Function *ex2 = Intrinsic::getDeclaration(module, Intrinsic::nvvm_ex2_approx_ftz_f, {ty});
  Constant *log2e = ConstantFP::get(builder_->getFloatTy(), 1.4426950408889634);
  std::vector<llvm::Type*> tys = {builder_->getFloatTy()};
  FunctionType *fn_ty = FunctionType::get(builder_->getFloatTy(), tys, false);
  InlineAsm *ex2 = InlineAsm::get(fn_ty, "ex2.approx.f32 $0, $1;", "=f,f", false);


  for_each(x, [&](indices_t idx){
    Value *ex2arg = builder_->CreateFMul(arg->get_value(idx), log2e);
    set_value(x, idx, builder_->CreateCall(ex2, std::vector<llvm::Value*>{ex2arg}));
  });
}

void generator::visit_log_inst(ir::log_inst* x){
  distributed_tile *arg = (distributed_tile*)tmap_.at(x->get_operand(0));
//  Function *fn = builder_->GetInsertBlock()->getParent();
//  Module *module = fn->getParent();
//  Type *ty = llvm_type(x->get_type()->get_scalar_ty(), *ctx_);
//  Function *ex2 = Intrinsic::getDeclaration(module, Intrinsic::nvvm_ex2_approx_ftz_f, {ty});
  Constant *rcplog2e = ConstantFP::get(builder_->getFloatTy(), 0.6931471805599453);
  std::vector<llvm::Type*> tys = {builder_->getFloatTy()};
  FunctionType *fn_ty = FunctionType::get(builder_->getFloatTy(), tys, false);
  InlineAsm *lg2 = InlineAsm::get(fn_ty, "lg2.approx.f32 $0, $1;", "=f,f", false);


  for_each(x, [&](indices_t idx){
    Value *lg2arg = builder_->CreateCall(lg2, std::vector<llvm::Value*>{arg->get_value(idx)});
    set_value(x, idx, builder_->CreateFMul(lg2arg, rcplog2e));
  });
}

void generator::visit_atomic_cas_inst(ir::atomic_cas_inst* cas) {
  BasicBlock *current = builder_->GetInsertBlock();
  Module *module = current->getModule();
  Value *tid = tgt_->get_local_id(module, *builder_, 0);
  Value *pred = builder_->CreateICmpEQ(tid, builder_->getInt32(0));
  BasicBlock *tid_0_bb = BasicBlock::Create(*ctx_, "tid_0", current->getParent());
  BasicBlock *tid_0_done_bb = BasicBlock::Create(*ctx_, "tid_0_done", current->getParent());
  tgt_->add_barrier(module, *builder_);
  tgt_->add_memfence(module, *builder_);
  builder_->CreateCondBr(pred, tid_0_bb, tid_0_done_bb);
  builder_->SetInsertPoint(tid_0_bb);
  Value *cas_ptr = vmap_.at(cas->get_operand(0));
  Value *cas_cmp = vmap_.at(cas->get_operand(1));
  Value *cas_val = vmap_.at(cas->get_operand(2));
  Value *old = builder_->CreateAtomicCmpXchg(cas_ptr, cas_cmp, cas_val,
                                             AtomicOrdering::Monotonic,
                                             AtomicOrdering::Monotonic);
  old = builder_->CreateExtractValue(old, std::vector<unsigned>{0});
  Value *atom_ptr;
  atom_ptr = builder_->CreateGEP(sh_mem_ptr_, builder_->getInt32(alloc_->offset(layouts_->get(layouts_->tmp(cas)))));
  atom_ptr = builder_->CreateBitCast(atom_ptr, PointerType::get(old->getType(), 3));

  builder_->CreateStore(old, atom_ptr);
  builder_->CreateBr(tid_0_done_bb);
  builder_->SetInsertPoint(tid_0_done_bb);
  tgt_->add_memfence(module, *builder_);
  tgt_->add_barrier(module, *builder_);
  vmap_[cas] = builder_->CreateLoad(atom_ptr);
}

void generator::visit_atomic_exch_inst(ir::atomic_exch_inst* xchg) {
  BasicBlock *current = builder_->GetInsertBlock();
  Module *module = current->getModule();
  Value *rmw_ptr = vmap_.at(xchg->get_operand(0));
  Value *rmw_val = vmap_.at(xchg->get_operand(1));
  Value *tid = tgt_->get_local_id(module, *builder_, 0);
  Value *pred = builder_->CreateICmpEQ(tid, builder_->getInt32(0));
  BasicBlock *tid_0_bb = BasicBlock::Create(*ctx_, "tid_0", current->getParent());
  BasicBlock *tid_0_done_bb = BasicBlock::Create(*ctx_, "tid_0_done", current->getParent());
  tgt_->add_memfence(module, *builder_);
  tgt_->add_barrier(module, *builder_);
  builder_->CreateCondBr(pred, tid_0_bb, tid_0_done_bb);
  builder_->SetInsertPoint(tid_0_bb);
  builder_->CreateAtomicRMW(AtomicRMWInst::Xchg, rmw_ptr, rmw_val,
                                          AtomicOrdering::Monotonic,
                                          SyncScope::System);
  builder_->CreateBr(tid_0_done_bb);
  builder_->SetInsertPoint(tid_0_done_bb);
  tgt_->add_memfence(module, *builder_);
}

void generator::visit_atomic_add_inst(ir::atomic_add_inst* add) {

    
  if(add->get_type()->is_tile_ty()){
    ir::value* ptr = add->get_operand(0);
    ir::value* val = add->get_operand(1);
    ir::value* msk = add->get_operand(2);
    distributed_tile* ptrs = (distributed_tile*)tmap_.at(ptr);
    distributed_tile* vals = (distributed_tile*)tmap_.at(val);
    distributed_tile* msks = (distributed_tile*)tmap_.at(msk);
    
    for_each(ptr, [&](indices_t idx){
      Value *rmw_ptr = ptrs->get_value(idx);
      Value *rmw_val = vals->get_value(idx);
      Value *rmw_msk = msks->get_value(idx);
      // num bytes
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
      rmw_ptr = builder_->CreateBitCast(rmw_ptr, ty->getPointerTo(1));
      // asm argument type
      std::vector<Type*> arg_ty = {rmw_msk->getType(), rmw_ptr->getType(), rmw_val->getType()};
      // asm function type
      FunctionType *fn_ty = FunctionType::get(ty, arg_ty, false);
      // asm string
      std::string mod = nbits == 32 ? "" : ".noftz";
      std::string asm_str = "@$0 atom.global.sys.add" + mod + ".f" + std::to_string(nbits) + " $1, [$2" + offset + "], $3;";
      std::string ty_id = nbits == 32 ? "f" : "h";
      std::string constraint = "b,=" + ty_id + ",l," + ty_id;
      // create inline asm
      InlineAsm *iasm = InlineAsm::get(fn_ty, asm_str, constraint, true);
      // call asm
      builder_->CreateCall(iasm, {rmw_msk, rmw_ptr, rmw_val});
    });
  }
  else{
    BasicBlock *current = builder_->GetInsertBlock();
    Module *module = current->getModule();
    Value *rmw_ptr = vmap_.at(add->get_operand(0));
    Value *rmw_val = vmap_.at(add->get_operand(1));
    Value *tid = tgt_->get_local_id(module, *builder_, 0);
    Value *pred = builder_->CreateICmpEQ(tid, builder_->getInt32(0));
    BasicBlock *tid_0_bb = BasicBlock::Create(*ctx_, "tid_0", current->getParent());
    BasicBlock *tid_0_done_bb = BasicBlock::Create(*ctx_, "tid_0_done", current->getParent());
    tgt_->add_memfence(module, *builder_);
    tgt_->add_barrier(module, *builder_);
    builder_->CreateCondBr(pred, tid_0_bb, tid_0_done_bb);
    builder_->SetInsertPoint(tid_0_bb);
    builder_->CreateAtomicRMW(AtomicRMWInst::FAdd, rmw_ptr, rmw_val,
                                            AtomicOrdering::Monotonic,
                                            SyncScope::System);
    builder_->CreateBr(tid_0_done_bb);
    builder_->SetInsertPoint(tid_0_done_bb);
    tgt_->add_memfence(module, *builder_);
  }
}

void generator::visit_hmma_dot(ir::dot_inst* dot, shared_tile *TA, shared_tile *TB, distributed_tile *TD, unsigned NK) {
  const auto& shapes = dot->get_type()->get_tile_shapes();
  machine_mma884_layout* hmma = (machine_mma884_layout*)machine_layouts_.at(layouts_->get(dot));
  TA->set_vector_size(4*hmma->pack_size_0_);
  TB->set_vector_size(4*hmma->pack_size_1_);
  TA->set_return_mode(true);
  TB->set_return_mode(true);

  std::map<std::vector<Value*>, std::vector<Value*>> fcs;

  for_each(dot, [&](indices_t idx){
    std::vector<Value*> key(idx.size() - 2);
    std::copy(idx.begin() + 2, idx.end(), key.begin());
    fcs[key].push_back(TD->get_value(idx));
  });

  Type *fp32_ty = builder_->getFloatTy();
  Type *fp16x2_ty = VectorType::get(builder_->getHalfTy(), 2);
  Type *fp32_pack8_ty = StructType::get(*ctx_, std::vector<llvm::Type*>{fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty});
  FunctionType *mma_ty = FunctionType::get(fp32_pack8_ty, std::vector<llvm::Type*>{fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty}, false);


  Value* u_thread_id = tgt_->get_local_id(builder_->GetInsertBlock()->getModule(), *builder_, 0);

  auto ord_a = layouts_->get(dot->get_operand(0))->get_order();
  auto ord_b = layouts_->get(dot->get_operand(1))->get_order();

  bool is_a_trans = is_trans(dot->get_operand(0));
  bool is_b_trans = is_trans(dot->get_operand(1));
  bool is_a_row = is_a_trans ^ (ord_a[0] != 0);
  bool is_b_row = is_b_trans ^ (ord_b[0] != 0);

  Value *offset_a_i = hmma->offset_a_i_;
  Value *offset_a_k = hmma->offset_a_k_;
  if(is_a_row){
    offset_a_i = builder_->CreateAdd(offset_a_i, builder_->CreateURem(u_thread_id, builder_->getInt32(4)));
    offset_a_k = builder_->getInt32(0);
  }

  Value *offset_b_j = hmma->offset_b_j_;
  Value *offset_b_k = hmma->offset_b_k_;
  if(!is_b_row){
    offset_b_j = builder_->CreateAdd(offset_b_j, builder_->CreateURem(u_thread_id, builder_->getInt32(4)));
    offset_b_k = builder_->getInt32(0);
  }

  std::string op_a = is_a_row ? "row" : "col";
  std::string op_b = is_b_row ? "row" : "col";

  InlineAsm *mma_fn = InlineAsm::get(mma_ty, " mma.sync.aligned.m8n8k4." + op_a + "." + op_b + ".f32.f16.f16.f32 "
                                             "{$0, $1, $2, $3, $4, $5, $6, $7}, "
                                             "{$8, $9}, "
                                             "{$10, $11}, "
                                             "{$0, $1, $2, $3, $4, $5, $6, $7};", "=f,=f,=f,=f,=f,=f,=f,=f,r,r,r,r,0,1,2,3,4,5,6,7", false);
  analysis::mma884_layout* layout = layouts_->get(dot)->to_mma884();

  unsigned fpw_0 = layout->fpw(0);
  unsigned fpw_1 = layout->fpw(1);
  unsigned wts_0 = fpw_0 * 8;
  unsigned wts_1 = fpw_1 * 8;
  unsigned wpt_0 = layout->wpt(0);
  unsigned wpt_1 = layout->wpt(1);
  unsigned stride_rep_i = wpt_0 * wts_0;
  unsigned stride_rep_j = wpt_1 * wts_1;
  unsigned num_rep_i = shapes[0] / stride_rep_i;
  unsigned ld_fc = num_rep_i * 2;


  for(auto& x: fcs){
    std::vector<Value *>& fc = x.second;
    for(unsigned pack_i = 0; pack_i < hmma->num_packs_0_; pack_i++)
    for(unsigned pack_j = 0; pack_j < hmma->num_packs_1_; pack_j++){
    for(unsigned K = 0; K < NK; K += 4){
      Value *_K = builder_->getInt32(K);
      Value *current_offset_a_i = builder_->CreateAdd(offset_a_i, builder_->getInt32(pack_i*stride_rep_i*hmma->pack_size_0_));
      Value *current_offset_b_i = builder_->CreateAdd(offset_b_j, builder_->getInt32(pack_j*stride_rep_j*hmma->pack_size_1_));
      indices_t idx_a = {current_offset_a_i, builder_->CreateAdd(offset_a_k, _K)};
      indices_t idx_b = {builder_->CreateAdd(offset_b_k, _K), current_offset_b_i};
      idx_a.insert(idx_a.end(), x.first.begin(), x.first.end());
      idx_b.insert(idx_b.end(), x.first.begin(), x.first.end());
      
      Value *ha = TA->get_value(idx_a);
      Value *hb = TB->get_value(idx_b);
      for(unsigned ii = 0; ii < hmma->pack_size_0_; ii++)
      for(unsigned jj = 0; jj < hmma->pack_size_1_; jj++){
        Value *ha0 = builder_->CreateBitCast(builder_->CreateExtractElement(ha, builder_->getInt32(ii*hmma->pack_size_0_ + 0)), fp16x2_ty);
        Value *ha1 = builder_->CreateBitCast(builder_->CreateExtractElement(ha, builder_->getInt32(ii*hmma->pack_size_0_ + 1)), fp16x2_ty);
        Value *hb0 = builder_->CreateBitCast(builder_->CreateExtractElement(hb, builder_->getInt32(jj*hmma->pack_size_0_ + 0)), fp16x2_ty);
        Value *hb1 = builder_->CreateBitCast(builder_->CreateExtractElement(hb, builder_->getInt32(jj*hmma->pack_size_0_ + 1)), fp16x2_ty);
        std::vector<size_t> idx = {
          (pack_i*2*hmma->pack_size_0_ + ii*2 + 0) + (pack_j*4*hmma->pack_size_1_ + jj*4 + 0)*ld_fc,
          (pack_i*2*hmma->pack_size_0_ + ii*2 + 0) + (pack_j*4*hmma->pack_size_1_ + jj*4 + 1)*ld_fc,
          (pack_i*2*hmma->pack_size_0_ + ii*2 + 1) + (pack_j*4*hmma->pack_size_1_ + jj*4 + 0)*ld_fc,
          (pack_i*2*hmma->pack_size_0_ + ii*2 + 1) + (pack_j*4*hmma->pack_size_1_ + jj*4 + 1)*ld_fc,
          (pack_i*2*hmma->pack_size_0_ + ii*2 + 0) + (pack_j*4*hmma->pack_size_1_ + jj*4 + 2)*ld_fc,
          (pack_i*2*hmma->pack_size_0_ + ii*2 + 0) + (pack_j*4*hmma->pack_size_1_ + jj*4 + 3)*ld_fc,
          (pack_i*2*hmma->pack_size_0_ + ii*2 + 1) + (pack_j*4*hmma->pack_size_1_ + jj*4 + 2)*ld_fc,
          (pack_i*2*hmma->pack_size_0_ + ii*2 + 1) + (pack_j*4*hmma->pack_size_1_ + jj*4 + 3)*ld_fc
        };
        Value *nc = builder_->CreateCall(mma_fn,  std::vector<llvm::Value*>{ha0, ha1, hb0, hb1, fc[idx[0]], fc[idx[1]], fc[idx[2]], fc[idx[3]], fc[idx[4]], fc[idx[5]], fc[idx[6]], fc[idx[7]]});
        fc[idx[0]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{0});
        fc[idx[1]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{1});
        fc[idx[2]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{2});
        fc[idx[3]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{3});
        fc[idx[4]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{4});
        fc[idx[5]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{5});
        fc[idx[6]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{6});
        fc[idx[7]] = builder_->CreateExtractValue(nc, std::vector<unsigned>{7});
      }
    }
    }
  }

  // write back
  unsigned i = 0;
  for_each(dot, [&](indices_t idx){
    std::vector<Value*> key(idx.size() - 2);
    std::copy(idx.begin() + 2, idx.end(), key.begin());
    if(i >= fcs.at(key).size())
      i = 0;
    set_value(dot, idx, fcs.at(key)[i++]);
  });

  TA->set_return_mode(false);
  TB->set_return_mode(false);

}
void generator::visit_scanline_dot(ir::dot_inst* dot, shared_tile *TA, shared_tile *TB, distributed_tile *TD, unsigned NK,
                                   Type *c_ty, Function *f_mul_add) {
  TA->set_vector_size(TD->axis(0).contiguous);
  TB->set_vector_size(TD->axis(1).contiguous);
  for_each(dot, [&](indices_t idx){
    Value *res = TD->get_value(idx);
    for(unsigned K = 0; K < NK; ++K){
      // input indices
      indices_t a_idx = {idx[0], builder_->getInt32(K)};
      indices_t b_idx = {builder_->getInt32(K), idx[1]};
      // add batching dimension
      for(size_t i = 2; i < idx.size(); i++){
        a_idx.insert(a_idx.end(), idx[i]);
        b_idx.insert(b_idx.end(), idx[i]);
      }
      // load value
      Value *a = TA->get_value(a_idx);
      Value *b = TB->get_value(b_idx);
      if(a->getType() != c_ty)
        a = builder_->CreateFPCast(a, c_ty);
      if(b->getType() != c_ty)
        b = builder_->CreateFPCast(b, c_ty);
      res = builder_->CreateCall(f_mul_add, std::vector<llvm::Value*>{a, b, res});
    }
    set_value(dot, idx, res);
  });
}

void generator::visit_outer_dot(ir::dot_inst* dot, distributed_tile *TA, distributed_tile *TB, distributed_tile *TD, unsigned NK,
                                Type *c_ty, Function *f_mul_add) {
  for_each(dot, [&](indices_t idx){
    Value *res = TD->get_value(idx);
    indices_t a_idx = {idx[0], builder_->getInt32(0)};
    indices_t b_idx = {builder_->getInt32(0), idx[1]};
    std::swap(a_idx[0], a_idx[1]);
    std::swap(b_idx[0], b_idx[1]);
    Value *a = TA->get_value(a_idx);
    Value *b = TB->get_value(b_idx);
    if(a->getType() != c_ty)
      a = builder_->CreateFPCast(a, c_ty);
    if(b->getType() != c_ty)
      b = builder_->CreateFPCast(b, c_ty);
    res = builder_->CreateCall(f_mul_add, std::vector<llvm::Value*>{a, b, res});
    set_value(dot, idx, res);
  });
}

void generator::visit_dot_inst(ir::dot_inst* dot) {
  Function *fn = builder_->GetInsertBlock()->getParent();

  Module *module = fn->getParent();
  ir::value *A = dot->get_operand(0);
  ir::value *B = dot->get_operand(1);
  ir::value *D = dot->get_operand(2);

  distributed_tile *TD = (distributed_tile*)tmap_.at(D);
  Type *c_ty = llvm_type(D->get_type()->get_scalar_ty(), *ctx_);
  Function *f_mul_add = Intrinsic::getDeclaration(module, Intrinsic::fmuladd, std::vector<llvm::Type*>{c_ty});
  auto A_shapes = A->get_type()->get_tile_shapes();
  size_t red_axis = 1;
  unsigned NK = A_shapes[red_axis];

  if(NK != 1) {
    shared_tile *TA = (shared_tile*)tmap_.at(A);
    shared_tile *TB = (shared_tile*)tmap_.at(B);
    if(layouts_->get(dot)->to_mma884())
      visit_hmma_dot(dot, TA, TB, TD, NK);
    else
      visit_scanline_dot(dot, TA, TB, TD, NK, c_ty, f_mul_add);
  }
  else {
    distributed_tile *TA = (distributed_tile*)tmap_.at(A);
    distributed_tile *TB = (distributed_tile*)tmap_.at(B);
    visit_outer_dot(dot, TA, TB, TD, NK, c_ty, f_mul_add);
  }
}

void generator::visit_trans_inst(ir::trans_inst* trans) {
  shared_tile* in = (shared_tile*)tmap_.at(trans->get_operand(0));
  shared_tile* out = new shared_tile(in->get_ty(), in->get_shapes(), in->get_order(), in->get_pointer(), *builder_, in->get_offset(), trans->get_perm());
  tmap_[trans] = out;
}

void generator::visit_sqrt_inst(ir::sqrt_inst* sqt) {
  for_each(sqt, [&](indices_t idx){
    Value *val = get_value(sqt->get_operand(0), idx);
    Module* module = builder_->GetInsertBlock()->getModule();
    Value *sqrt = Intrinsic::getDeclaration(module, Intrinsic::sqrt, std::vector<llvm::Type*>{val->getType()});
    Value *ret = builder_->CreateCall(sqrt, std::vector<llvm::Value*>{val});
    set_value(sqt, idx, ret);
  });
}

void generator::visit_reduce_inst(ir::reduce_inst* x) {
  std::map<indices_t, Value*> partial;
  ir::value *arg = x->get_operand(0);
  distributed_tile* arg_tile = (distributed_tile*)tmap_.at(arg);
  ir::reduce_inst::op_t op = x->get_op();
  unsigned axis = x->get_axis();

  Type *fp32_ty = builder_->getFloatTy();
  FunctionType *fmaxmin_ty = FunctionType::get(fp32_ty, std::vector<llvm::Type*>{fp32_ty, fp32_ty}, false);
  InlineAsm *fmin = InlineAsm::get(fmaxmin_ty, "min.ftz.f32 $0, $1, $2;", "=f,f,f", false);
  InlineAsm *fmax = InlineAsm::get(fmaxmin_ty, "max.ftz.f32 $0, $1, $2;", "=f,f,f", false);

  auto accumulate = [&](Value* x, Value *y) -> Value* {
    switch(op) {
      case ir::reduce_inst::ADD: return builder_->CreateAdd(x, y);
      case ir::reduce_inst::SUB: return builder_->CreateSub(x, y);
      case ir::reduce_inst::MAX:{
        if(x->getType()->isIntegerTy())
          return builder_->CreateSelect(builder_->CreateICmpSGE(x, y), x, y);
        else
          return builder_->CreateMaxNum(x, y);
      }
      case ir::reduce_inst::MIN:{
        if(x->getType()->isIntegerTy())
          return builder_->CreateSelect(builder_->CreateICmpSLE(x, y), x, y);
        else
          return builder_->CreateMinNum(x, y);
      }
      case ir::reduce_inst::FADD: return builder_->CreateFAdd(x, y);
      case ir::reduce_inst::FSUB: return builder_->CreateFSub(x, y);
    case ir::reduce_inst::FMAX: return builder_->CreateCall(fmax, std::vector<llvm::Value*>{x, y});
      case ir::reduce_inst::FMIN: return builder_->CreateCall(fmin, std::vector<llvm::Value*>{x, y});
      default: assert(false); return nullptr;
    }
  };

  Value *neutral;
  switch(op) {
    case ir::reduce_inst::ADD: neutral = builder_->getInt32(0); break;
    case ir::reduce_inst::SUB: neutral = builder_->getInt32(0); break;
    case ir::reduce_inst::MAX: neutral = builder_->getInt32(INT32_MIN); break;
    case ir::reduce_inst::MIN: neutral = builder_->getInt32(INT32_MAX); break;
    case ir::reduce_inst::FADD: neutral = ConstantFP::get(arg_tile->get_ty(), 0); break;
    case ir::reduce_inst::FSUB: neutral = ConstantFP::get(arg_tile->get_ty(), 0); break;
    case ir::reduce_inst::FMAX: neutral = ConstantFP::get(arg_tile->get_ty(), -INFINITY); break;
    case ir::reduce_inst::FMIN: neutral = ConstantFP::get(arg_tile->get_ty(), INFINITY); break;
    default: assert(false); break;
  }



  analysis::data_layout* arg_layout = layouts_->get(arg);
  if(auto* L = dynamic_cast<analysis::scanline_layout*>(arg_layout)){
    bool can_optimize = L->get_rank() == 1;
    /*
    for(size_t r = 0; r < L->get_rank(); r++){
      if(r != axis)
        can_optimize = can_optimize && (L->mts(r) == L->get_shape()[r]);
    }
    */
    if(can_optimize){
      Value *thread_acc = nullptr;
      // reduce within thread
      arg_tile->for_each([&](indices_t idx) {
        Value *current = arg_tile->get_value(idx);
        if(thread_acc == nullptr)
          thread_acc = current;
        else
          thread_acc = accumulate(thread_acc, current);
      });
      // reduce within wrap
      FunctionType *fn_ty = FunctionType::get(thread_acc->getType(), std::vector<llvm::Type*>{thread_acc->getType(), builder_->getInt32Ty()}, false);
      InlineAsm *shfl_xor = InlineAsm::get(fn_ty, "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;", "=f,f,r", false);
      Value *warp_acc = thread_acc;
      for(int i = 16; i > 0; i >>= 1)
        warp_acc = accumulate(warp_acc, builder_->CreateCall(shfl_xor, std::vector<llvm::Value*>{warp_acc, builder_->getInt32(i)}));
      // shared memory pointer
      unsigned addr_space = sh_mem_ptr_->getType()->getPointerAddressSpace();
      Type *res_ty = arg_tile->get_ty();
      Value *sh_mem_ptr = builder_->CreateBitCast(sh_mem_ptr_, PointerType::get(res_ty, addr_space));
      Value* u_thread_id = tgt_->get_local_id(builder_->GetInsertBlock()->getModule(), *builder_, 0);
      Value* warp_id = builder_->CreateUDiv(u_thread_id, builder_->getInt32(32));
      Value *write_ptr = builder_->CreateGEP(sh_mem_ptr, warp_id);
      // store warp result in shared memory
      tgt_->add_barrier(mod_, *builder_);
      builder_->CreateStore(warp_acc, write_ptr);
      tgt_->add_barrier(mod_, *builder_);
      // accumulate all warps
      Value *load_ptr = builder_->CreateGEP(sh_mem_ptr, u_thread_id);
      Value* is_first_warp = builder_->CreateICmpEQ(warp_id, builder_->getInt32(0));
      BasicBlock* bb_final_acc = BasicBlock::Create(*ctx_, "bb_final_acc", builder_->GetInsertBlock()->getParent());
      BasicBlock* bb_final_acc_done = BasicBlock::Create(*ctx_, "bb_final_acc_done", builder_->GetInsertBlock()->getParent());
      builder_->CreateCondBr(is_first_warp, bb_final_acc, bb_final_acc_done);
      builder_->SetInsertPoint(bb_final_acc);
      Value* final_val = builder_->CreateLoad(load_ptr);
      for(int i = (num_warps_+1)/2; i > 0; i >>= 1)
        final_val = accumulate(final_val, builder_->CreateCall(shfl_xor, std::vector<llvm::Value*>{final_val, builder_->getInt32(i)}));
      builder_->CreateStore(final_val, load_ptr);
      builder_->CreateBr(bb_final_acc_done);
//      // store first warp done
      builder_->SetInsertPoint(bb_final_acc_done);
      // write back
      tgt_->add_barrier(mod_, *builder_);
      final_val = builder_->CreateLoad(sh_mem_ptr);
      for_each(x, [&](indices_t idx) {
        set_value(x, idx, final_val);
      });
      return;
    }
  }

  // reduce within thread
  arg_tile->for_each([&](indices_t idx) {
    indices_t pidx = idx;
    pidx[axis] = builder_->getInt32(0);
    Value *current = arg_tile->get_value(idx);
    // current partial result is not initialized -- create
    if(partial.find(pidx) == partial.end())
      partial[pidx] = current;
    // current partial result is initialized -- accumulate
    else
      partial[pidx] = accumulate(partial[pidx], current);
  });

  // reduce within blocks
  machine_data_layout *slayout = machine_layouts_.at(layouts_->get(layouts_->tmp(x)));
  shared_tile *stile = (shared_tile*)slayout->create(x);
  unsigned depth = stile->get_shapes()[axis];

  unsigned addr_space = sh_mem_ptr_->getType()->getPointerAddressSpace();
  Type *res_ty = arg_tile->get_ty();
  Value *base_ptr = builder_->CreateBitCast(sh_mem_ptr_, PointerType::get(res_ty, addr_space));
  for(auto& x: partial) {
    // current element being computed
    Value *lane = axes_.at(a_axes_->get(arg, axis)).thread_id;
    Value *&result = x.second;
    indices_t write_idx = x.first;
    write_idx[axis] = lane;
    // shared memory write  pointer
    Value *write_offset = shared_tile::shared_offset(*builder_, stile->get_shapes(), stile->get_perm(), stile->get_order(), write_idx);
    Value *write_ptr = builder_->CreateGEP(base_ptr, write_offset);
    // initialize shared memory
    tgt_->add_barrier(mod_, *builder_);
    builder_->CreateStore(result, write_ptr);
    // build result
    for(unsigned i = depth/2; i > 0; i >>= 1){
      // current indices
      indices_t current(write_idx.size(), builder_->getInt32(0));
      current[axis] = builder_->getInt32(i);
      // shared memory offset
      Value *read_offset = shared_tile::shared_offset(*builder_, stile->get_shapes(), stile->get_perm(), stile->get_order(), current);
      Value *is_active = builder_->CreateICmpULT(lane, builder_->getInt32(i));
      read_offset = builder_->CreateSelect(is_active, read_offset, builder_->getInt32(0));
      // shared memory read pointer
      Value *read_ptr = builder_->CreateGEP(write_ptr, read_offset);
      tgt_->add_barrier(mod_, *builder_);
      Value *next = builder_->CreateLoad(read_ptr);
      // accumulate
      result = accumulate(result, next);
      // write back
      tgt_->add_barrier(mod_, *builder_);
      builder_->CreateStore(result, write_ptr);
    }
  }
  tgt_->add_barrier(mod_, *builder_);

  // write back
  for_each(x, [&](indices_t idx) {
    indices_t red_idx = idx;
    red_idx.insert(red_idx.begin() + axis, builder_->getInt32(0));
    Value *read_offset = shared_tile::shared_offset(*builder_, stile->get_shapes(), stile->get_perm(), stile->get_order(),  red_idx);
    Value *read_ptr = builder_->CreateGEP(base_ptr, read_offset);
    set_value(x, idx, builder_->CreateLoad(read_ptr));
  });
}

void generator::visit_select_inst(ir::select_inst* select) {
  for_each(select, [&](indices_t idx){
    Value *pred = get_value(select->get_operand(0), idx);
    Value *if_value = get_value(select->get_operand(1), idx);
    Value *else_value = get_value(select->get_operand(2), idx);
    Value *ret = builder_->CreateSelect(pred, if_value, else_value);
    set_value(select, idx, ret);
  });

}

void generator::visit_recoalesce_inst(ir::recoalesce_inst* rc) {
  ir::value *op = rc->get_operand(0);
  ir::tile_type::tile_shapes_t shape = rc->get_type()->get_tile_shapes();
  size_t rank = shape.size();
  // temporary layout
  shared_tile *tmp = (shared_tile*)machine_layouts_.at(layouts_->get(layouts_->tmp(rc)))
                                   ->create(rc);
  // pointer to temporary shared memory
  Type *ty = llvm_type(rc->get_type()->get_scalar_ty(), *ctx_);
  // layouts
  analysis::mma884_layout* in_layout = layouts_->get(op)->to_mma884();
  analysis::scanline_layout* out_layout = layouts_->get(rc)->to_scanline();
  // machine tiles
  distributed_tile *in_dt = (distributed_tile*)(tmap_.at(op));
  distributed_tile *out_dt = (distributed_tile*)(tmap_.at(rc));
  // WMMA configuration
  long wmma_pt[3] = { 2, 4, 1};
  long wmma[3] = { 8*in_layout->wpt(0)*in_layout->fpw(0),
                   8*in_layout->wpt(1)*in_layout->fpw(1),
                   1};
  // Work per thread for input  layout
  long in_pt[3]  = { shape[0] / wmma[0],
                     shape[1] / wmma[1],
                     1 };
  // Work per thread for output layout
  long out_pt[3] = { shape[0] / out_layout->mts(0),
                     shape[1] / out_layout->mts(1),
                     1 };
  if(rank > 2){
    wmma[2] = in_layout->wpt(2)*in_layout->fpw(2);
    in_pt[2] = shape[2] / wmma[2];
    out_pt[2] = shape[2] / out_layout->mts(2);
  }
  // Orders
  auto ord = out_layout->get_order();
  if(ord.size() < 3)
    ord.push_back(2);
  // pointer lanes
  std::vector<std::vector<Value*>> ptrs;
  for(int in_zz = 0; in_zz < wmma_pt[ord[2]]; in_zz++) {
    std::vector<Value*> current;
    for(int in_cc = 0; in_cc < wmma_pt[ord[1]]; in_cc++) {
      Value *base;
      base = builder_->CreateGEP(sh_mem_ptr_, builder_->getInt32(alloc_->offset(layouts_->get(layouts_->tmp(rc)))));
      base = builder_->CreateBitCast(base, PointerType::get(ty, 3));

      // shared memory stride
      Value *stride_0 = builder_->getInt32(tmp->get_shapes()[ord[0]]);
      // indices
      Value *idx_cc = axes_.at(a_axes_->get(op, ord[1])).values[in_cc];
      // offset
      Value *off = builder_->CreateMul(stride_0, idx_cc);
      if(rank > 2){
        Value *stride_1 = builder_->CreateMul(stride_0,
                                              builder_->getInt32(tmp->get_shapes()[ord[1]]));
        Value *idx_zz = axes_.at(a_axes_->get(op, ord[2])).values[in_zz];
        off = builder_->CreateAdd(off, builder_->CreateMul(stride_1, idx_zz));
      }
      current.push_back(builder_->CreateGEP(base, off));
    }
    ptrs.push_back(current);
  }
  // Re-coalesce loops
  for(int in_z = 0; in_z < in_pt[ord[2]]; in_z++)
  for(int in_c = 0; in_c < in_pt[ord[1]]; in_c++){
    // write to shared
    tgt_->add_barrier(mod_, *builder_);
    for(int in_zz = 0; in_zz < wmma_pt[ord[2]]; in_zz++)
    for(int in_cc = 0; in_cc < wmma_pt[ord[1]]; in_cc++){
      std::vector<int> starts(rank), len(rank);
      starts[ord[0]] = 0;
      starts[ord[1]] = in_c*wmma_pt[ord[1]] + in_cc;
      len[ord[0]] = wmma_pt[ord[0]]*in_pt[ord[0]];
      len[ord[1]] = 1;
      if(rank > 2){
        starts[ord[2]] = in_z*wmma_pt[ord[2]] + in_zz;
        len[ord[2]] = 1;
      }
      in_dt->for_each([&](indices_t idx){
        Value *write_ptr = builder_->CreateGEP(ptrs[in_zz][in_cc], idx[ord[0]]);
        builder_->CreateStore(in_dt->get_value(idx), write_ptr);
      }, starts, len);
    }
    tgt_->add_barrier(mod_, *builder_);
    // load from shared
    for(int out_zz = 0; out_zz < out_pt[ord[2]] / in_pt[ord[2]]; out_zz++)
    for(int out_cc = 0; out_cc < out_pt[ord[1]] / in_pt[ord[1]]; out_cc++){
      std::vector<int> starts(rank), len(rank);
      starts[ord[0]] = 0;
      starts[ord[1]] = in_c*(out_pt[ord[1]] / in_pt[ord[1]]) + out_cc;
      len[ord[0]] = out_pt[ord[0]];
      len[ord[1]] = 1;
      if(rank > 2){
        starts[ord[2]] = in_z*(out_pt[ord[2]] / in_pt[ord[2]]) + out_zz;
        len[ord[2]] = 1;
      }
      out_dt->for_each([&](indices_t idx){
        indices_t read_idx(rank);
        read_idx[ord[0]] = idx[ord[0]];
        read_idx[ord[1]] = axes_.at(a_axes_->get(rc, ord[1])).values[out_cc];
        if(rank > 2)
          read_idx[ord[2]] = axes_.at(a_axes_->get(rc, ord[2])).values[out_zz];
        out_dt->set_value(idx, tmp->get_value(read_idx));
      }, starts, len);
    }
  }
  tgt_->add_barrier(mod_, *builder_);
}

void generator::visit_copy_to_shared_inst(ir::copy_to_shared_inst* cts) {
  unsigned vector_size = 1;
  ir::value *arg = cts->get_operand(0);
  analysis::shared_layout* out_layout = layouts_->get(cts)->to_shared();
  analysis::scanline_layout* in_layout = layouts_->get(arg)->to_scanline();
  auto out_order = out_layout->get_order();
  auto in_order = in_layout->get_order();
  // tiles
  if(out_order == in_order)
    vector_size = in_layout->nts(in_order[0]);

  std::map<unsigned, Value*> packets;
  for_each(arg, [&](indices_t idx){
    distributed_tile* in = (distributed_tile*)tmap_.at(arg);
    unsigned linear = in->get_linear_index(idx);
    unsigned id = linear / vector_size;
    Value *in_value = in->get_value(idx);
    if(linear % vector_size == 0)
      packets[id] = UndefValue::get(VectorType::get(in_value->getType(), vector_size));
    packets[id] = builder_->CreateInsertElement(packets.at(id), in_value, linear % vector_size);
  });

  for_each(arg, [&](indices_t idx){
    distributed_tile* in = (distributed_tile*)tmap_.at(arg);
    shared_tile* result = (shared_tile*)tmap_.at(cts);
    unsigned linear = in->get_linear_index(idx);
    unsigned id = linear / vector_size;
    if(linear % vector_size == 0)
      result->set_value(idx, packets[id]);
  });
}
void generator::visit_copy_from_shared_inst(ir::copy_from_shared_inst* cfs) {
  for_each(cfs, [&](indices_t idx){
    set_value(cfs, idx, get_value(cfs->get_operand(0), idx));
  });
}

void generator::visit_barrier_inst(ir::barrier_inst*) {
  Module *module = builder_->GetInsertBlock()->getModule();
  tgt_->add_barrier(module, *builder_);
}

void generator::visit_make_range_dyn(ir::make_range_dyn* x) {
  for_each(x, [&](indices_t idx){
    assert(idx.size() == 1);
    if(idx[0] == builder_->getInt32(0))
      set_value(x, idx, idx[0]);
    else{
      BinaryOperator *bin_add = dyn_cast<BinaryOperator>(idx[0]);
      assert(bin_add);
      Value *res = bin_add->getOperand(0);
      set_value(x, idx, res);
    }
  });
}

void generator::visit_make_range_sta(ir::make_range_sta* x) {
  for_each(x, [&](indices_t idx){
    assert(idx.size() == 1);
    if(idx[0] == builder_->getInt32(0)){
      set_value(x, idx, idx[0]);
    }
    else{
      BinaryOperator *bin_add = dyn_cast<BinaryOperator>(idx[0]);
      assert(bin_add);
      Value *res = bin_add->getOperand(1);
      assert(isa<Constant>(res));
      set_value(x, idx, res);
    }
  });
}

void generator::visit_make_range(ir::make_range* x) {
  for_each(x, [&](indices_t idx){
    assert(idx.size() == 1);
    set_value(x, idx, idx[0]);
  });
}



void generator::visit_undef_value(ir::undef_value *ud) {
  vmap_[ud] = llvm::UndefValue::get(llvm_type(ud->get_type(), *ctx_));
}

void generator::visit_constant_int(ir::constant_int *cst){
  Type *ty = llvm_type(cst->get_type()->get_scalar_ty(), *ctx_);
  vmap_[cst] = ConstantInt::get(ty, cst->get_value());
}

void generator::visit_constant_fp(ir::constant_fp *cst){
  Type *ty = llvm_type(cst->get_type()->get_scalar_ty(), *ctx_);
  vmap_[cst] = ConstantFP::get(ty, cst->get_value());
}

void generator::visit_alloc_const(ir::alloc_const *alloc) {
  unsigned size = ((ir::constant_int*)alloc->get_operand(0))->get_value();
  Type *element_ty = llvm_type(alloc->get_type()->get_pointer_element_ty(), *ctx_);
  Type *array_ty = llvm::ArrayType::get(element_ty, size);
  Value *array = new llvm::GlobalVariable(*mod_, array_ty, false, llvm::GlobalVariable::ExternalLinkage,
                                            nullptr, alloc->get_name(), nullptr, llvm::GlobalVariable::NotThreadLocal, 4);
  vmap_[alloc] = builder_->CreateBitCast(array, element_ty->getPointerTo(4));
}


void generator::visit_function(ir::function* fn) {
  LLVMContext &ctx = builder_->getContext();
  FunctionType *fn_ty = (FunctionType*)llvm_type(fn->get_fn_type(), *ctx_);
  if(!tgt_->is_gpu()){
    Type *fn_ret_ty = fn_ty->getReturnType();
    std::vector<Type*> fn_args_ty;
    for(unsigned i = 0; i < fn_ty->getNumParams(); i++)
      fn_args_ty.push_back(fn_ty->getParamType(i));
    fn_args_ty.push_back(builder_->getInt32Ty());
    fn_args_ty.push_back(builder_->getInt32Ty());
    fn_args_ty.push_back(builder_->getInt32Ty());
    fn_ty = FunctionType::get(fn_ret_ty, fn_args_ty, false);
  }
  Function *ret = Function::Create(fn_ty, Function::ExternalLinkage, fn->get_name(), mod_);
  // set attributes
  for(auto attr_pair: fn->attrs()){
    unsigned id = attr_pair.first;
    for(ir::attribute attr: attr_pair.second)
    if(attr.is_llvm_attr()){
      llvm::Attribute llattr = llvm_attr(ctx, attr);
      if(llattr.getKindAsEnum() != llvm::Attribute::None)
        ret->addAttribute(id, llvm_attr(ctx, attr));
    }
  }
  // set metadata
  if(tgt_->is_gpu()){
      tgt_->set_kernel(*builder_, ctx, mod_, ret);
      Metadata *md_args[] = {
        ValueAsMetadata::get(ret),
        MDString::get(ctx, "maxntidx"),
        ValueAsMetadata::get(builder_->getInt32(num_warps_*32))
      };
      mod_->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(MDNode::get(ctx, md_args));
  }
  // set arguments
  for(unsigned i = 0; i < fn->args().size(); i++)
    vmap_[fn->args()[i]] = &*(ret->arg_begin() + i);
  // create blocks
  for(ir::basic_block *block: fn->blocks()) {
    BasicBlock *dst_block = BasicBlock::Create(ctx, block->get_name(), ret);
    vmap_[block] = dst_block;
  }
  builder_->SetInsertPoint((BasicBlock*)vmap_[fn->blocks()[0]]);
  // initialize layouts
  for(auto x: layouts_->get_all()){
    visit_layout(x.second);
  }
  // generate LLVM-IR code
  for(ir::basic_block *block: fn->blocks())
    visit_basic_block(block);
  // finalize
  finalize_function(fn);
}



void generator::visit_layout_hmma_884(analysis::mma884_layout* layout) {
  machine_layouts_[layout] = new machine_mma884_layout(mod_, &*builder_, tgt_, a_axes_, axes_, layout);
}

void generator::visit_layout_scanline(analysis::scanline_layout* layout) {
  machine_layouts_[layout] = new machine_scanline_layout(mod_, &*builder_, tgt_, a_axes_, axes_, layout);
}

void generator::visit_layout_shared(analysis::shared_layout* layout) {

  machine_layouts_[layout] = new machine_shared_layout(mod_, &*builder_, tgt_, alloc_, sh_mem_ptr_, layout, vmap_, tmap_);
}

void generator::visit_basic_block(ir::basic_block * block) {
  BasicBlock *parent = (BasicBlock*)vmap_[block];
  builder_->SetInsertPoint(parent);
  for(ir::instruction *i: block->get_inst_list()){
    // std::cout << typeid(*i).name() << std::endl;
    visit_value(i);
  }
  vmap_[block] = builder_->GetInsertBlock();
}

void generator::visit_argument(ir::argument* arg) {

}

void generator::for_each(ir::value *x, const std::function<void(indices_t)>& fn) {
  if(!x->get_type()->is_tile_ty())
    return fn({});
  else {
//    if(tmap_.find(x) == tmap_.end())
//      tmap_[x] = machine_layouts_.at(layouts_->get(x))->create(x);
    if(auto *dt = dynamic_cast<distributed_tile*>(tmap_.at(x)))
      dt->for_each(fn);
  }
}

Value* generator::get_value(ir::value *x, const indices_t& idx) {
  if(x->get_type()->is_tile_ty())
    return tmap_.at(x)->get_value(idx);
  return vmap_.at(x);
}

void generator::set_value(ir::value *x, const indices_t& idx, Value* v) {
  if(x->get_type()->is_tile_ty())
    tmap_.at(x)->set_value(idx, v);
  else
    vmap_[x] = v;
}


void generator::finalize_shared_layout(analysis::shared_layout *shared) {
  if(shared->get_double_buffer()) {
    auto info = *shared->get_double_buffer();
    ir::phi_node *phi = info.phi;
    PHINode *ptr = (PHINode*)((shared_tile*)tmap_.at(phi))->get_pointer();
    PHINode *offset = (PHINode*)((shared_tile*)tmap_.at(phi))->get_offset();
    for(unsigned n = 0; n < phi->get_num_incoming(); n++){
      ir::basic_block* inc_block = phi->get_incoming_block(n);
      ir::value* inc_val = phi->get_incoming_value(n);
      BasicBlock *llvm_inc_block = (BasicBlock*)vmap_.at(inc_block);
      shared_tile *inc_shared = (shared_tile*)tmap_.at(inc_val);
      if(inc_val == info.latch){
        builder_->SetInsertPoint(llvm_inc_block->getTerminator());
        Value *next_offset = builder_->CreateNeg(offset);
        offset->addIncoming(next_offset, llvm_inc_block);
      }
      else {
        unsigned num_bytes = shared->get_type()->get_primitive_size_in_bits() / 8;
        offset->addIncoming(builder_->getInt32(shared->get_size() / (2*num_bytes)), llvm_inc_block);
      }
      ptr->addIncoming(inc_shared->get_pointer(), llvm_inc_block);
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
}

void generator::finalize_phi_node(ir::phi_node *phi) {
  auto it = tmap_.find(phi);
  if(it != tmap_.end() && dynamic_cast<shared_tile*>(it->second))
    return;
  for(unsigned n = 0; n < phi->get_num_incoming(); n++){
    ir::basic_block *inc_block = phi->get_incoming_block(n);
    BasicBlock *llvm_inc_block = (BasicBlock*)vmap_.at(inc_block);
    for_each(phi, [&](indices_t idx){
      PHINode *llvm_phi = (PHINode*)get_value(phi, idx);
      Value *llvm_inc_val = get_value(phi->get_incoming_value(n), idx);
      llvm_phi->addIncoming(llvm_inc_val, llvm_inc_block);
    });
  }
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
    ArrayType *array_ty = ArrayType::get(int_32_ty, alloc_size/4);
    Type *ptr_ty = PointerType::get(int_8_ty, 3);
    GlobalVariable *sh_mem_array =
      new GlobalVariable(*mod_, array_ty, false, GlobalVariable::ExternalLinkage,
                         nullptr, "__shared_ptr", nullptr, GlobalVariable::NotThreadLocal, 3);
    sh_mem_ptr_ = builder_->CreateBitCast(sh_mem_array, ptr_ty);
  }
  // visit functions
  for(ir::function *fn: src.get_function_list())
    visit_function(fn);
}


}
}
