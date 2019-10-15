#include <numeric>
#include "triton/codegen/selection.h"
#include "triton/codegen/target.h"
#include "triton/codegen/analysis/liveness.h"
#include "triton/codegen/analysis/layout.h"
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/transform/coalesce.h"
#include "triton/codegen/instructions.h"
#include "triton/ir/context.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/type.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/InlineAsm.h"

namespace triton{
namespace codegen{

using namespace llvm;

/* Distributed Tile */
void distributed_tile::init_indices() {
  std::vector<size_t> id(axes_.size(), 0);
  // create iteration order
  std::vector<size_t> order(id.size());
  std::iota(order.begin(), order.end(), 0);
  auto cmp = [&](int x, int y) {
    return axes_[x].contiguous > axes_[y].contiguous;
  };
  std::sort(order.begin(), order.end(), cmp);
  // build
  size_t k = 0;
  while(true) {
    indices_t current;
    for(size_t d = 0; d < id.size(); d++)
      current.push_back(axes_[d].values[id[d]]);
    size_t sz = indices_.size();
    indices_[current] = sz;
    values_[current] = nullptr;
    ordered_indices_.push_back(current);
    id[order[0]]++;
    while(id[order[k]] == axes_[order[k]].values.size()){
      if(k == id.size() - 1)
        return;
      id[order[k++]] = 0;
      id[order[k]]++;
    }
    k = 0;
  }
}

llvm::Type *distributed_tile::make_vector_ty(llvm::Type *ty, size_t vector_size) {
  if(vector_size == 1)
    return ty;
  return VectorType::get(ty, vector_size);
}

distributed_tile::distributed_tile(Type *ty, const shapes_t &shapes, const std::vector<int>& order, const axes_t &axes, llvm::IRBuilder<> &builder, bool vectorize)
    : tile(make_vector_ty(ty, vectorize?axes[0].contiguous:1), shapes), axes_(axes), order_(order), builder_(builder) {
  vector_size_ = vectorize?ty_->getVectorNumElements():1;
  init_indices();
}

void distributed_tile::set_value(indices_t idx, Value *x) {
  assert(x->getType() == ty_ && "cannot set a value of different type");
  Value *&result = values_[idx];
  assert(!result && "value cannot be set twice");
  result = x;
}

Value* distributed_tile::get_value(indices_t idx) {
  Value *result = values_.at(idx);
  assert(result && "value has not been set");
  return result;
}

unsigned distributed_tile::get_linear_index(indices_t idx) {
  return indices_[idx];
}

indices_t distributed_tile::get_ordered_indices(unsigned id) {
  return ordered_indices_.at(id);
}


void distributed_tile::for_each(std::function<void (indices_t)> fn) {
  for(unsigned i = 0; i < ordered_indices_.size(); i++){
    if(i % vector_size_ == 0)
      fn(ordered_indices_[i]);
  }
}

/* Shared Tile */
void shared_tile::extract_constant(Value *arg, Value *&non_cst, Value *&cst) {
  BinaryOperator *bin_op = dyn_cast<BinaryOperator>(arg);
  Constant *_0 = ConstantInt::get(Type::getInt32Ty(arg->getContext()), 0);
  if(dyn_cast<Constant>(arg)){
    cst = arg;
    non_cst = _0;
    return;
  }
  if(!bin_op || bin_op->getOpcode() != llvm::BinaryOperator::Add){
    non_cst = arg;
    cst = _0;
    return;
  }
  Constant *cst_lhs = dyn_cast<Constant>(bin_op->getOperand(0));
  Constant *cst_rhs = dyn_cast<Constant>(bin_op->getOperand(1));
  if(cst_lhs && cst_rhs){
    cst = arg;
    non_cst = _0;
  }
  else if(cst_lhs){
    cst = cst_lhs;
    non_cst = bin_op->getOperand(1);
  }
  else if(cst_rhs){
    cst = cst_rhs;
    non_cst = bin_op->getOperand(0);
  }
  else{
    non_cst = arg;
    cst = _0;
  }
}

void shared_tile::extract_constant(const indices_t &arg_idx, indices_t &non_cst_idx, indices_t &cst_idx) {
  non_cst_idx.clear();
  cst_idx.clear();
  for(Value *idx: arg_idx){
    Value *non_cst, *cst;
    extract_constant(idx, non_cst, cst);
    non_cst_idx.push_back(non_cst);
    cst_idx.push_back(cst);
  }
}


Value* shared_tile::shared_offset(llvm::IRBuilder<> &builder, const shapes_t& shapes, const std::vector<int>& perm, const std::vector<int>& order, indices_t idx) {
  // strides
  std::vector<Value*> strides(order.size());
  strides[order[0]] = builder.getInt32(1);
  for(size_t i = 1; i < idx.size(); i++)
    strides[order[i]] = builder.CreateMul(strides[order[i-1]], builder.getInt32(shapes[order[i-1]]));
  // result
  Value *result = builder.getInt32(0);
  for(size_t i = 0; i < strides.size(); i++)
    result = builder.CreateAdd(result, builder.CreateMul(idx[perm[i]], strides[i]));
  return result;
}

shared_tile::shared_tile(Type *ty, const shapes_t &shapes, const std::vector<int>& order, Value *ptr, llvm::IRBuilder<> &builder, Value *offset, const std::vector<int>& perm):
  tile(ty, shapes), order_(order), ptr_(ptr), builder_(builder), offset_(offset), vector_size_(1), perm_(perm){
  return_vector_ = false;
  if(perm_.empty()){
    perm_.resize(shapes.size());
    std::iota(perm_.begin(), perm_.end(), 0);
  }
}

void shared_tile::set_value(indices_t idx, Value *value) {
  Value *ptr = builder_.CreateGEP(ptr_, shared_offset(builder_, shapes_, perm_, order_, idx));
  unsigned addr_space = ptr->getType()->getPointerAddressSpace();
  ptr = builder_.CreateBitCast(ptr, value->getType()->getPointerTo(addr_space));
  builder_.CreateStore(value, ptr);
}

void shared_tile::set_vector_size(unsigned vector_size) {
  vector_size_ = vector_size;
}

void shared_tile::set_return_mode(bool return_vector){
  return_vector_ = return_vector;
}


Value* shared_tile::get_value(indices_t idx) {
  indices_t non_cst_idx, cst_idx;
  extract_constant(idx, non_cst_idx, cst_idx);
  Value *&base_ptr = ptr_cache_[non_cst_idx];
  unsigned vector_size = vector_size_;
  Type *ty = ty_;
  if(ty->isHalfTy() && (vector_size % 2 == 0)){
    ty = IntegerType::get(ty->getContext(), 32);
    vector_size = vector_size / 2;
  }
  if(base_ptr == nullptr){
//    BasicBlock* store = builder_.GetInsertBlock();
//    if(!non_cst_idx.empty())
//    if(isa<Instruction>(non_cst_idx.front())){
//      builder_.SetInsertPoint((Instruction*)non_cst_idx.front());
//    }
    base_ptr = builder_.CreateGEP(ptr_, shared_offset(builder_, shapes_, perm_, order_, non_cst_idx));
    if(vector_size_ > 1){
      Type *vec_ty = VectorType::get(ty, vector_size);
      Type *vec_ptr_ty = PointerType::get(vec_ty, base_ptr->getType()->getPointerAddressSpace());
      base_ptr = builder_.CreateBitCast(base_ptr, vec_ptr_ty);
    }
//    builder_.SetInsertPoint(store);
  }
  Value *offset = shared_offset(builder_, shapes_, perm_, order_, cst_idx);
  Value *div = offset;
  if(vector_size_ > 1)
    div = builder_.CreateUDiv(offset, builder_.getInt32(vector_size_));
  Value *ptr = builder_.CreateGEP(base_ptr, div);
  Value *result = builder_.CreateLoad(ptr);
  if(return_vector_ == false && vector_size_ > 1) {
    Value *rem = builder_.CreateURem(offset, builder_.getInt32(vector_size_));
    result = builder_.CreateExtractElement(result, rem);
  }
  return result;
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

/* convert ir::type to Type */
Type *selection::llvm_type(ir::type *ty, LLVMContext &ctx) {
  // function
  if(auto* tt = dynamic_cast<ir::function_type*>(ty)){
    Type *return_ty = llvm_type(tt->get_return_ty(), ctx);
    std::vector<Type*> param_tys;
    std::transform(tt->params_begin(), tt->params_end(), std::back_inserter(param_tys),
                   [this,&ctx](ir::type* t){ return llvm_type(t, ctx);});
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

Type *type(ir::type *ty, LLVMContext &ctx) {
  // function
  if(auto* tt = dynamic_cast<ir::function_type*>(ty)){
    Type *return_ty = type(tt->get_return_ty(), ctx);
    std::vector<Type*> param_tys;
    std::transform(tt->params_begin(), tt->params_end(), std::back_inserter(param_tys),
                   [&ctx](ir::type* t){ return type(t, ctx);});
    return FunctionType::get(return_ty, param_tys, false);
  }
  // pointer
  if(ty->is_pointer_ty()){
    Type *elt_ty = type(ty->get_pointer_element_ty(), ctx);
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


/*  -------------------
 *  ---- Init Axes ----
 *  ------------------- */

// Grid construction
std::vector<Value*> delinearize(Value *trailing, const std::vector<int>& order, std::vector<int> &shapes, IRBuilder<> &builder){
  size_t dim = shapes.size();
  std::vector<Value*> result(dim);
  for(unsigned k = 0; k < dim - 1; k++){
    Constant *dim_k = builder.getInt32(shapes[order[k]]);
    Value *rem = builder.CreateURem(trailing, dim_k);
    trailing = builder.CreateUDiv(trailing, dim_k);
    result[order[k]] = rem;
  }
  result[order[dim - 1]] = trailing;
  return result;
}

inline int32_t ceil(int32_t num, int32_t div){
  return (num + div - 1)/div;
}

/*  -------------------
 *  ---- Init Tiles ----
 *  ------------------- */

void selection::create_shared_tile(ir::value *v, IRBuilder<> &builder, Value *sh_mem_ptr) {
  if(tmap_.find(v) != tmap_.end())
    return;
  analysis::layout_shared_t *layout = (analysis::layout_shared_t*)layouts_->get(v);
  auto order = layout->order;
  auto shapes = layout->shapes;
  shapes[order[0]] += layout->pad;

  Type* ty = llvm_type(v->get_type()->get_scalar_ty(), builder.getContext());
  // shared copy
  PointerType *ptr_ty = ty->getPointerTo(sh_mem_ptr->getType()->getPointerAddressSpace());
  // double-buffered
  if(layout->double_buffer) {
    auto info = *layout->double_buffer;
    ir::phi_node *phi = info.phi;
    BasicBlock *parent = (BasicBlock*)vmap_[phi->get_parent()];
    if(parent->empty())
      builder.SetInsertPoint(parent);
    else
      builder.SetInsertPoint(&*parent->getFirstNonPHI());
    // create double-buffered pointer
    PHINode *ptr = builder.CreatePHI(ptr_ty, 2);
    PHINode *offset = builder.CreatePHI(builder.getInt32Ty(), 2);
    // next pointer
    Value *pre_ptr = builder.CreateGEP(sh_mem_ptr, builder.getInt32(alloc_->offset(v)));
    pre_ptr = builder.CreateBitCast(pre_ptr, ptr->getType());
    Value *next_ptr = builder.CreateGEP(ptr, offset, "next_ptr");
    tmap_.insert({phi, new shared_tile(ty, shapes, order, ptr, builder, offset)});
    tmap_.insert({v, new shared_tile(ty, shapes, order, pre_ptr, builder)});
    tmap_.insert({info.latch, new shared_tile(ty, shapes, order, next_ptr, builder)});
  }
  else {
    size_t offset = alloc_->offset(v);
    Value *ptr = builder.CreateGEP(sh_mem_ptr, builder.getInt32(offset));
    ptr = builder.CreateBitCast(ptr, ptr_ty);
    tmap_.insert({v, new shared_tile(ty, shapes, order, ptr, builder)});
  }
}


bool is_trans(ir::value *v) {
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


void selection::lower_value(ir::value *src, IRBuilder<> &builder, generator* gen, std::set<ir::value*>& seen) {
  if(!seen.insert(src).second)
    return;

  BasicBlock *current = builder.GetInsertBlock();
  if(src->get_type()->is_tile_ty()){
    builder.SetInsertPoint(&*builder.GetInsertBlock()->getParent()->begin());
    auto *i = dynamic_cast<ir::instruction*>(src);
    if(i && layouts_->get(i)->type == analysis::SHARED)
      create_shared_tile(i, builder, sh_mem_ptr_);
    else
      tmap_[src] = ((machine_layout_distributed_t*)gen->get_machine_layout(layouts_->get(src)))->create(src);
  }
  builder.SetInsertPoint(current);


  auto *inst = dynamic_cast<ir::instruction*>(src);
  if(inst && !dynamic_cast<ir::phi_node*>(src))
    for(ir::value *op: inst->ops())
      lower_value(op, builder, gen, seen);

  builder.SetInsertPoint(current);
  auto *phi = dynamic_cast<ir::phi_node*>(src);
  if(phi && !current->empty() && current->getFirstNonPHI())
    builder.SetInsertPoint(&*current->getFirstNonPHI());

  if(auto *usr = dynamic_cast<ir::user*>(src))
    usr->accept(gen);

  if(phi && !current->empty() && current->getFirstNonPHI())
    builder.SetInsertPoint(current);
}

/*  ----------------------------
 *  ---- Generate LLVM code ----
 *  ---------------------------- */

inline llvm::Attribute llvm_attr(llvm::LLVMContext& ctx, ir::attribute attr) {
  switch(attr.get_kind()){
    case ir::noalias: return llvm::Attribute::get(ctx, llvm::Attribute::NoAlias);
    case ir::readonly: return llvm::Attribute::get(ctx, llvm::Attribute::ReadOnly);
    case ir::writeonly: return llvm::Attribute::get(ctx, llvm::Attribute::WriteOnly);
    case ir::aligned: return llvm::Attribute::get(ctx, llvm::Attribute::Alignment, attr.get_value());
    default: throw std::runtime_error("cannot convert ir::attribute_t to llvm::Attribute");
  }
}


Value* selection::alloc_shared(IRBuilder<> &builder, Module& dst) {
  Value *ret = nullptr;
  LLVMContext &ctx = builder.getContext();
  if(tgt_->is_gpu())
  if(unsigned alloc_size = alloc_->allocated_size()){
    Type *int_8_ty = Type::getInt8Ty(ctx);
    ArrayType *array_ty = ArrayType::get(int_8_ty, alloc_size);
    Type *ptr_ty = PointerType::get(int_8_ty, 3);
    GlobalVariable *sh_mem_array =
      new GlobalVariable(dst, array_ty, false, GlobalVariable::ExternalLinkage,
                         nullptr, "__shared_ptr", nullptr, GlobalVariable::NotThreadLocal, 3);
    ret = builder.CreateBitCast(sh_mem_array, ptr_ty);
  }
  return ret;
}

void selection::run(ir::module &src, Module &dst) {
  vmap_.clear();
  tmap_.clear();

  LLVMContext &dst_ctx = dst.getContext();
  IRBuilder<> dst_builder(dst_ctx);

  // allocate shared memory
  sh_mem_ptr_ = alloc_shared(dst_builder, dst);

  // iterate over functions
  std::set<ir::value*> seen;

  // create tile
  generator gen(&dst_ctx, &dst, &dst_builder, a_axes_, axes_,  vmap_, tmap_, tgt_, layouts_, alignment_, alloc_, sh_mem_ptr_,
                offset_a_i_, offset_a_k_, offset_b_j_, offset_b_k_, num_packs_0_, num_packs_1_, pack_size_0_, pack_size_1_, num_warps_ );

  for(ir::alloc_const *x: src.allocs())
    x->accept(&gen);

  for(ir::function *fn: src.get_function_list()) {

    fn->accept(&gen);

    // initialize layouts
    for(auto x: layouts_->get_all())
      x.second->accept(&gen);

    // generate LLVM-IR code
    std::map<ir::basic_block*, BasicBlock*> last_block;
    for(ir::basic_block *block: fn->blocks()) {
      BasicBlock *parent = (BasicBlock*)vmap_[block];
      dst_builder.SetInsertPoint(parent);
      for(ir::instruction *i: block->get_inst_list())
        lower_value(i, dst_builder, &gen, seen);
      last_block[block] = dst_builder.GetInsertBlock();
    }

    // finalize double-buffering
    for(const auto& x: layouts_->get_all()) {
      if(x.second->double_buffer) {
        auto info = *x.second->double_buffer;
        ir::phi_node *phi = info.phi;
        PHINode *ptr = (PHINode*)((shared_tile*)tmap_.at(phi))->get_pointer();
        PHINode *offset = (PHINode*)((shared_tile*)tmap_.at(phi))->get_offset();
        for(unsigned n = 0; n < phi->get_num_incoming(); n++){
          ir::basic_block* inc_block = phi->get_incoming_block(n);
          ir::value* inc_val = phi->get_incoming_value(n);
          BasicBlock *llvm_inc_block = last_block.at(inc_block);
          shared_tile *inc_shared = (shared_tile*)tmap_.at(inc_val);
          if(inc_val == info.latch){
            dst_builder.SetInsertPoint(llvm_inc_block->getTerminator());
            Value *next_offset = dst_builder.CreateNeg(offset);
            offset->addIncoming(next_offset, llvm_inc_block);
          }
          else {
            unsigned num_bytes = x.second->ty->get_primitive_size_in_bits() / 8;
            offset->addIncoming(dst_builder.getInt32(x.second->size / (2*num_bytes)), llvm_inc_block);
          }
          ptr->addIncoming(inc_shared->get_pointer(), llvm_inc_block);
        }
      }
    }

    // finalize phi
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *inst: block->get_inst_list())
    if(auto *phi = dynamic_cast<ir::phi_node*>(inst)){
      if(tmap_.find(phi) == tmap_.end() ||
        !dynamic_cast<shared_tile*>(tmap_.at(phi))) {
        for(unsigned n = 0; n < phi->get_num_incoming(); n++){
          ir::value *inc_val = phi->get_incoming_value(n);
          ir::basic_block *inc_block = phi->get_incoming_block(n);
          BasicBlock *llvm_inc_block = last_block.at(inc_block);
          if(phi->get_type()->is_tile_ty()) {
            distributed_tile *phi_tile = (distributed_tile*)tmap_.at(phi);
            distributed_tile *inc_tile = (distributed_tile*)tmap_.at(inc_val);
            phi_tile->for_each([&](indices_t idx){
              PHINode *llvm_phi = (PHINode*)phi_tile->get_value(idx);
              Value *llvm_inc_val = inc_tile->get_value(idx);
              llvm_phi->addIncoming(llvm_inc_val, llvm_inc_block);
            });
          }
          else {
            PHINode *llvm_phi = (PHINode*)vmap_.at(phi);
            Value *llvm_inc_val = vmap_.at(inc_val);
            llvm_phi->addIncoming(llvm_inc_val, llvm_inc_block);
          }
        }
      }
    }
  }
}





void generator::visit_phi_node(ir::phi_node* phi) {
  Type *ty = type(phi->get_type()->get_scalar_ty(), *ctx_);
  unsigned num_ops = phi->get_num_operands();
  for_each(phi, [&](indices_t idx){
    set_value(phi, idx, builder_->Insert(PHINode::Create(ty, num_ops)));
  });
}

void generator::visit_binary_operator(ir::binary_operator*binop) {
  for_each(binop, [&](indices_t idx){
    Value *lhs = get_value(binop->get_operand(0), idx);
    Value *rhs = get_value(binop->get_operand(1), idx);
    Value *ret = builder_->Insert(BinaryOperator::Create(llvm_op(binop->get_op()), lhs, rhs));
    set_value(binop, idx, ret);
  });
}

void generator::visit_getelementptr_inst(ir::getelementptr_inst* gep) {
  for_each(gep, [&](indices_t idx){
    Value *ptr = get_value(gep->get_operand(0), idx);
    std::vector<Value*> idx_vals;
    std::transform(gep->idx_begin(), gep->idx_end(), std::back_inserter(idx_vals),
                   [&](ir::value* x){ return get_value(x, idx);});
    Type *source_ty = type(gep->get_source_elt_ty()->get_scalar_ty(), *ctx_);
    Value *ret = builder_->Insert(GetElementPtrInst::CreateInBounds(source_ty, ptr, idx_vals));
    set_value(gep, idx, ret);
  });
}

void generator::visit_icmp_inst(ir::icmp_inst* icmp) {
  for_each(icmp, [&](indices_t idx){
    ir::cmp_pred_t pred = icmp->get_pred();
    Value *lhs = get_value(icmp->get_operand(0), idx);
    Value *rhs = get_value(icmp->get_operand(1), idx);
    Value *ret = builder_->Insert(CmpInst::Create(Instruction::ICmp, llvm_pred(pred), lhs, rhs));
    set_value(icmp, idx, ret);
  });
}

void generator::visit_fcmp_inst(ir::fcmp_inst* fcmp) {
  for_each(fcmp, [&](indices_t idx){
    ir::cmp_pred_t pred = fcmp->get_pred();
    Value *lhs = get_value(fcmp->get_operand(0), idx);
    Value *rhs = get_value(fcmp->get_operand(1), idx);
    Value *ret = builder_->Insert(FCmpInst::Create(Instruction::FCmp, llvm_pred(pred), lhs, rhs));
    set_value(fcmp, idx, ret);
  });
}

void generator::visit_cast_inst(ir::cast_inst* cast) {
  for_each(cast, [&](indices_t idx){
    Value *arg = get_value(cast->get_operand(0), idx);
    Type *dst_ty = type(cast->get_type()->get_scalar_ty(), *ctx_);
    Value *ret = builder_->Insert(CastInst::Create(llvm_op(cast->get_op()), arg, dst_ty));
    set_value(cast, idx, ret);
  });
}

void generator::visit_return_inst(ir::return_inst* rr) {
  ir::value *ret_val = rr->get_return_value();
  builder_->Insert(ReturnInst::Create(*ctx_, ret_val ? vmap_.at(ret_val) : nullptr));
}

void generator::visit_cond_branch_inst(ir::cond_branch_inst* br) {
  BasicBlock *true_dest  = (BasicBlock*)vmap_.at(br->get_true_dest());
  BasicBlock *false_dest = (BasicBlock*)vmap_.at(br->get_false_dest());
  Value *cond = vmap_.at(br->get_cond());
  builder_->Insert(BranchInst::Create(true_dest, false_dest, cond));
}

void generator::visit_uncond_branch_inst(ir::uncond_branch_inst* br) {
  BasicBlock *dest = (BasicBlock*)vmap_.at(br->get_dest());
  builder_->Insert(BranchInst::Create(dest));
}


void generator::visit_unmasked_load_inst(ir::unmasked_load_inst* x) {
  distributed_tile* result = (distributed_tile*)tmap_.at(x);
  // find vector size
  ir::value *ptr = x->get_pointer_operand();
  size_t ld = layouts_->get(ptr)->order[0];
  unsigned alignment = alignment_->get(ptr, ld);
  unsigned vector_size = std::min<unsigned>(result->axis(ld).contiguous, alignment);
  distributed_tile *pointers = (distributed_tile*)tmap_.at(ptr);
  // vector loads
  std::map<unsigned, Value*> packets;
  result->for_each([&](indices_t idx){
    unsigned linear = result->get_linear_index(idx);
    unsigned id = linear / vector_size;
    if(linear % vector_size == 0) {
      Value *ptr = pointers->get_value(idx);
      ptr = builder_->CreateBitCast(ptr, PointerType::get(VectorType::get(result->get_ty(), vector_size),
                                                        ptr->getType()->getPointerAddressSpace()));
      packets[id] = builder_->CreateLoad(ptr);
    }
  });
  // extract result element
  result->for_each([&](indices_t idx){
    unsigned linear = result->get_linear_index(idx);
    unsigned id = linear / vector_size;
    result->set_value(idx, builder_->CreateExtractElement(packets.at(id), linear % vector_size));
  });
}

void generator::visit_masked_load_inst(ir::masked_load_inst* x) {
  // find vector size
  distributed_tile* result = (distributed_tile*)tmap_.at(x);
  ir::value *ptr = x->get_pointer_operand();
  size_t ld = layouts_->get(ptr)->order[0];
  unsigned alignment = alignment_->get(ptr, ld);
  unsigned vector_size = std::min<unsigned>(result->axis(ld).contiguous, alignment);
  distributed_tile *pointers = (distributed_tile*)tmap_.at(ptr);
  distributed_tile *masks = (distributed_tile*)tmap_.at(x->get_mask_operand());
  distributed_tile *false_values = (distributed_tile*)tmap_.at(x->get_false_value_operand());
  std::map<unsigned, Value*> packets;
  result->for_each([&](indices_t idx){
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
          result_false = builder_->CreateVectorSplat(vector_size, llvm::UndefValue::get(result_false->getType()));
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
//          Type *fp16x2_pack4_ty = StructType::get(ctx, {fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty});
//          FunctionType *ty = FunctionType::get(fp16x2_pack4_ty, {mask->getType(), ptr->getType()}, false);
//          std::string asm_str = "@$0 ld.global.nc.b32 {$1, $2, $3, $4}, [$5" + offset + "];";
//          if(false_values)
//            asm_str += "\n\t@!$0 mov.v4.b32 {$1, $2, $3, $4}, {0, 0, 0, 0};";
//          InlineAsm *iasm = InlineAsm::get(ty, asm_str, "b,=r,=r,=r,=r,l", true);
//          Value *current_result = builder_->CreateCall(iasm, {mask, ptr});

      packets[id] = current_result;
    }
  });
  // extract result element
  result->for_each([&](indices_t idx){
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
  distributed_tile* scalars = (distributed_tile*)tmap_.at(st->get_value_operand());
  ir::value *mask = st->get_mask_operand();
  distributed_tile* preds = (distributed_tile*)tmap_.at(mask);
  ptrs->for_each([&](indices_t idx){
    Value *scalar = scalars->get_value(idx);
    Value *ptr = ptrs->get_value(idx);
    Value *pred = preds->get_value(idx);
    Function *parent = builder_->GetInsertBlock()->getParent();
    BasicBlock *mask_then_bb = BasicBlock::Create(*ctx_, "mask_then", parent);
    BasicBlock *mask_done_bb = BasicBlock::Create(*ctx_, "mask_done", parent);
    builder_->CreateCondBr(pred, mask_then_bb, mask_done_bb);
    builder_->SetInsertPoint(mask_then_bb);
    builder_->CreateStore(scalar, ptr);
    builder_->CreateBr(mask_done_bb);
    builder_->SetInsertPoint(mask_done_bb);
//      std::string offset = "";
//      if(GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(ptr))
//      if(gep->getNumIndices() == 1)
//      if(ConstantInt *cst = dyn_cast<ConstantInt>(gep->idx_begin())){
//        offset = " + " + std::to_string(cst->getValue().getSExtValue()*4);
//      }
//      FunctionType *ty = FunctionType::get(Type::getVoidTy(ctx), {pred->getType(), ptr->getType(), scalar->getType()}, false);
//      std::string asm_str = "@$0 st.global.b32 [$1" + offset + "], $2;";
//      InlineAsm *iasm = InlineAsm::get(ty, asm_str, "b,l,f", true);
//      builder.CreateCall(iasm, {pred, ptr, scalar});
  });
}


void generator::visit_reshape_inst(ir::reshape_inst* reshape) {
  distributed_tile* result = (distributed_tile*)tmap_.at(reshape);
  ir::value* in = reshape->get_operand(0);
  distributed_tile *in_tile = (distributed_tile*)tmap_.at(in);
  for_each(reshape, [&](indices_t out_idx){
    unsigned pos = result->get_linear_index(out_idx);
    indices_t in_idx = in_tile->get_ordered_indices(pos);
    result->set_value(out_idx, in_tile->get_value(in_idx));
  });
}

void generator::visit_splat_inst(ir::splat_inst* splat) {
  Value *in = get_value(splat->get_operand(0), {});
  for_each(splat, [&](indices_t idx){
    set_value(splat, idx, in);
  });
}

void generator::visit_broadcast_inst(ir::broadcast_inst* bcast) {
  distributed_tile* result = (distributed_tile*)tmap_.at(bcast);
  ir::value* in = bcast->get_operand(0);
  const auto& in_shapes = in->get_type()->get_tile_shapes();
  distributed_tile *in_tile = (distributed_tile*)tmap_.at(in);
  result->for_each([&](indices_t out_idx){
    indices_t in_idx = out_idx;
    for(size_t k = 0; k < in_idx.size(); k++){
      if(in_shapes[k] == 1)
        in_idx[k] = builder_->getInt32(0);
    }
    result->set_value(out_idx, in_tile->get_value(in_idx));
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

void generator::visit_atomic_cas_inst(ir::atomic_cas_inst* cas) {
  BasicBlock *current = builder_->GetInsertBlock();
  Module *module = current->getModule();
  Value *tid = tgt_->get_local_id(module, *builder_, 0);
  Value *pred = builder_->CreateICmpEQ(tid, builder_->getInt32(0));
  BasicBlock *tid_0_bb = BasicBlock::Create(*ctx_, "tid_0", current->getParent());
  BasicBlock *tid_0_done_bb = BasicBlock::Create(*ctx_, "tid_0_done", current->getParent());
  Value *ptr = builder_->CreateGEP(sh_mem_ptr_, builder_->getInt32(alloc_->offset(cas)));
  ptr = builder_->CreateBitCast(ptr, PointerType::get(builder_->getInt32Ty(), ptr->getType()->getPointerAddressSpace()));
  tgt_->add_memfence(module, *builder_);
  tgt_->add_barrier(module, *builder_);
  builder_->CreateCondBr(pred, tid_0_bb, tid_0_done_bb);
  builder_->SetInsertPoint(tid_0_bb);
  Value *cas_ptr = vmap_.at(cas->get_operand(0));
  Value *cas_cmp = vmap_.at(cas->get_operand(1));
  Value *cas_val = vmap_.at(cas->get_operand(2));
  Value *old = builder_->CreateAtomicCmpXchg(cas_ptr, cas_cmp, cas_val, AtomicOrdering::Monotonic, AtomicOrdering::Monotonic);
  old = builder_->CreateExtractValue(old, {0});
  builder_->CreateStore(old, ptr);
  builder_->CreateBr(tid_0_done_bb);
  builder_->SetInsertPoint(tid_0_done_bb);
  tgt_->add_memfence(module, *builder_);
  tgt_->add_barrier(module, *builder_);
  vmap_[cas] = builder_->CreateLoad(ptr);
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
  vmap_[xchg] = builder_->CreateAtomicRMW(AtomicRMWInst::Xchg, rmw_ptr, rmw_val, AtomicOrdering::Monotonic, SyncScope::System);
  builder_->CreateBr(tid_0_done_bb);
  builder_->SetInsertPoint(tid_0_done_bb);
  tgt_->add_memfence(module, *builder_);
  tgt_->add_barrier(module, *builder_);
}

void generator::visit_atomic_add_inst(ir::atomic_add_inst*) {
  throw std::runtime_error("unsupported");
}

void generator::visit_hmma_dot(ir::dot_inst* dot, distributed_tile *TC, shared_tile *TA, shared_tile *TB, distributed_tile *TD, unsigned NK) {
  const auto& shapes = dot->get_type()->get_tile_shapes();

  TA->set_vector_size(4*pack_size_0_);
  TB->set_vector_size(4*pack_size_1_);
  TA->set_return_mode(true);
  TB->set_return_mode(true);

  std::map<std::vector<Value*>, std::vector<Value*>> fcs;

  TC->for_each([&](indices_t idx){
    std::vector<Value*> key(idx.size() - 2);
    std::copy(idx.begin() + 2, idx.end(), key.begin());
    fcs[key].push_back(TD->get_value(idx));
  });

  Type *fp32_ty = builder_->getFloatTy();
  Type *fp16x2_ty = VectorType::get(builder_->getHalfTy(), 2);
  Type *fp32_pack8_ty = StructType::get(*ctx_, {fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty});
  FunctionType *mma_ty = FunctionType::get(fp32_pack8_ty, {fp16x2_ty, fp16x2_ty, fp16x2_ty, fp16x2_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty, fp32_ty}, false);

  Value *offset_a_i = offset_a_i_;
  Value *offset_a_k = offset_a_k_;
  Value *offset_b_j = offset_b_j_;
  Value *offset_b_k = offset_b_k_;

  Value* u_thread_id = tgt_->get_local_id(builder_->GetInsertBlock()->getModule(), *builder_, 0);

  auto ord_a = layouts_->get(dot->get_operand(0))->order;
  auto ord_b = layouts_->get(dot->get_operand(1))->order;

  bool is_a_trans = is_trans(dot->get_operand(0));
  bool is_b_trans = is_trans(dot->get_operand(1));
  bool is_a_row = is_a_trans ^ (ord_a[ord_a.size() - 2] == 1);
  bool is_b_row = is_b_trans ^ (ord_b[ord_b.size() - 2] == 1);


  if(is_a_row){
    offset_a_i = builder_->CreateAdd(offset_a_i, builder_->CreateURem(u_thread_id, builder_->getInt32(4)));
    offset_a_k = builder_->getInt32(0);
  }
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

  unsigned fpw_0 = layouts_->get(dot)->fpw.at(0);
  unsigned fpw_1 = layouts_->get(dot)->fpw.at(1);
  unsigned wts_0 = fpw_0 * 8;
  unsigned wts_1 = fpw_1 * 8;
  unsigned wpt_0 = layouts_->get(dot)->wpt.at(0);
  unsigned wpt_1 = layouts_->get(dot)->wpt.at(1);
  unsigned stride_rep_i = wpt_0 * wts_0;
  unsigned stride_rep_j = wpt_1 * wts_1;
  unsigned num_rep_i = shapes[0] / stride_rep_i;
  unsigned ld_fc = num_rep_i * 2;


  for(auto& x: fcs){
    std::vector<Value *>& fc = x.second;
    for(unsigned pack_i = 0; pack_i < num_packs_0_; pack_i++)
    for(unsigned pack_j = 0; pack_j < num_packs_1_; pack_j++){
    for(unsigned K = 0; K < NK; K += 4){
      Value *_K = builder_->getInt32(K);
      Value *current_offset_a_i = builder_->CreateAdd(offset_a_i, builder_->getInt32(pack_i*stride_rep_i*pack_size_0_));
      Value *current_offset_b_i = builder_->CreateAdd(offset_b_j, builder_->getInt32(pack_j*stride_rep_j*pack_size_1_));
      indices_t idx_a = {current_offset_a_i, builder_->CreateAdd(offset_a_k, _K)};
      indices_t idx_b = {builder_->CreateAdd(offset_b_k, _K), current_offset_b_i};
      idx_a.insert(idx_a.end(), x.first.begin(), x.first.end());
      idx_b.insert(idx_b.end(), x.first.begin(), x.first.end());
      Value *ha = TA->get_value(idx_a);
      Value *hb = TB->get_value(idx_b);
      for(unsigned ii = 0; ii < pack_size_0_; ii++)
      for(unsigned jj = 0; jj < pack_size_1_; jj++){
        Value *ha0 = builder_->CreateBitCast(builder_->CreateExtractElement(ha, builder_->getInt32(ii*pack_size_0_ + 0)), fp16x2_ty);
        Value *ha1 = builder_->CreateBitCast(builder_->CreateExtractElement(ha, builder_->getInt32(ii*pack_size_0_ + 1)), fp16x2_ty);
        Value *hb0 = builder_->CreateBitCast(builder_->CreateExtractElement(hb, builder_->getInt32(jj*pack_size_0_ + 0)), fp16x2_ty);
        Value *hb1 = builder_->CreateBitCast(builder_->CreateExtractElement(hb, builder_->getInt32(jj*pack_size_0_ + 1)), fp16x2_ty);
        std::vector<size_t> idx = {
          (pack_i*2*pack_size_0_ + ii*2 + 0) + (pack_j*4*pack_size_1_ + jj*4 + 0)*ld_fc,
          (pack_i*2*pack_size_0_ + ii*2 + 0) + (pack_j*4*pack_size_1_ + jj*4 + 1)*ld_fc,
          (pack_i*2*pack_size_0_ + ii*2 + 1) + (pack_j*4*pack_size_1_ + jj*4 + 0)*ld_fc,
          (pack_i*2*pack_size_0_ + ii*2 + 1) + (pack_j*4*pack_size_1_ + jj*4 + 1)*ld_fc,
          (pack_i*2*pack_size_0_ + ii*2 + 0) + (pack_j*4*pack_size_1_ + jj*4 + 2)*ld_fc,
          (pack_i*2*pack_size_0_ + ii*2 + 0) + (pack_j*4*pack_size_1_ + jj*4 + 3)*ld_fc,
          (pack_i*2*pack_size_0_ + ii*2 + 1) + (pack_j*4*pack_size_1_ + jj*4 + 2)*ld_fc,
          (pack_i*2*pack_size_0_ + ii*2 + 1) + (pack_j*4*pack_size_1_ + jj*4 + 3)*ld_fc
        };
        Value *nc = builder_->CreateCall(mma_fn, {ha0, ha1, hb0, hb1, fc[idx[0]], fc[idx[1]], fc[idx[2]], fc[idx[3]], fc[idx[4]], fc[idx[5]], fc[idx[6]], fc[idx[7]]});
        fc[idx[0]] = builder_->CreateExtractValue(nc, {0});
        fc[idx[1]] = builder_->CreateExtractValue(nc, {1});
        fc[idx[2]] = builder_->CreateExtractValue(nc, {2});
        fc[idx[3]] = builder_->CreateExtractValue(nc, {3});
        fc[idx[4]] = builder_->CreateExtractValue(nc, {4});
        fc[idx[5]] = builder_->CreateExtractValue(nc, {5});
        fc[idx[6]] = builder_->CreateExtractValue(nc, {6});
        fc[idx[7]] = builder_->CreateExtractValue(nc, {7});
      }
    }
    }
  }

  // write back
  unsigned i = 0;
  TC->for_each([&](indices_t idx){
    std::vector<Value*> key(idx.size() - 2);
    std::copy(idx.begin() + 2, idx.end(), key.begin());
    if(i >= fcs.at(key).size())
      i = 0;
    TC->set_value(idx, fcs.at(key)[i++]);
  });

  TA->set_return_mode(false);
  TB->set_return_mode(false);

}
void generator::visit_scanline_dot(ir::dot_inst* dot, distributed_tile *TC, shared_tile *TA, shared_tile *TB, distributed_tile *TD, unsigned NK,
                                   Type *c_ty, Function *f_mul_add) {
  TA->set_vector_size(TC->axis(0).contiguous);
  TB->set_vector_size(TC->axis(1).contiguous);
  TC->for_each([&](indices_t idx){
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
      res = builder_->CreateCall(f_mul_add, {a, b, res});
    }
    TC->set_value(idx, res);
  });
}

void generator::visit_outer_dot(ir::dot_inst*, distributed_tile *TC, distributed_tile *TA, distributed_tile *TB, distributed_tile *TD, unsigned NK,
                                Type *c_ty, Function *f_mul_add) {
  TC->for_each([&](indices_t idx){
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
    res = builder_->CreateCall(f_mul_add, {a, b, res});
    TC->set_value(idx, res);
  });
}

void generator::visit_dot_inst(ir::dot_inst* dot) {
  Function *fn = builder_->GetInsertBlock()->getParent();

  distributed_tile* TC = (distributed_tile*)tmap_.at(dot);
  Module *module = fn->getParent();
  ir::value *A = dot->get_operand(0);
  ir::value *B = dot->get_operand(1);
  ir::value *D = dot->get_operand(2);

  distributed_tile *TD = (distributed_tile*)tmap_.at(D);
  Type *c_ty = type(D->get_type()->get_scalar_ty(), *ctx_);
  Function *f_mul_add = Intrinsic::getDeclaration(module, Intrinsic::fmuladd, {c_ty});
  auto A_shapes = A->get_type()->get_tile_shapes();
  size_t red_axis = 1;
  unsigned NK = A_shapes[red_axis];

  if(NK != 1) {
    shared_tile *TA = (shared_tile*)tmap_.at(A);
    shared_tile *TB = (shared_tile*)tmap_.at(B);
    if(layouts_->get(dot)->type == analysis::HMMA_884)
      visit_hmma_dot(dot, TC, TA, TB, TD, NK);
    else
      visit_scanline_dot(dot, TC, TA, TB, TD, NK, c_ty, f_mul_add);
  }
  else {
    distributed_tile *TA = (distributed_tile*)tmap_.at(A);
    distributed_tile *TB = (distributed_tile*)tmap_.at(B);
    visit_outer_dot(dot, TC, TA, TB, TD, NK, c_ty, f_mul_add);
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
    Value *sqrt = Intrinsic::getDeclaration(module, Intrinsic::sqrt, {val->getType()});
    Value *ret = builder_->CreateCall(sqrt, {val});
    set_value(sqt, idx, ret);
  });
}

void generator::visit_reduce_inst(ir::reduce_inst*) {
  throw std::runtime_error("not implemented");
}

void generator::visit_select_inst(ir::select_inst* select) {
  for_each(select, [&](indices_t idx){
    Value *pred = get_value(select->get_operand(0), idx);
    Value *if_value = get_value(select->get_operand(1), idx);
    Value *else_value = get_value(select->get_operand(2), idx);
    Value *ret = builder_->Insert(SelectInst::Create(pred, if_value, else_value));
    set_value(select, idx, ret);
  });

}

void generator::visit_copy_to_shared_inst(ir::copy_to_shared_inst* cts) {
  unsigned vector_size = 1;
  auto x_order = layouts_->get(cts)->order;
  ir::value *arg = cts->get_operand(0);
  auto arg_order = layouts_->get(arg)->order;
  // tiles
  shared_tile* result = (shared_tile*)tmap_.at(cts);
  distributed_tile* in = (distributed_tile*)tmap_.at(arg);
  if(x_order == arg_order){
    size_t ld = arg_order[0];
    vector_size = layouts_->get(arg)->nts.at(ld);
  }

  std::map<unsigned, Value*> packets;
  in->for_each([&](indices_t idx){
    unsigned linear = in->get_linear_index(idx);
    unsigned id = linear / vector_size;
    Value *in_value = in->get_value(idx);
    if(linear % vector_size == 0)
      packets[id] = UndefValue::get(VectorType::get(in_value->getType(), vector_size));
    packets[id] = builder_->CreateInsertElement(packets.at(id), in_value, linear % vector_size);
  });
  in->for_each([&](indices_t idx){
    unsigned linear = in->get_linear_index(idx);
    unsigned id = linear / vector_size;
    if(linear % vector_size == 0)
      result->set_value(idx, packets[id]);
  });
}

void generator::visit_copy_from_shared_inst(ir::copy_from_shared_inst* cfs) {
  distributed_tile* result = (distributed_tile*)tmap_.at(cfs);
  shared_tile* arg = (shared_tile*)tmap_.at(cfs->get_operand(0));
  result->for_each([&](indices_t idx){
    result->set_value(idx, arg->get_value(idx));
  });
}

void generator::visit_barrier_inst(ir::barrier_inst*) {
  Module *module = builder_->GetInsertBlock()->getModule();
  tgt_->add_barrier(module, *builder_);
}

void generator::visit_make_range_dyn(ir::make_range_dyn* x) {
  distributed_tile* result = (distributed_tile*)tmap_.at(x);
  result->for_each([&](indices_t idx){
    assert(idx.size() == 1);
    BinaryOperator *bin_add = dyn_cast<BinaryOperator>(idx[0]);
    assert(bin_add);
    Value *res = bin_add->getOperand(0);
    result->set_value(idx, res);
  });
}

void generator::visit_make_range_sta(ir::make_range_sta* x) {
  distributed_tile *T = (distributed_tile *)tmap_.at(x);
  T->for_each([&](indices_t idx){
    assert(idx.size() == 1);
    BinaryOperator *bin_add = dyn_cast<BinaryOperator>(idx[0]);
    assert(bin_add);
    Value *res = bin_add->getOperand(1);
    assert(isa<Constant>(res));
    T->set_value(idx, res);
  });
}

void generator::visit_make_range(ir::make_range* x) {
  distributed_tile *T = (distributed_tile *)tmap_.at(x);
  T->for_each([&](indices_t idx){
    assert(idx.size() == 1);
    T->set_value(idx, idx[0]);
  });
}



void generator::visit_undef_value(ir::undef_value *ud) {
  vmap_[ud] = llvm::UndefValue::get(type(ud->get_type(), *ctx_));
}

void generator::visit_constant_int(ir::constant_int *cst){
  Type *ty = type(cst->get_type()->get_scalar_ty(), *ctx_);
  vmap_[cst] = ConstantInt::get(ty, cst->get_value());
}

void generator::visit_constant_fp(ir::constant_fp *cst){
  Type *ty = type(cst->get_type()->get_scalar_ty(), *ctx_);
  vmap_[cst] = ConstantFP::get(ty, cst->get_value());
}

void generator::visit_alloc_const(ir::alloc_const *alloc) {
  unsigned size = ((ir::constant_int*)alloc->get_operand(0))->get_value();
  Type *element_ty = type(alloc->get_type()->get_pointer_element_ty(), *ctx_);
  Type *array_ty = llvm::ArrayType::get(element_ty, size);
  Value *array = new llvm::GlobalVariable(*mod_, array_ty, false, llvm::GlobalVariable::ExternalLinkage,
                                            nullptr, alloc->get_name(), nullptr, llvm::GlobalVariable::NotThreadLocal, 4);
  vmap_[alloc] = builder_->CreateBitCast(array, element_ty->getPointerTo(4));
}


void generator::visit_function(ir::function* fn) {
  LLVMContext &ctx = builder_->getContext();
  FunctionType *fn_ty = (FunctionType*)type(fn->get_fn_type(), *ctx_);
  FunctionType *dst_fn_ty = fn_ty;
  if(!tgt_->is_gpu()){
    Type *dst_fn_ret_ty = fn_ty->getReturnType();
    std::vector<Type*> dst_fn_args_ty;
    for(unsigned i = 0; i < fn_ty->getNumParams(); i++)
      dst_fn_args_ty.push_back(fn_ty->getParamType(i));
    dst_fn_args_ty.push_back(builder_->getInt32Ty());
    dst_fn_args_ty.push_back(builder_->getInt32Ty());
    dst_fn_args_ty.push_back(builder_->getInt32Ty());
    dst_fn_ty = FunctionType::get(dst_fn_ret_ty, dst_fn_args_ty, false);
  }
  Function *ret = Function::Create(dst_fn_ty, Function::ExternalLinkage, fn->get_name(), mod_);
  // set attributes
  for(auto attr_pair: fn->attrs()){
    unsigned id = attr_pair.first;
    for(ir::attribute attr: attr_pair.second)
    if(attr.is_llvm_attr())
      ret->addAttribute(id, llvm_attr(ctx, attr));
  }
  // set metadata
  tgt_->set_kernel(*builder_, ctx, mod_, ret);
  Metadata *md_args[] = {
    ValueAsMetadata::get(ret),
    MDString::get(ctx, "maxntidx"),
    ValueAsMetadata::get(builder_->getInt32(num_warps_*32))
  };
  mod_->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(MDNode::get(ctx, md_args));
  // map parameters
  for(unsigned i = 0; i < fn->args().size(); i++)
    vmap_[fn->args()[i]] = &*(ret->arg_begin() + i);
  // create blocks
  for(ir::basic_block *block: fn->blocks()) {
    BasicBlock *dst_block = BasicBlock::Create(ctx, block->get_name(), ret);
    vmap_[block] = dst_block;
  }
  builder_->SetInsertPoint((BasicBlock*)vmap_[fn->blocks()[0]]);
  fn_ = ret;
}

void generator::visit_layout_hmma_884(analysis::layout_hmma_884_t* layout) {
  machine_layouts_[layout] = new machine_layout_hmma_884_t(mod_, builder_, tgt_, type(layout->ty->get_scalar_ty(), *ctx_), a_axes_, axes_,
                                                           offset_a_i_, offset_a_k_, offset_b_j_, offset_b_k_,
                                                           pack_size_0_, pack_size_1_,
                                                           num_packs_0_, num_packs_1_,
                                                           layout);
}

void generator::visit_layout_scanline(analysis::layout_scanline_t* layout) {
  machine_layouts_[layout] = new machine_layout_scanline_t(mod_, builder_, tgt_, type(layout->ty->get_scalar_ty(), *ctx_), a_axes_, axes_, layout);
}

void generator::visit_layout_shared(analysis::layout_shared_t* layout) {

  machine_layouts_[layout] = new machine_layout_shared_t();
}

void generator::for_each(ir::value *x, const std::function<void(indices_t)>& fn) {
  if(!x->get_type()->is_tile_ty())
    return fn({});
  else {
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



shared_tile* machine_layout_shared_t::create(ir::value *v) {

}

machine_layout_distributed_t::machine_layout_distributed_t(Module *mod, Builder *builder, target *tgt, Type *ty,
                             analysis::axes *a_axes, std::map<unsigned, distributed_axis>& axes,
                             analysis::layout_t *layout)
  : mod_(mod), builder_(builder), tgt_(tgt), ty_(ty), a_axes_(a_axes), axes_(axes), layout_(layout) {

}

distributed_tile* machine_layout_distributed_t::create(ir::value *v) {
  Type *ty = type(v->get_type()->get_scalar_ty(), builder_->getContext());
  const auto &shapes = v->get_type()->get_tile_shapes();
  std::vector<distributed_axis> axes(shapes.size());
  for(size_t d = 0; d < shapes.size(); d++){
    if(shapes[d] > 1){
      unsigned x = a_axes_->get(v, d);
      axes[d] = axes_.at(x);
    }
    else{
      axes[d].contiguous = 1;
      axes[d].values = {builder_->getInt32(0)};
    }
  }
  return new distributed_tile(ty, shapes, layout_->order, axes, *builder_, false);
}

machine_layout_hmma_884_t::machine_layout_hmma_884_t(Module *mod, Builder *builder,
                          target *tgt, Type *ty, analysis::axes *a_axes,
                          std::map<unsigned, distributed_axis>& axes,
                          Value *&offset_a_i, Value *&offset_a_k, Value *&offset_b_j, Value *&offset_b_k,
                          unsigned &pack_size_0, unsigned &pack_size_1,
                          unsigned &num_packs_0, unsigned &num_packs_1,
                          analysis::layout_hmma_884_t* layout)
  : machine_layout_distributed_t(mod, builder, tgt, ty, a_axes, axes, layout),
    offset_a_i_(offset_a_i), offset_a_k_(offset_a_k), offset_b_j_(offset_b_j), offset_b_k_(offset_b_k),
    pack_size_0_(pack_size_0), pack_size_1_(pack_size_1), num_packs_0_(num_packs_0), num_packs_1_(num_packs_1) {

  Value *warp_size = builder_->getInt32(32);
  Value* u_thread_id_0 = tgt_->get_local_id(mod_, *builder_, 0);
  Value *u_thread_id = builder_->CreateURem(u_thread_id_0, warp_size);
  Value *u_warp_id = builder_->CreateUDiv(u_thread_id_0, warp_size);

  const auto& shapes = layout->shapes;
  if(shapes.size() > 3)
    throw std::runtime_error("unsupported");

  bool is_batched = shapes.size() >= 3;

  Value *_1 = builder_->getInt32(1);
  Value *_2 = builder_->getInt32(2);
  Value *_3 = builder_->getInt32(3);
  Value *_4 = builder_->getInt32(4);
  Value *_16 = builder_->getInt32(16);

  // fragments per warp
  unsigned fpw_0 = layout->fpw.at(0);
  unsigned fpw_1 = layout->fpw.at(1);
  unsigned fpw_2 = is_batched ? layout->fpw.at(2) : 1;
  // warps per tile
  unsigned wpt_0 = layout->wpt.at(0);
  unsigned wpt_1 = layout->wpt.at(1);
  unsigned wpt_2 = is_batched ? layout->wpt.at(2) : 1;
  // hmma warp tile size
  unsigned hmma_wts_0 = fpw_0 * 8;
  unsigned hmma_wts_1 = fpw_1 * 8;
  unsigned hmma_wts_2 = is_batched ? fpw_2 : 1;
  // hmma block tile size
  unsigned hmma_bts_0 = hmma_wts_0 * wpt_0;
  unsigned hmma_bts_1 = hmma_wts_1 * wpt_1;
  unsigned hmma_bts_2 = is_batched ? hmma_wts_2 * wpt_2 : 1;
  // number of repetition
  unsigned num_rep_0 = shapes[0] / hmma_bts_0;
  unsigned num_rep_1 = shapes[1] / hmma_bts_1;
  unsigned num_rep_2 = is_batched ? shapes[2] / hmma_bts_2 : 1;
  // size of each pack (interleaving)
  pack_size_0_ = std::min<unsigned>(num_rep_0, 1);
  pack_size_1_ = std::min<unsigned>(num_rep_1, 1);
  // number of packs (interleaving)
  num_packs_0_ = num_rep_0 / pack_size_0_;
  num_packs_1_ = num_rep_1 / pack_size_1_;

  /* intra warp offset */
  // offset of quad in pair
  Value *in_pair_off_a = builder_->CreateMul(builder_->CreateUDiv(builder_->CreateAnd(u_thread_id, _16), builder_->getInt32(4)),
                                           builder_->getInt32(fpw_0 * pack_size_0_));
  Value *in_pair_off_b = builder_->CreateMul(builder_->CreateUDiv(builder_->CreateAnd(u_thread_id, _16), builder_->getInt32(4)),
                                           builder_->getInt32(fpw_1 * pack_size_1_));

  // Quad pair id
  Value *pair_a_id = builder_->CreateUDiv(builder_->CreateURem(u_thread_id, _16), _4);
  Value *pair_b_id = builder_->CreateUDiv(builder_->CreateURem(u_thread_id, _16), _4);
  pair_a_id = builder_->CreateURem(pair_a_id, builder_->getInt32(fpw_0));
  pair_b_id = builder_->CreateUDiv(pair_b_id, builder_->getInt32(fpw_0));
  pair_b_id = builder_->CreateURem(pair_b_id, builder_->getInt32(fpw_1));
  // Quad pair offset
  Value *pair_a_off = builder_->CreateMul(pair_a_id, builder_->getInt32(4 * pack_size_0_));
  Value *pair_b_off = builder_->CreateMul(pair_b_id, builder_->getInt32(4 * pack_size_1_));

  /* inter warp offset */
  Value *warp_id_0 = builder_->CreateURem(u_warp_id, builder_->getInt32(wpt_0));
  Value *warp_id_12 = builder_->CreateUDiv(u_warp_id, builder_->getInt32(wpt_0));
  Value *warp_id_1 = builder_->CreateURem(warp_id_12, builder_->getInt32(wpt_1));
  Value *warp_id_2 = builder_->CreateUDiv(warp_id_12, builder_->getInt32(wpt_1));
  Value *warp_offset_i = builder_->CreateMul(warp_id_0, builder_->getInt32(hmma_wts_0 * pack_size_0_));
  Value *warp_offset_j = builder_->CreateMul(warp_id_1, builder_->getInt32(hmma_wts_1 * pack_size_1_));

  /* offsets */
  // a offset
  offset_a_i_ = builder_->CreateAdd(warp_offset_i, builder_->CreateAdd(pair_a_off, in_pair_off_a));
  offset_a_k_ = builder_->CreateAnd(u_thread_id, _3);
  // b offsets
  offset_b_j_ = builder_->CreateAdd(warp_offset_j, builder_->CreateAdd(pair_b_off, in_pair_off_b));
  offset_b_k_ = builder_->CreateAnd(u_thread_id, _3);

  // c offsets
  Value *offset_c_i = builder_->CreateAdd(builder_->CreateAnd(u_thread_id, _1), offset_a_i_);
  Value *offset_c_j = builder_->CreateAdd(builder_->CreateAnd(u_thread_id, _2),
                                        builder_->CreateAdd(warp_offset_j, pair_b_off));

  /* indices */
  // i indices
  std::vector<Value*> idx_i;
  for(unsigned pack = 0; pack < num_packs_0_; pack++)
  for(unsigned ii = 0; ii < pack_size_0_; ii++)
  for(unsigned i = 0; i < 2; i++){
    idx_i.push_back(builder_->CreateAdd(offset_c_i, builder_->getInt32(pack*hmma_bts_0*pack_size_0_ + ii*4 + i*2)));
  }
  // j indices
  std::vector<Value*> idx_j;
  for(unsigned pack = 0; pack < num_packs_1_; pack++)
  for(unsigned jj = 0; jj < pack_size_1_; jj++)
  for(unsigned j = 0; j < 2; j++){
    idx_j.push_back(builder_->CreateAdd(offset_c_j, builder_->getInt32(pack*hmma_bts_1*pack_size_1_ + jj*4 + j*4*fpw_1*pack_size_1_)));
    idx_j.push_back(builder_->CreateAdd(offset_c_j, builder_->getInt32(pack*hmma_bts_1*pack_size_1_ + jj*4 + j*4*fpw_1*pack_size_1_ + 1)));
  }
  // z indices
  std::vector<Value*> idx_z;
  for(unsigned pack = 0; pack < num_rep_2; pack++)
    idx_z.push_back(builder_->CreateAdd(warp_id_2, builder_->getInt32(pack*hmma_bts_2)));


  /* axes */
  axes_[layout->axes[0]] = distributed_axis{1, idx_i, warp_id_0};
  axes_[layout->axes[1]] = distributed_axis{1, idx_j, warp_id_1};
  if(is_batched)
    axes_[layout->axes[2]] = distributed_axis{1, idx_z, warp_id_2};
}


machine_layout_scanline_t::machine_layout_scanline_t(Module *mod, Builder *builder,
                                                     target *tgt, Type *ty,
                                                     analysis::axes *a_axes, std::map<unsigned, distributed_axis> &axes,
                                                     analysis::layout_scanline_t* layout)
  : machine_layout_distributed_t(mod, builder, tgt, ty, a_axes, axes, layout) {

  Value *warp_size = builder_->getInt32(32);
  Value* u_thread_id_0 = tgt_->get_local_id(mod_, *builder_, 0);
  Value *u_thread_id = builder_->CreateURem(u_thread_id_0, warp_size);
  Value *u_warp_id = builder_->CreateUDiv(u_thread_id_0, warp_size);

  auto order = layout->order;
  const auto& shapes = layout->shapes;
  size_t dim = shapes.size();
  std::vector<int> nts = layout->nts;
  std::vector<int> mts = layout->mts;
  Value* full_thread_id = builder_->CreateAdd(builder_->CreateMul(u_warp_id, builder_->getInt32(32)), u_thread_id);
  std::vector<Value*> thread_id = delinearize(full_thread_id, order, mts, *builder_);
  // Create axes
  for(unsigned k = 0; k < dim; k++) {
    std::string str_k = std::to_string(k);
    Value *contiguous_k = builder_->getInt32(nts[k]);
    Value *scaled_thread_id = builder_->CreateMul(thread_id[k], contiguous_k);
    unsigned per_block  = nts[k] * mts[k];
    unsigned per_thread = nts[k] * shapes[k] / per_block;
    std::vector<Value*> idx_list(per_thread);
    for(unsigned n = 0 ; n < per_thread; n++){
      unsigned offset = n / nts[k] * per_block + n % nts[k];
      idx_list[n] = builder_->CreateAdd(scaled_thread_id, builder_->getInt32(offset), "idx_" + str_k + "_" + std::to_string(n));
    }
    axes_[layout->axes[k]] = distributed_axis{nts[k], idx_list, thread_id[k]};
  }
}



}
}
