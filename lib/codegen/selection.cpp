#include "codegen/selection.h"
#include "codegen/tune.h"
#include "codegen/allocation.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "ir/context.h"
#include "ir/module.h"
#include "ir/function.h"
#include "ir/type.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"

namespace tdl{
namespace codegen{

using namespace llvm;

/* Distributed Tile */
void distributed_tile::init_indices() {
  std::vector<size_t> id(axes_.size(), 0);
  size_t k = 0;
  while(true) {
    indices_t current;
    for(size_t d = 0; d < id.size(); d++)
      current.push_back(axes_[d].values[id[d]]);
    indices_[current] = indices_.size();
    id[0]++;
    while(id[k] == axes_[k].values.size()){
      if(k == id.size() - 1)
        return;
      id[k++] = 0;
      id[k]++;
    }
    k = 0;
  }
}


distributed_tile::distributed_tile(Type *ty, const shapes_t &shapes, const axes_t &axes)
    : tile(ty, shapes), axes_(axes) {
  init_indices();
  for(size_t i = 0; i < indices_.size(); i++)
    values_.push_back(UndefValue::get(ty_));
}

/* Serialized distributed tile */
void serialized_distributed_tile::set_value(indices_t idx, Value *v) {
  values_[indices_[idx]] = v;
}

void serialized_distributed_tile::get_value(indices_t idx) {
  return values_[indices_[idx]];
}

void serialized_distributed_tile::for_each(std::function<void (indices_t)> fn) {
  for(auto &idx: indices_)
    fn(idx.first);
}

/* Vectorized distributed tile */
llvm::Type *vectorized_distributed_tile::make_vector_ty(llvm::Type *ty, size_t vector_size) {
  if(vector_size == 1)
    return ty;
  return VectorType::get(ty, vector_size);
}

vectorized_distributed_tile::vectorized_distributed_tile(Type *ty, const shapes_t &shapes, const axes_t &axes, llvm::IRBuilder<> &builder)
    : distributed_tile(make_vector_ty(ty, axes[0].contiguous), shapes), axes_(axes), builder_(builder) {
  vector_size_ = 1;
  if(ty_->isVectorTy())
    vector_size_ = ty_->getVectorNumElements();
}


void distributed_tile::set_value(indices_t idx, Value *v) {
  unsigned value_idx = indices_[idx];
  Value *&result = values_[value_idx/vector_size_*vector_size_];
  if(v->getType() == result->getType()) {
    assert(value_idx % vector_size_ == 0);
    result = v;
  }
  // insert scalar in vector
  else {
    std::cout << v->getType()->getScalarType()->getTypeID() << " " << result->getType()->getScalarType()->getTypeID() << std::endl;
    assert(vector_size_==1 || result->getType()->isVectorTy());
    assert(v->getType()->getScalarType() == result->getType()->getScalarType());
    result = builder_.CreateInsertElement(result, v, value_idx % vector_size_);
  }
}

Value* distributed_tile::get_value(indices_t idx) {
  unsigned value_idx = indices_[idx];
  Value *&result = values_[value_idx/vector_size_*vector_size_];
  if(vectorize_ || vector_size_ == 1) {
    assert(value_idx % vector_size_ == 0);
    return result;
  }
  // extract scalar from vector
  else {
    assert(result->getType()->isVectorTy());
    return builder_.CreateExtractElement(result, value_idx % vector_size_);
  }
  return result;
}

void distributed_tile::for_each(std::function<void (indices_t)> fn) {
  for(auto &idx: indices_) {
    if(!vectorize_ || (idx.second % vector_size_ == 0))
      fn(idx.first);
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


Value* shared_tile::shared_offset(indices_t idx) {
  Value *result = builder_.getInt32(0);
  result = builder_.CreateAdd(result, idx[0]);
  for(size_t i = 1; i < idx.size(); i++)
    result = builder_.CreateAdd(result, builder_.CreateMul(idx[i], builder_.getInt32(shapes_[i-1])));
  return result;
}

shared_tile::shared_tile(Type *ty, const shapes_t &shapes, Value *ptr, llvm::IRBuilder<> &builder):
  tile(ty, shapes), ptr_(ptr), builder_(builder) {
}

void shared_tile::set_value(indices_t idx, Value *value) {
  Value *ptr = builder_.CreateGEP(ptr_, shared_offset(idx));
  unsigned addr_space = ptr->getType()->getPointerAddressSpace();
  ptr = builder_.CreateBitCast(ptr, value->getType()->getPointerTo(addr_space));
  builder_.CreateStore(value, ptr);
}

Value* shared_tile::get_value(indices_t idx) {
  indices_t non_cst_idx, cst_idx;
  extract_constant(idx, non_cst_idx, cst_idx);
  Value *&base_ptr = ptr_cache_[non_cst_idx];
  if(base_ptr == nullptr){
    base_ptr = builder_.CreateGEP(ptr_, shared_offset(non_cst_idx));
//    Type *vec_ty = VectorType::get(base_ptr->getType()->getPointerElementType(), vec_);
//    Type *vec_ptr_ty = PointerType::get(vec_ty, base_ptr->getType()->getPointerElementType());
//    base_ptr = builder_.CreateBitCast(base_ptr, vec_ptr_ty);
  }
  Value *ptr = builder_.CreateGEP(base_ptr, shared_offset(cst_idx));
  return builder_.CreateLoad(ptr);
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

/* convert ir::constant to Constant */
Constant *selection::llvm_constant(ir::constant *cst, LLVMContext &ctx) {
  Type *dst_ty = llvm_type(cst->get_type(), ctx);
  if(auto* cc = dynamic_cast<ir::constant_int*>(cst))
    return ConstantInt::get(dst_ty, cc->get_value());
  if(auto* cc = dynamic_cast<ir::constant_fp*>(cst))
    return ConstantFP::get(dst_ty, cc->get_value());
  // unknown constant
  throw std::runtime_error("unknown conversion from ir::constant to Constant");
}


/* convert ir::instruction to llvm::Instruction */
Instruction *selection::llvm_inst(ir::instruction *inst, std::function<Value*(ir::value*)> value, IRBuilder<> &builder) {
  LLVMContext & ctx = builder.getContext();
  auto block = [&](ir::basic_block *x) { return (BasicBlock*)vmap_.at(x); };
  auto type = [&](ir::type *x) { return llvm_type(x, ctx); };
  if(auto* ii = dynamic_cast<ir::cond_branch_inst*>(inst)){
    BasicBlock *true_dest  = block(ii->get_true_dest());
    BasicBlock *false_dest = block(ii->get_false_dest());
    Value *cond = value(ii->get_cond());
    return builder.Insert(BranchInst::Create(true_dest, false_dest, cond));
  }
  if(auto* ii = dynamic_cast<ir::uncond_branch_inst*>(inst)){
    BasicBlock *dest = block(ii->get_dest());
    return builder.Insert(BranchInst::Create(dest));
  }
  if(auto* ii = dynamic_cast<ir::phi_node*>(inst)){
    Type *ty = type(ii->get_type()->get_scalar_ty());
    unsigned num_ops = ii->get_num_operands();
    return builder.Insert(PHINode::Create(ty, num_ops));
  }
  if(auto* ii = dynamic_cast<ir::return_inst*>(inst)){
    ir::value *ret_val = ii->get_return_value();
    return builder.Insert(ReturnInst::Create(ctx, ret_val?value(ret_val):nullptr));
  }
  if(auto* ii = dynamic_cast<ir::binary_operator*>(inst)){
    Value *lhs = value(ii->get_operand(0));
    Value *rhs = value(ii->get_operand(1));
    return builder.Insert(BinaryOperator::Create(ii->get_op(), lhs, rhs));
  }
  if(auto* ii = dynamic_cast<ir::icmp_inst*>(inst)){
    CmpInst::Predicate pred = ii->get_pred();
    Value *lhs = value(ii->get_operand(0));
    Value *rhs = value(ii->get_operand(1));
    return builder.Insert(CmpInst::Create(Instruction::ICmp, pred, lhs, rhs));
  }
  if(auto* ii = dynamic_cast<ir::fcmp_inst*>(inst)){
    CmpInst::Predicate pred = ii->get_pred();
    Value *lhs = value(ii->get_operand(0));
    Value *rhs = value(ii->get_operand(1));
    return builder.Insert(FCmpInst::Create(Instruction::FCmp, pred, lhs, rhs));
  }
  if(auto* ii = dynamic_cast<ir::cast_inst*>(inst)){
    Value *arg = value(ii->get_operand(0));
    Type *dst_ty = type(ii->get_type()->get_scalar_ty());
    return builder.Insert(CastInst::Create(ii->get_op(), arg, dst_ty));
  }
  if(auto* ii = dynamic_cast<ir::getelementptr_inst*>(inst)){
    std::vector<Value*> idx_vals;
    std::transform(ii->idx_begin(), ii->idx_end(), std::back_inserter(idx_vals),
                   [&value](ir::value* x){ return value(x);});
    Type *source_ty = type(ii->get_source_elt_ty()->get_scalar_ty());
    Value *arg = value(ii->get_operand(0));
    return builder.Insert(GetElementPtrInst::Create(source_ty, arg, idx_vals));
  }
  if(ir::load_inst* ii = dynamic_cast<ir::load_inst*>(inst)){
    Value *ptr = value(ii->get_pointer_operand());
    return builder.Insert(new LoadInst(ptr));
  }
  // unknown instruction
  throw std::runtime_error("unknown conversion from ir::instruction to Instruction");
}

/* convert ir::value to llvm::Value */
Value* selection::llvm_value(ir::value *v, IRBuilder<> &builder) {
  assert(!v->get_type()->is_tile_ty());
  LLVMContext &ctx = builder.getContext();
  if(vmap_.find(v) != vmap_.end())
    return vmap_.at(v);
  // create operands
  if(auto *cc = dynamic_cast<ir::constant*>(v))
    return llvm_constant(cc, ctx);
  // instruction
  if(auto *ii = dynamic_cast<ir::instruction*>(v)){
    auto value = [&](ir::value *x) { return llvm_value(x, builder); };
    return llvm_inst(ii, value, builder);
  }
  // unknown value
  throw std::runtime_error("unknown conversion from ir::value to Value");
}

// Grid construction
std::vector<Value*> delinearize(Value *trailing, std::vector<unsigned> &shapes, IRBuilder<> &builder){
  size_t dim = shapes.size();
  std::vector<Value*> result(dim);
  for(unsigned k = 0; k < dim - 1; k++){
    Constant *dim_k = builder.getInt32(shapes[k]);
    Value *rem = builder.CreateURem(trailing, dim_k);
    trailing = builder.CreateUDiv(trailing, dim_k);
    result[k] = rem;
  }
  result[dim - 1] = trailing;
  return result;
}

void selection::init_axes(ir::value *v, IRBuilder<> &builder, Value *u_thread_id, Value *u_warp_id) {
  const auto& shapes = v->get_type()->get_tile_shapes();
  size_t dim = shapes.size();
  std::vector<unsigned> contiguous(dim);
  std::vector<unsigned> warp_size(dim);
  std::vector<unsigned> n_warps(dim);
  for(unsigned i = 0; i < shapes.size(); i++){
    std::string str_i = std::to_string(i);
    contiguous[i] = *params_->get_param(v, "p0.d" + str_i);
    warp_size[i] = *params_->get_param(v, "p1.d" + str_i);
    n_warps[i] = *params_->get_param(v, "p2.d" + str_i);
  }
  std::vector<Value*> thread_id_in_warp = delinearize(u_thread_id, warp_size, builder);
  std::vector<Value*> warp_id = delinearize(u_warp_id, n_warps, builder);
  // Create axes
  for(unsigned k = 0; k < dim; k++) {
    std::string str_k = std::to_string(k);
    Value *warp_size_k = builder.getInt32(warp_size[k]);
    Value *contiguous_k = builder.getInt32(contiguous[k]);
    Value *thread_id   = builder.CreateAdd(thread_id_in_warp[k], builder.CreateMul(warp_id[k], warp_size_k));
    thread_id = builder.CreateMul(thread_id, contiguous_k);
    unsigned per_block = contiguous[k] * warp_size[k] * n_warps[k];
    unsigned per_thread = contiguous[k] * shapes[k] / per_block;
    std::vector<Value*> idx_list(per_thread);
    for(unsigned n = 0 ; n < per_thread; n++){
      unsigned offset = n / contiguous[k] * per_block + n % contiguous[k];
      idx_list[n] = builder.CreateAdd(thread_id, builder.getInt32(offset), "idx_" + str_k + "_" + std::to_string(n));
    }
    axes_[params_->get_param(v, "p0.d" + str_k)] = distributed_axis{contiguous[k], idx_list};
  }
}

void selection::create_grids(std::vector<ir::value*> &grids,
                             std::map<unsigned*, ir::value*> &references,
                             ir::function *fn) {
  // get number of dimensions greater than 1
  auto get_tile_gt1_dim = [&](ir::value *v){
    unsigned result = 0;
    for(unsigned shape: v->get_type()->get_tile_shapes()) {
      result += (shape > 1)?shape:0;
    }
    return result;
  };
  // bind references
  std::set<ir::value*> seen;
  std::function<void(ir::value*)> bind_references = [&](ir::value *v)
  {
    // skip
    if(!v->get_type()->is_tile_ty() || !seen.insert(v).second)
      return;
    // recurse
    if(auto *user = dynamic_cast<ir::user*>(v))
      for(ir::value *op: user->ops())
        bind_references(op);
    // bind
    const auto& shapes = v->get_type()->get_tile_shapes();
    bool is_shared = dynamic_cast<ir::copy_to_shared_inst*>(v);
    if(is_shared)
      return;
    for(size_t d = 0; d < shapes.size(); d++){
      if(shapes[d] == 1)
        continue;
      unsigned *x = params_->get_param(v, "p0.d" + std::to_string(d));
      ir::value *&r = references[x];
      if(!r || get_tile_gt1_dim(v) > get_tile_gt1_dim(r))
        r = v;
    }
  };

  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list())
    bind_references(i);

  // create grid
  for(auto &ref: references)
    if(std::find(grids.begin(), grids.end(), ref.second) == grids.end())
      grids.push_back(ref.second);
}

void selection::create_tile(ir::value *v, IRBuilder<> &builder,
                            const std::map<unsigned*, ir::value*>& references,
                            std::set<ir::value*> &seen, Value *sh_mem_ptr) {
  if(!v->get_type()->is_tile_ty() || !seen.insert(v).second)
    return;
  if(auto *user = dynamic_cast<ir::user*>(v))
    for(ir::value *op: user->ops())
      create_tile(op, builder, references, seen, sh_mem_ptr);
  LLVMContext &ctx = builder.getContext();
  const auto& shapes = v->get_type()->get_tile_shapes();
  Type* ty = llvm_type(v->get_type()->get_scalar_ty(), ctx);
  // create shared tile
  bool is_shared = dynamic_cast<ir::copy_to_shared_inst*>(v);
  if(is_shared){
    size_t offset = alloc_->get_offset(v);
    Value *ptr = builder.CreateGEP(sh_mem_ptr, builder.getInt32(offset));
    ptr = builder.CreateBitCast(ptr, ty->getPointerTo(ptr->getType()->getPointerAddressSpace()));
    tmap_.insert({v, new shared_tile(ty, shapes, ptr, builder)});
  }
  // create distributed tile
  else {
    const auto &shapes = v->get_type()->get_tile_shapes();
    std::vector<distributed_axis> axes(shapes.size());
    for(size_t d = 0; d < shapes.size(); d++){
      if(shapes[d] > 1){
        unsigned *x = params_->get_param(v, "p0.d" + std::to_string(d));
        axes[d] = axes_.at(x);
      }
      else{
        axes[d].contiguous = 1;
        axes[d].values = {builder.getInt32(0)};
      }
    }
    bool vectorize = dynamic_cast<ir::load_inst*>(v);
    distributed_tile *T = new distributed_tile(ty, shapes, axes, builder, vectorize);
    tmap_.insert({v, T});
    // constant range
    if(dynamic_cast<ir::constant*>(v)){
      T->for_each([&](indices_t idx){
        assert(idx.size() == 1);
        T->set_value(idx, idx[0]);
      });
    }

  }
}

void selection::init_grids(ir::function *fn, IRBuilder<> &builder, Value *sh_mem_ptr){
  // fetch linear ID
  Module *mod = builder.GetInsertBlock()->getParent()->getParent();
  Function *get_thread_id = Intrinsic::getDeclaration(mod, Intrinsic::nvvm_read_ptx_sreg_tid_x);
  Value *warp_size = builder.getInt32(32);
  Value *u_thread_id = builder.CreateCall(get_thread_id, {});
  Value *u_thread_warp_id = builder.CreateURem(u_thread_id, warp_size);
  Value *u_warp_id = builder.CreateUDiv(u_thread_id, warp_size);
  // create grid
  std::vector<ir::value*> grids;
  std::map<unsigned*, ir::value*> references;
  create_grids(grids, references, fn);
  for(ir::value* i: grids)
    init_axes(i, builder, u_thread_warp_id, u_warp_id);
  // create tile
  std::set<ir::value*> seen;
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list()){
    if(!i->get_type()->is_tile_ty())
      continue;
    create_tile(i, builder, references, seen, sh_mem_ptr);
  }
}


void selection::lower_tile_instruction(ir::instruction *ins, llvm::IRBuilder<> &builder) {
  Module *module = builder.GetInsertBlock()->getModule();
  LLVMContext &ctx = builder.getContext();
  // store
  if(auto *x = dynamic_cast<ir::store_inst*>(ins)) {
    distributed_tile* ptr = (distributed_tile*)tmap_.at(x->get_pointer_operand());
    tile *value = tmap_.at(x->get_value_operand());
    ptr->for_each([&](indices_t idx){
      builder.CreateStore(value->get_value(idx), ptr->get_value(idx));
    });
  }
  else {
    tile *ti = tmap_[ins];
    distributed_tile* result = (distributed_tile*)ti;
    if(!ins->get_type()->is_tile_ty())
      return;
    const auto& shapes = ins->get_type()->get_tile_shapes();
    // global_range
    if(auto *x = dynamic_cast<ir::get_global_range_inst*>(ins)) {
      static std::array<Intrinsic::ID, 3> ctaid = {
        Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
        Intrinsic::nvvm_read_ptx_sreg_ctaid_y,
        Intrinsic::nvvm_read_ptx_sreg_ctaid_z
      };
      Function *get_group_id = Intrinsic::getDeclaration(module, ctaid[x->get_axis()]);
      Value *group_id = builder.CreateCall(get_group_id, {});
      Value *offset = builder.CreateMul(builder.getInt32(shapes[0]), group_id);
      result->for_each([&](indices_t idx){
        BinaryOperator *bin = static_cast<BinaryOperator*>(idx[0]);
        result->set_value(idx, builder.CreateAdd(bin, offset));
      });
    }
    // reshape
    else if(dynamic_cast<ir::reshape_inst*>(ins)) {
      ir::value* in = ins->get_operand(0);
      distributed_tile *in_tile = (distributed_tile*)tmap_.at(in);
      result->for_each([&](indices_t out_idx){
        indices_t in_idx;
        for(size_t k = 0; k < shapes.size(); k++){
          if(shapes[k] > 1)
            in_idx.push_back(out_idx[k]);
        }
        result->set_value(out_idx, in_tile->get_value(in_idx));
      });
    }
    // splat
    else if(dynamic_cast<ir::splat_inst*>(ins)) {
      result->for_each([&](indices_t idx) {
        result->set_value(idx, llvm_value(ins->get_operand(0), builder));
      });
    }
    // broadcast
    else if(dynamic_cast<ir::broadcast_inst*>(ins)) {
      ir::value* in = ins->get_operand(0);
      const auto& in_shapes = in->get_type()->get_tile_shapes();
      distributed_tile *in_tile = (distributed_tile*)tmap_.at(in);
      result->for_each([&](indices_t out_idx){
        indices_t in_idx = out_idx;
        for(size_t k = 0; k < in_idx.size(); k++){
          if(in_shapes[k] == 1)
            in_idx[k] = builder.getInt32(0);
        }
        result->set_value(out_idx, in_tile->get_value(in_idx));
      });
    }
    // copy to shared
    else if(dynamic_cast<ir::copy_to_shared_inst*>(ins)) {
      distributed_tile* in = (distributed_tile*)tmap_.at(ins->get_operand(0));
      in->for_each([&](indices_t idx){
        ti->set_value(idx, in->get_value(idx));
      });
    }
    // matrix multiplication
    else if(dynamic_cast<ir::matmul_inst*>(ins)) {
      ir::value *A = ins->get_operand(0);
      ir::value *B = ins->get_operand(1);
      ir::value *C = ins->get_operand(2);
      Function *f_mul_add = Intrinsic::getDeclaration(module, Intrinsic::fmuladd, {llvm_type(C->get_type()->get_scalar_ty(), ctx)});
      result->for_each([&](indices_t idx){
        Value *res = tmap_.at(C)->get_value(idx);
        unsigned NK = A->get_type()->get_tile_shapes()[1];
        for(unsigned K = 0; K < NK; ++K){
          indices_t a_idx = {idx[0], builder.getInt32(K)};
          indices_t b_idx = {idx[1], builder.getInt32(K)};
          Value *a = tmap_.at(A)->get_value(a_idx);
          Value *b = tmap_.at(B)->get_value(b_idx);
          res = builder.CreateCall(f_mul_add, {a, b, res});
        }
        result->set_value(idx, res);
      });
    }
    // element-wise
    else {
      result->for_each([&](indices_t idx){
        auto value = [&](ir::value *x) {
          if(x->get_type()->is_tile_ty())
            return tmap_.at(x)->get_value(idx);
          else
            return llvm_value(x, builder);
        };
        result->set_value(idx, llvm_inst(ins, value, builder));
      });
    }
  }


}

void selection::lower_instruction(ir::instruction *src, IRBuilder<> &builder) {
  if(src->has_tile_result_or_op()) {
    lower_tile_instruction(src, builder);
  }
  else {
    Instruction *i = (Instruction*)llvm_value(src, builder);
    vmap_[src] = i;
  }
}

void selection::run(ir::module &src, Module &dst){
  vmap_.clear();
  LLVMContext &dst_ctx = dst.getContext();
  IRBuilder<> dst_builder(dst_ctx);

  // iterate over functions
  for(ir::function *fn: src.get_function_list()) {
    // create LLVM function
    FunctionType *fn_ty = (FunctionType*)llvm_type(fn->get_fn_type(), dst_ctx);
    Function *dst_fn = Function::Create(fn_ty, Function::ExternalLinkage, fn->get_name(), &dst);
    // Set metadata
    llvm::Metadata *md_args[] = {
      llvm::ValueAsMetadata::get(dst_fn),
      llvm::MDString::get(dst_ctx, "kernel"),
      llvm::ValueAsMetadata::get(dst_builder.getInt32(1))
    };
    dst.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(llvm::MDNode::get(dst_ctx, md_args));

    // map parameters
    for(unsigned i = 0; i < fn->args().size(); i++)
      vmap_[fn->args()[i]] = &*(dst_fn->arg_begin() + i);
    // create blocks
    for(ir::basic_block *block: fn->blocks()) {
      BasicBlock *dst_block = BasicBlock::Create(dst_ctx, block->get_name(), dst_fn);
      vmap_[block] = dst_block;
    }
    dst_builder.SetInsertPoint((BasicBlock*)vmap_[fn->blocks()[0]]);
    // allocate shared memory
    Value *sh_mem_ptr = nullptr;
    if(unsigned alloc_size = alloc_->get_allocated_size()){
      Type *int_8_ty = Type::getInt8Ty(dst_ctx);
      ArrayType *array_ty = ArrayType::get(int_8_ty, alloc_size);
      Type *ptr_ty = PointerType::get(int_8_ty, 3);
      GlobalVariable *sh_mem_array =
        new GlobalVariable(*dst_fn->getParent(), array_ty, false, GlobalVariable::InternalLinkage,
                           nullptr, "__shared_ptr", nullptr, GlobalVariable::NotThreadLocal, 3);
      sh_mem_ptr = dst_builder.CreateBitCast(sh_mem_array, ptr_ty);
    }
    // create grids
    init_grids(fn, dst_builder, sh_mem_ptr);
    // iterate through block
    for(ir::basic_block *block: fn->blocks()) {
      dst_builder.SetInsertPoint((BasicBlock*)vmap_[block]);
      for(ir::instruction *i: block->get_inst_list())
        lower_instruction(i, dst_builder);
    }
    // add phi operands
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *inst: block->get_inst_list())
    if(auto *phi = dynamic_cast<ir::phi_node*>(inst)){
      for(unsigned n = 0; n < phi->get_num_incoming(); n++){
        ir::value *inc_val = phi->get_incoming_value(n);
        ir::basic_block *inc_block = phi->get_incoming_block(n);
        BasicBlock *llvm_inc_block = (BasicBlock*)vmap_[inc_block];
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
}
