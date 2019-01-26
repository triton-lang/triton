#include "codegen/selection.h"
#include "codegen/tune.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "ir/context.h"
#include "ir/module.h"
#include "ir/function.h"
#include "ir/type.h"


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
Instruction *selection::llvm_inst(ir::instruction *inst, LLVMContext & ctx) {
  auto value = [&](ir::value *x) { return llvm_value(x, ctx); };
  auto block = [&](ir::basic_block *x) { return bmap_.at(x); };
  auto type = [&](ir::type *x) { return llvm_type(x, ctx); };
  if(auto* ii = dynamic_cast<ir::cond_branch_inst*>(inst)){
    BasicBlock *true_dest  = block(ii->get_true_dest());
    BasicBlock *false_dest = block(ii->get_false_dest());
    Value *cond = value(ii->get_cond());
    return BranchInst::Create(true_dest, false_dest, cond);
  }
  if(auto* ii = dynamic_cast<ir::uncond_branch_inst*>(inst)){
    BasicBlock *dest = block(ii->get_dest());
    return BranchInst::Create(dest);
  }
  if(auto* ii = dynamic_cast<ir::phi_node*>(inst)){
    Type *ty = type(ii->get_type());
    unsigned num_ops = ii->get_num_operands();
    return PHINode::Create(ty, num_ops, ii->get_name());
  }
  if(auto* ii = dynamic_cast<ir::return_inst*>(inst)){
    ir::value *ret_val = ii->get_return_value();
    return ReturnInst::Create(ctx, ret_val?value(ret_val):nullptr);
  }
  if(auto* ii = dynamic_cast<ir::binary_operator*>(inst)){
    Value *lhs = value(ii->get_operand(0));
    Value *rhs = value(ii->get_operand(1));
    return BinaryOperator::Create(ii->get_op(), lhs, rhs, ii->get_name());
  }
  if(auto* ii = dynamic_cast<ir::icmp_inst*>(inst)){
    CmpInst::Predicate pred = ii->get_pred();
    Value *lhs = value(ii->get_operand(0));
    Value *rhs = value(ii->get_operand(1));
    return CmpInst::Create(Instruction::ICmp, pred, lhs, rhs, ii->get_name());
  }
  if(auto* ii = dynamic_cast<ir::fcmp_inst*>(inst)){
    CmpInst::Predicate pred = ii->get_pred();
    Value *lhs = value(ii->get_operand(0));
    Value *rhs = value(ii->get_operand(1));
    return FCmpInst::Create(Instruction::FCmp, pred, lhs, rhs, ii->get_name());
  }
  if(auto* ii = dynamic_cast<ir::cast_inst*>(inst)){
    Value *arg = value(ii->get_operand(0));
    Type *dst_ty = type(ii->get_type());
    return CastInst::Create(ii->get_op(), arg, dst_ty, ii->get_name());
  }
  if(auto* ii = dynamic_cast<ir::getelementptr_inst*>(inst)){
    std::vector<Value*> idx_vals;
    std::transform(ii->idx_begin(), ii->idx_end(), std::back_inserter(idx_vals),
                   [&value](ir::value* x){ return value(x);});
    Type *source_ty = type(ii->get_source_elt_ty());
    Value *arg = value(ii->get_operand(0));
    return GetElementPtrInst::Create(source_ty, arg, idx_vals, ii->get_name());
  }
  if(ir::load_inst* ii = dynamic_cast<ir::load_inst*>(inst)){
    Value *ptr = value(ii->get_pointer_operand());
    return new LoadInst(ptr, ii->get_name());
  }
  // unknown instruction
  throw std::runtime_error("unknown conversion from ir::type to Type");
}

/* convert ir::value to llvm::Value */
Value* selection::llvm_value(ir::value *v, LLVMContext &ctx) {
  assert(!v->get_type()->is_tile_ty());
  if(vmap_.find(v) != vmap_.end())
    return vmap_.at(v);
  // create operands
  if(auto *uu = dynamic_cast<ir::user*>(v))
    for(ir::value* u: uu->ops())
      vmap_[u] = llvm_value(u, ctx);
  if(auto *cc = dynamic_cast<ir::constant*>(v))
    return llvm_constant(cc, ctx);
  // instruction
  if(auto *ii = dynamic_cast<ir::instruction*>(v))
    return llvm_inst(ii, ctx);
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

void selection::init_axes(ir::instruction *instr, IRBuilder<> &builder, Value *u_thread_id, Value *u_warp_id) {
  const auto& shapes = instr->get_type()->get_tile_shapes();
  size_t dim = shapes.size();
  std::vector<unsigned> contiguous(dim);
  std::vector<unsigned> warp_size(dim);
  std::vector<unsigned> n_warps(dim);
  for(unsigned i = 0; i < shapes.size(); i++){
    std::string str_i = std::to_string(i);
    contiguous[i] = *params_->get_param(instr, "p0.d" + str_i);
    warp_size[i] = *params_->get_param(instr, "p1.d" + str_i);
    n_warps[i] = *params_->get_param(instr, "p2.d" + str_i);
  }
  std::vector<Value*> thread_id_in_warp = delinearize(u_thread_id, warp_size, builder);
  std::vector<Value*> warp_id = delinearize(u_warp_id, n_warps, builder);
  // Create axes
  std::vector<distributed_axis> axes(dim);
  for(unsigned k = 0; k < dim; k++) {
    Value *warp_size_k = builder.getInt32(warp_size[k]);
    Value *contiguous_k = builder.getInt32(contiguous[k]);
    Value *thread_id   = builder.CreateAdd(thread_id_in_warp[k], builder.CreateMul(warp_id[k], warp_size_k));
    thread_id = builder.CreateMul(thread_id, contiguous_k);
    unsigned per_block = contiguous[k] * warp_size[k] * n_warps[k];
    unsigned per_thread = contiguous[k] * shapes[k] / per_block;
    std::vector<Value*> idx_list(per_thread);
    for(unsigned n = 0 ; n < per_thread; n++){
      unsigned offset = n / contiguous[k] * per_block + n % contiguous[k];
      idx_list[n] = builder.CreateAdd(thread_id, builder.getInt32(offset));
    }
    axes[k] = distributed_axis{idx_list};
  }
  // Store axes
  axes_[instr] = axes;
}

void selection::create_grids(std::vector<ir::instruction*> &grids,
                             std::map<unsigned*, ir::instruction*> &references,
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
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list()){
    if(!i->get_type()->is_tile_ty())
      continue;
    const auto& shapes = i->get_type()->get_tile_shapes();
    bool is_shared = dynamic_cast<ir::copy_to_shared_inst*>(i);
    if(is_shared)
      continue;
    for(size_t d = 0; d < shapes.size(); d++){
      if(shapes[d] == 1)
        continue;
      unsigned *x = params_->get_param(i, "p0.d" + std::to_string(d));
      ir::instruction *&r = references[x];
      if(!r || get_tile_gt1_dim(i) > get_tile_gt1_dim(r))
        r = i;
    }
  }
  // create grid
  for(auto &ref: references)
    if(std::find(grids.begin(), grids.end(), ref.second) == grids.end())
      grids.push_back(ref.second);
}

void selection::init_grids(ir::function *fn, IRBuilder<> &builder){
  // fetch linear ID
  Module *mod = builder.GetInsertBlock()->getParent()->getParent();
  LLVMContext &ctx = builder.getContext();
  Function *get_thread_id = Intrinsic::getDeclaration(mod, Intrinsic::nvvm_read_ptx_sreg_tid_x);
  Value *warp_size = builder.getInt32(32);
  Value *u_thread_id = builder.CreateCall(get_thread_id, {});
  Value *u_thread_warp_id = builder.CreateURem(u_thread_id, warp_size);
  Value *u_warp_id = builder.CreateUDiv(u_thread_id, warp_size);
  // create grid
  std::vector<ir::instruction*> grids;
  std::map<unsigned*, ir::instruction*> references;
  create_grids(grids, references, fn);
  for(ir::instruction* i: grids)
    init_axes(i, builder, u_thread_warp_id, u_warp_id);
  // create tile
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list()){
    if(!i->get_type()->is_tile_ty())
      continue;
    bool is_shared = dynamic_cast<ir::copy_to_shared_inst*>(i);
    const auto& shapes = i->get_type()->get_tile_shapes();
    Type* ty = llvm_type(i->get_type(), ctx);
    // create shared tile
    if(is_shared){
      tmap_.insert({i, new shared_tile(ty, shapes)});
    }
    // create distributed tile
    else {
      const auto &shapes = i->get_type()->get_tile_shapes();
      std::vector<distributed_axis> axes(shapes.size());
      for(size_t d = 0; d < shapes.size(); d++){
        if(shapes[d] > 1){
          unsigned *x = params_->get_param(i, "p0.d" + std::to_string(d));
          axes[d] = axes_.at(references.at(x))[d];
        }
        else
          axes[d].values = {builder.getInt32(0)};
      }
      tmap_.insert({i, new distributed_tile(ty, shapes, axes)});
    }
  }
}

void selection::lower_tile_instruction(ir::instruction *src, llvm::IRBuilder<> &builder) {
  std::cout << typeid(*src).name() << std::endl;
}

void selection::lower_instruction(ir::instruction *src, IRBuilder<> &builder) {
  LLVMContext &ctx = builder.getContext();
  if(src->has_tile_result_or_op()) {
    lower_tile_instruction(src, builder);
  }
  else {
    Instruction *i = llvm_inst(src, ctx);
    vmap_[src] = i;
    builder.Insert(i);
  }
}

void selection::run(ir::module &src, Module &dst){
  vmap_.clear();
  bmap_.clear();
  LLVMContext &dst_ctx = dst.getContext();
  IRBuilder<> dst_builder(dst_ctx);
  // iterate over functions
  for(ir::function *fn: src.get_function_list()) {
    // create LLVM function
    FunctionType *fn_ty = (FunctionType*)llvm_type(fn->get_fn_type(), dst_ctx);
    Function *dst_fn = Function::Create(fn_ty, Function::ExternalLinkage, "kernel", &dst);
    // map parameters
    for(unsigned i = 0; i < fn->args().size(); i++)
      vmap_[fn->args()[i]] = &*(dst_fn->arg_begin() + i);
    // create blocks
    for(ir::basic_block *block: fn->blocks()) {
      BasicBlock *dst_block = BasicBlock::Create(dst_ctx, block->get_name(), dst_fn);
      bmap_[block] = dst_block;
    }
    // create grids
    dst_builder.SetInsertPoint(bmap_[fn->blocks()[0]]);
    init_grids(fn, dst_builder);
    // iterate through block
    for(ir::basic_block *block: fn->blocks()) {
      dst_builder.SetInsertPoint(bmap_[block]);
      for(ir::instruction *i: block->get_inst_list())
        lower_instruction(i, dst_builder);
    }
    // add phi operands
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *inst: block->get_inst_list())
    if(auto *phi = dynamic_cast<ir::phi_node*>(inst)){
      PHINode *dst_phi = (PHINode*)vmap_.at(phi);
      for(unsigned i = 0; i < phi->get_num_incoming(); i++){
        ir::value *inc_val = phi->get_incoming_value(i);
        ir::basic_block *inc_block = phi->get_incoming_block(i);
        Value *llvm_inc_val = llvm_value(inc_val, dst_ctx);
        BasicBlock *llvm_block = bmap_[inc_block];
        dst_phi->addIncoming(llvm_inc_val, llvm_block);
      }
    }
  }
}


}
}
