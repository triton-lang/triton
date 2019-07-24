#include <algorithm>
#include "triton/codegen/reassociate.h"
#include "triton/codegen/alignment_info.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/cfg.h"
#include "triton/codegen/tune.h"

namespace triton {
namespace codegen{

//inline Constant *get_gep_cst_offset(GetElementPtrInst *gep){
//  std::vector<Value*> idx_vals;
//  std::transform(gep->idx_begin(), gep->idx_end(),
//                 std::back_inserter(idx_vals),
//                 [](Value* x){ return x;});
//  if(idx_vals.size() > 1)
//    return nullptr;
//  Value *idx = idx_vals[0];
//  if(isa<Constant>(idx))
//    return idx;
//  if(Instruction *BinOp = is_bin_add(idx)){
//    Value *LHS = BinOp->getOperand(0);
//    Value *RHS = BinOp->getOperand(1);
//    if(Constant* Res = dyn_cast<Constant>(LHS))
//      return Res;
//    if(Constant* Res = dyn_cast<Constant>(RHS))
//      return Res;
//  }
//  return nullptr;
//}


inline ir::instruction* reassociate::is_bin_add(ir::value *x) {
  ir::binary_operator *bin_op = dynamic_cast<ir::binary_operator*>(x);
  bool is_bin_add = bin_op && bin_op->get_op()==llvm::Instruction::Add;
  if(is_bin_add)
    return (ir::instruction*)x;
  return nullptr;
}

inline bool is_cst(ir::value *x) {
  if(dynamic_cast<ir::constant*>(x))
    return true;
  if(auto *v = dynamic_cast<ir::retile_inst*>(x))
    return is_cst(v->get_operand(0));
  return false;
}

ir::value *reassociate::reassociate_idx(ir::value *old_value,
                                              ir::builder &builder,
                                              ir::value *&noncst,
                                              ir::value *&cst){
  // value doesn't change by default
  ir::value* new_value = old_value;
  cst = nullptr;
  noncst = old_value;

  // handle retiling
  if(ir::instruction* op = dynamic_cast<ir::retile_inst*>(old_value)){
    auto shapes = op->get_type()->get_tile_shapes();
    ir::value *old_arg = op->get_operand(0);
    ir::value *new_arg = reassociate_idx(old_arg, builder, noncst, cst);
    // retile(x + y) = retile(x) + retile(y)
    if(ir::instruction* bin_add = is_bin_add(new_arg))
    if(cst){
      ir::value *old_lhs = bin_add->get_operand(0);
      ir::value *old_rhs = bin_add->get_operand(1);
      ir::value *new_lhs = nullptr;
      ir::value *new_rhs = nullptr;
      if(dynamic_cast<ir::reshape_inst*>(op)){
        builder.set_insert_point(op);
        new_lhs = builder.create_reshape(old_lhs, shapes);
        new_rhs = builder.create_reshape(old_rhs, shapes);
        new_value = builder.create_add(new_lhs, new_rhs, op->get_name());
      }
      if(dynamic_cast<ir::broadcast_inst*>(op)){
        builder.set_insert_point(op);
        new_lhs = builder.create_broadcast(old_lhs, shapes);
        new_rhs = builder.create_broadcast(old_rhs, shapes);
        new_value = builder.create_add(new_lhs, new_rhs, op->get_name());
      }
      if(dynamic_cast<ir::splat_inst*>(op)){
        builder.set_insert_point(op);
        new_lhs = builder.create_splat(old_lhs, shapes);
        new_rhs = builder.create_splat(old_rhs, shapes);
        new_value = builder.create_add(new_lhs, new_rhs, op->get_name());
      }
      if(new_value != old_value){
        params_->copy(new_value, old_value);
        params_->copy(new_lhs, old_value);
        params_->copy(new_rhs, old_value);
      }
    }
  }

  // handle binary addition
  if(ir::instruction* op = is_bin_add(old_value)){
    builder.set_insert_point(op);
    std::string name = op->get_name();
    ir::value *lhs = reassociate_idx(op->get_operand (0), builder, noncst, cst);
    ir::value *rhs = reassociate_idx(op->get_operand(1), builder, noncst, cst);
    builder.set_insert_point(op);
    // (x + y) + z
    if(ir::instruction* bin_lhs = is_bin_add(lhs)){
      ir::value *llhs = bin_lhs->get_operand(0);
      ir::value *rlhs = bin_lhs->get_operand(1);
      // (cst + x) + y -> cst + (x + y)
      if(is_cst(llhs))
        new_value = builder.create_add(llhs, builder.create_add(rlhs, rhs), name);
      // (x + cst) + y -> cst + (x + y)
      if(is_cst(rlhs))
        new_value = builder.create_add(rlhs, builder.create_add(llhs, rhs), name);
    }
    // x + (y + z)
    if(ir::instruction* bin_rhs = is_bin_add(rhs)){
      ir::value *lrhs = bin_rhs->get_operand(0);
      ir::value *rrhs = bin_rhs->get_operand(1);
      // x + (cst + y) -> cst + (x + y)
      if(is_cst(lrhs))
        new_value = builder.create_add(lrhs, builder.create_add(rrhs, lhs), name, cst);
      // x + (y + cst) -> cst + (x + y)
      if(is_cst(rrhs))
        new_value = builder.create_add(rrhs, builder.create_add(lrhs, lhs), name, cst);
    }
    if(new_value != old_value){
      params_->copy(new_value, old_value);
      params_->copy(((ir::instruction*)new_value)->get_operand(0), old_value);
      params_->copy(((ir::instruction*)new_value)->get_operand(1), old_value);
    }
  }

  // extract constant and non-constant
  if(ir::instruction *bin_add = is_bin_add(new_value)){
    ir::value *new_lhs = bin_add->get_operand(0);
    ir::value *new_rhs = bin_add->get_operand(1);
    if(is_cst(new_lhs)){
      cst = new_lhs;
      noncst = new_rhs;
    }
    if(is_cst(new_rhs)){
      cst = new_rhs;
      noncst = new_lhs;
    }
  }

  // clean-up if some re-ordering happened
  if(old_value != new_value){
    old_value->replace_all_uses_with(new_value);
  }

  return new_value;
}

reassociate::reassociate(tune* params, alignment_info* align)
  : params_(params), align_(align)
{ }


/* run */
void reassociate::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();

  // constant_range -> nv_dynamic_range_idx + nv_static_range_idx
  for(ir::function *fn: mod.get_function_list()){
    std::vector<ir::constant_range*> ranges;
    std::vector<ir::basic_block*> rpo = ir::cfg::reverse_post_order(fn);
    for(ir::basic_block *block: rpo){
      // iterate through instruction
      for(ir::instruction *i: block->get_inst_list())
      for(ir::value* op: i->ops())
      if(auto *range = dynamic_cast<ir::constant_range*>(op))
        ranges.push_back(range);
    }

    builder.set_insert_point(rpo.front()->get_first_non_phi());
    for(ir::constant_range* old_range: ranges){
      ir::value* dyn_range = builder.insert(ir::nv_dynamic_range_idx_inst::create(old_range->get_type()));
      ir::value* static_range = ir::nv_static_range_idx::get(old_range);
      ir::value* new_range = builder.create_add(dyn_range, static_range);
      old_range->replace_all_uses_with(new_range);
      params_->copy(dyn_range, old_range);
      params_->copy(static_range, old_range);
      params_->copy(new_range, old_range);
    }
  }

  // reassociate
  std::map<ir::value*, cst_info> infos;
  std::map<ir::basic_block*, std::set<ir::value*>> re_ordered;

  for(ir::function *fn: mod.get_function_list()){
    std::vector<ir::basic_block*> rpo = ir::cfg::reverse_post_order(fn);
    // iterate through blocks
    for(ir::basic_block *block: rpo){
    // iterate through instruction
    for(ir::instruction *i: block->get_inst_list()){
    // getelementptr instruction
    if(ir::getelementptr_inst *pz = dynamic_cast<ir::getelementptr_inst*>(i)){
      // unpack GEP instruction
      ir::value* py = pz->get_pointer_operand();
      ir::value* offset = *pz->idx_begin();
      // reassociate index
      ir::value *sta = nullptr;
      ir::value *dyn = offset;
      reassociate_idx(offset, builder, dyn, sta);
      if(sta){
        builder.set_insert_point(pz);
        ir::value *dyn_ptr = builder.create_gep(py, {dyn});
        ir::value *sta_ptr = builder.create_gep(dyn_ptr, {sta});
        params_->copy(dyn_ptr, pz);
        params_->copy(sta_ptr, pz);
        align_->copy(sta_ptr, pz);
        pz->replace_all_uses_with(sta_ptr);
        infos[sta_ptr].dyn_ptr = (ir::getelementptr_inst*)dyn_ptr;
        infos[sta_ptr].sta_ptr = (ir::getelementptr_inst*)sta_ptr;
      }
      // reassociate phi-node pointer
      if(ir::phi_node* phi = dynamic_cast<ir::phi_node*>(py)){
        // only optimize the case where py = phi pa, pz for now
        std::vector<ir::value*> ops = phi->ops();
        if(ops.size() != 2)
          continue;
        if(ops[0] != pz && ops[1] != pz)
           continue;
        // grab  incoming
        size_t idx_z = (ops[0] == pz) ? 0 : 1;
        size_t idx_a = (ops[0] == pz) ? 1 : 0;
        // check if pa is known to have constant offset
        ir::value *vpa = phi->get_incoming_value(idx_a);
        auto it = infos.find(vpa);
        if(it == infos.end())
          continue;
        ir::getelementptr_inst *pa = (ir::getelementptr_inst*)vpa;
        // unpack dynamically/statically offset pointer
        ir::getelementptr_inst *dyn_ptr = it->second.dyn_ptr;
        ir::getelementptr_inst *sta_ptr = it->second.sta_ptr;
        // we take static offset out of the phi function
        builder.set_insert_point(phi);
        ir::phi_node *new_phi = builder.create_phi(phi->get_type(), 2);
        // new pz for phi has the same offsets
        builder.set_insert_point(pz);
        std::vector<ir::value*> idxs(pz->idx_begin(), pz->idx_end());
        ir::value *new_phi_pz = builder.create_gep(new_phi, idxs);
        // fold the static offset into the new pz value
        ir::value *new_pz = builder.create_gep(new_phi_pz, {*sta_ptr->idx_begin()});
        // populate incoming values
        new_phi->add_incoming(dyn_ptr, phi->get_incoming_block(idx_a));
        new_phi->add_incoming(new_phi_pz, phi->get_incoming_block(idx_z));
        // replace phi uses
        phi->replace_all_uses_with(new_phi);
        // replace pz uses
        pz->replace_all_uses_with(new_pz);
        // copy params
        params_->copy(new_phi_pz, pz);
        params_->copy(new_phi, phi);
        params_->copy(new_pz, pz);
        align_->copy(new_pz, pz);
      }

//      // reassociate pointer
//      reassociate_ptr(pz, builder, offsets);


    }
    }
    }
  }
}

}
}
