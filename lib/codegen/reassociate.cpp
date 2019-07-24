#include <algorithm>
#include "triton/codegen/reassociate.h"
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


// reassociate pointer
// pz = py + a = (px + (cst + b)) + a -> (px + b) + (cst + a)
ir::value *reassociate::reassociate_ptr(ir::getelementptr_inst* pz,
                                        ir::builder &builder,
                                        std::map<ir::value*, cst_info> &info) {
  ir::value *a = *pz->idx_begin();
  ir::value *vpy = pz->get_pointer_operand();
  if(info.find(vpy) == info.end())
    return nullptr;
  ir::getelementptr_inst *py = (ir::getelementptr_inst*)vpy;
  ir::value *px = py->get_pointer_operand();
  ir::value *cst = info.at(py).sta;
  ir::value *b = info.at(py).dyn;
  ir::value *new_py = builder.create_gep(px, {b});
  ir::value *new_a = builder.create_add(cst, a);
  ir::value *new_pz = builder.create_gep(new_py, {new_a});
  params_->copy(new_pz, pz);
  params_->copy(new_py, vpy);
  params_->copy(new_a, a);
  pz->replace_all_uses_with(new_pz);
  return pz;
}

ir::value *reassociate::reassociate_idx(ir::value *old_value,
                                              ir::builder &builder,
                                              std::vector<ir::instruction*>& to_delete,
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
    ir::value *new_arg = reassociate_idx(old_arg, builder, to_delete, noncst, cst);
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
        to_delete.push_back(op);
      }
    }
  }

  // handle binary addition
  if(ir::instruction* op = is_bin_add(old_value)){
    builder.set_insert_point(op);
    std::string name = op->get_name();
    ir::value *lhs = reassociate_idx(op->get_operand (0), builder, to_delete, noncst, cst);
    ir::value *rhs = reassociate_idx(op->get_operand(1), builder, to_delete, noncst, cst);
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
      if(new_value != old_value){
        to_delete.push_back(bin_lhs);
      }
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
      if(new_value != op)
        to_delete.push_back(bin_rhs);
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
    if(auto *x = dynamic_cast<ir::instruction*>(old_value))
      to_delete.push_back(x);
  }

  return new_value;
}

reassociate::reassociate(tune* params)
  : params_(params)
{ }


/* run */
void reassociate::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();
  std::vector<ir::instruction*> to_delete;

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

      // pz = py + offset
      // tries to achieve pz = py + (cst + a)
      // by modifying py and/or offset
      ir::value* py = pz->get_pointer_operand();
      ir::value* offset = *pz->idx_begin();

      // reassociate index
      ir::value *sta = nullptr;
      ir::value *dyn = offset;
      reassociate_idx(pz, builder, to_delete, dyn, sta);
      if(sta){
        infos[pz] = {sta, dyn};
        re_ordered[block].insert(pz);
      }

//      // reassociate pointer
//      reassociate_ptr(pz, builder, offsets);

//      // reassociate phi-node
//      if(ir::phi_node* phi = dynamic_cast<ir::phi_node*>(py)){
//        // only optimize the case where py = phi pa, pz
//        std::vector<ir::value*> ops = phi->ops();
//        if(!(ops.size() == 2 && (ops[0] == pz || ops[1] == pz)))
//           continue;
//        size_t idx_z = (ops[0] == pz) ? 0 : 1;
//        size_t idx_a = (idx_z + 1) % 2;
//        ir::value *vpa = phi->get_incoming_value(idx_a);
//        ir::value *block_a = phi->get_incoming_block(idx_a);
//        ir::value *block_z = phi->get_incoming_value(idx_z);
//        auto it = infos.find(vpa);
//        if(it == infos.end())
//          continue;
//        ir::value *b = it->a;
//        // pa = px + (cst + b)
//        ir::getelementptr_inst *pa = (ir::getelementptr_inst*)vpa;
//        ir::getelementptr_inst *px = pa->get_pointer_operand();
//        // new_pa = px + b
//        ir::getelementptr_inst *new_pa = builder.create_gep(px, {b});
//        // new_pz = py + (offset + a)
//        ir::getelementptr_inst *new_offset = builder.create_add(it->cst, dyn);
//        ir::getelementptr_inst *new_pz = builder.create_gep(pz->get_pointer_operand(), {new_offset});
//      }
    }
    }
    }
  }
  // erase dead code
  for(ir::instruction* i: to_delete)
    i->erase_from_parent();
}

}
}
