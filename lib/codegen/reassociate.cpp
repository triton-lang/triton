#include <algorithm>
#include "triton/codegen/reassociate.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/cfg.h"

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


inline ir::value *reassociate::reorder_op(ir::value *old_value,
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
    ir::value *new_arg = reorder_op(old_arg, builder, to_delete, noncst, cst);
    // retile(x + y) = retile(x) + retile(y)
    if(ir::instruction* bin_add = is_bin_add(new_arg))
    if(cst){
      ir::value *old_lhs = bin_add->get_operand(0);
      ir::value *old_rhs = bin_add->get_operand(1);
      if(dynamic_cast<ir::reshape_inst*>(op)){
        builder.set_insert_point(op);
        ir::value *new_lhs = builder.create_reshape(old_lhs, shapes);
        ir::value *new_rhs = builder.create_reshape(old_rhs, shapes);
        new_value = builder.create_add(new_lhs, new_rhs, op->get_name());
      }
      if(dynamic_cast<ir::broadcast_inst*>(op)){
        builder.set_insert_point(op);
        ir::value *new_lhs = builder.create_broadcast(old_lhs, shapes);
        ir::value *new_rhs = builder.create_broadcast(old_rhs, shapes);
        new_value = builder.create_add(new_lhs, new_rhs, op->get_name());
      }
      if(dynamic_cast<ir::splat_inst*>(op)){
        builder.set_insert_point(op);
        ir::value *new_lhs = builder.create_splat(old_lhs, shapes);
        ir::value *new_rhs = builder.create_splat(old_rhs, shapes);
        new_value = builder.create_add(new_lhs, new_rhs, op->get_name());
      }
      to_delete.push_back(op);
    }
  }

  // handle binary addition
  if(ir::instruction* op = is_bin_add(old_value)){
    builder.set_insert_point(op);
    std::string name = op->get_name();
    ir::value *lhs = reorder_op(op->get_operand (0), builder, to_delete, noncst, cst);
    ir::value *rhs = reorder_op(op->get_operand(1), builder, to_delete, noncst, cst);
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
      if(new_value != op)
        to_delete.push_back(bin_lhs);
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

reassociate::reassociate() {

}

void reassociate::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();
  std::vector<ir::instruction*> to_delete;
  for(ir::function *fn: mod.get_function_list()){
    std::vector<ir::basic_block*> rpo = ir::cfg::reverse_post_order(fn);
    bool done = false;
    do{
      // iterate through blocks
      for(ir::basic_block *block: rpo){
        // iterate through instruction
        for(ir::instruction *i: block->get_inst_list()){
          if(auto *gep = dynamic_cast<ir::getelementptr_inst*>(i)){
            std::vector<ir::value*> idxs(gep->idx_begin(), gep->idx_end());
            ir::value *cst = nullptr;
            ir::value *noncst = idxs[0];
            reorder_op(noncst, builder, to_delete, noncst, cst);
//            std::cout << gep->get_name() << " " << noncst << " " << cst << std::endl;
          }
        }
        done = true;
      }
    }while(!done);
  }
  // erase dead code
  for(ir::instruction* i: to_delete)
    i->erase_from_parent();
}

}
}
