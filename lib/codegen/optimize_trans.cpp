#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/codegen/optimize_trans.h"

namespace triton {
namespace codegen{


ir::value* optimize_trans::replace_phi(ir::value* value,
                                       ir::builder& builder,
                                       const std::vector<ir::constant_int*> &perm){
  if(auto phi = dynamic_cast<ir::phi_node*>(value)) {
    // transpose operands
    std::vector<ir::value*> incs;
    for(unsigned n = 0; n < phi->get_num_incoming(); n++)
      incs.push_back(replace_phi(phi->get_incoming_value(n), builder, perm));
    // create phi for transposed values
    builder.set_insert_point(phi);
    ir::phi_node* result = builder.create_phi(incs[0]->get_type(), incs.size(), phi->get_name());
    for(unsigned n = 0; n < phi->get_num_incoming(); n++)
      result->add_incoming(incs[n], phi->get_incoming_block(n));
    phi->replace_all_uses_with(result);
    return result;
  }
  else if(auto i = dynamic_cast<ir::instruction*>(value)){
    ir::basic_block* block = i->get_parent();
    auto it = std::find(block->begin(), block->end(), i);
    it++;
    builder.set_insert_point(it);
    ir::instruction *trans = (ir::instruction*)builder.create_trans(i, perm);
    i->replace_all_uses_with(trans);
    trans->set_operand(0, i);
    return trans;
  }
  throw std::runtime_error("cannot transpose phi");
}


void optimize_trans::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();
  // iterate
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction* i: block->get_inst_list()){
    // transposition
    if(auto trans = dynamic_cast<ir::trans_inst*>(i)) {
      auto users = trans->get_users();
      auto ops = trans->ops();
      if(users.size() > 1 || ops.size() > 1)
        continue;
      ir::value* op = *ops.begin();
      // todo: chains of transpositions
      // trans(phi) -> phi(trans(), trans()...)
      if(dynamic_cast<ir::phi_node*>(op)){
        ir::value* new_phi = replace_phi(op, builder, trans->get_perm());
        trans->replace_all_uses_with(new_phi);
      }
    }
    // reductions
    if(auto x = dynamic_cast<ir::reduce_inst*>(i)) {
      ir::constant_int *one = ir::constant_int::get(ir::type::get_int32_ty(i->get_type()->get_context()), 1);
      ir::value *arg = x->get_operand(0);
      auto shapes = arg->get_type()->get_tile_shapes();
      if(shapes[x->get_axis()] == one){
        builder.set_insert_point(x);
        ir::value* new_red = builder.create_reshape(arg, x->get_type()->get_tile_shapes());
        x->replace_all_uses_with(new_red);
      }
    }

  }
}

}
}
