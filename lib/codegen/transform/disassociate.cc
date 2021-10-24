#include "triton/codegen/transform/disassociate.h"
#include "triton/ir/utils.h"
#include "triton/ir/instructions.h"
#include "triton/ir/builder.h"
#include "triton/ir/module.h"
#include <iostream>

namespace triton {
namespace codegen{
namespace transform{

ir::instruction* rematerialize(ir::builder& bld, ir::instruction *root,
                          std::set<ir::value*>& seen) {
  if (dynamic_cast<ir::phi_node*>(root))
    return root;
  if(!seen.insert(root).second)
    return root;
  if(!root->get_type()->is_block_ty())
    return root;

  bld.set_insert_point(root);
  ir::instruction *new_root = bld.insert(root->clone());
  for(ir::value *op: root->ops()){
    ir::instruction *i = dynamic_cast<ir::instruction*>(op);
    if(!i || i->get_id() == ir::INST_REDUCE)
      continue;
    ir::instruction* new_op = rematerialize(bld, i, seen);
    new_root->replace_uses_of_with(op, new_op);
  }
  return new_root;
}

void disassociate::run(ir::module &mod) {
  ir::builder &bld = mod.get_builder();

//  ir::for_each_instruction(mod, [&](ir::instruction *i){
//    bld.set_insert_point(i);
//    for(ir::value* op: i->ops()){
//      auto reshape = dynamic_cast<ir::make_range*>(op);
//      if(!reshape)
//        continue;
//      ir::instruction* new_op = bld.insert(reshape->clone());
//      i->replace_uses_of_with(op, new_op);
//    }
//  });


  ir::for_each_instruction(mod, [&](ir::instruction *i){
    if(dynamic_cast<ir::reshape_inst*>(i) || dynamic_cast<ir::splat_inst*>(i)){
      std::set<ir::value*> seen;
      ir::instruction* new_i = rematerialize(bld, i, seen);
      i->replace_all_uses_with(new_i);
    }
  });


}


}
}
}
