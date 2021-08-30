#include "triton/codegen/transform/disassociate.h"
#include "triton/ir/utils.h"
#include "triton/ir/instructions.h"
#include "triton/ir/builder.h"
#include "triton/ir/module.h"
#include <iostream>

namespace triton {
namespace codegen{
namespace transform{

ir::instruction* extract_retile_chain(ir::builder& bld, ir::instruction *root,
                          std::set<ir::value*>& seen) {
  if(!seen.insert(root).second)
    return root;
  bld.set_insert_point(root);
  ir::instruction *new_root = bld.insert(root->clone());
  if(dynamic_cast<ir::make_range*>(root) ||
     dynamic_cast<ir::splat_inst*>(root)){
    return new_root;
  }
  for(ir::value *op: root->ops()){
    ir::instruction *i = dynamic_cast<ir::instruction*>(op);
    if(!i)
      continue;
    ir::instruction* new_op = extract_retile_chain(bld, i, seen);
    root->replace_uses_of_with(op, new_op);
  }
  return new_root;
}

void disassociate::run(ir::module &mod) {
  ir::builder &bld = mod.get_builder();

  std::map<ir::user*, std::map<int, std::set<ir::user*>>> clone_info;
  ir::for_each_instruction(mod, [&](ir::instruction *i){
    if(dynamic_cast<ir::reshape_inst*>(i)){
      std::set<ir::value*> seen;
      ir::instruction* new_i = extract_retile_chain(bld, i, seen);
      i->replace_all_uses_with(new_i);
    }
  });


}


}
}
}
