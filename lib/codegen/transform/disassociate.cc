#include "triton/codegen/transform/disassociate.h"
#include "triton/ir/utils.h"
#include "triton/ir/instructions.h"
#include "triton/ir/builder.h"
#include "triton/ir/module.h"
#include <iostream>

namespace triton {
namespace codegen{
namespace transform{

void extract_retile_chain(ir::user *root,
                          const std::vector<ir::user*>& current,
                          std::vector<std::vector<ir::user*>>& result,
                          std::set<ir::value*>& seen) {
  if(!seen.insert(root).second)
    return;
  if(dynamic_cast<ir::make_range*>(root) || dynamic_cast<ir::splat_inst*>(root)){
    std::vector<ir::user*> next = current;
    next.push_back(root);
    result.push_back(next);
    return;
  }
  for(ir::value *op: root->ops()){
    ir::user *u = dynamic_cast<ir::user*>(op);
    if(!u)
      continue;
    std::vector<ir::user*> next = current;
    next.push_back(u);
    extract_retile_chain(u, next, result, seen);
  }
}

void disassociate::run(ir::module &mod) {
  ir::builder &bld = mod.get_builder();

  std::map<ir::user*, std::vector<std::vector<ir::user*>>> clone_info;
  ir::for_each_instruction(mod, [&](ir::instruction *i){
    if(dynamic_cast<ir::reshape_inst*>(i)){
      std::vector<std::vector<ir::user*>> chains;
      std::set<ir::value*> seen;
      if(!dynamic_cast<ir::user*>(i->get_operand(0)))
        return;
      extract_retile_chain(i, {}, chains, seen);
      if(chains.size())
        clone_info[i] = chains;
    }
  });


  for(auto x: clone_info){
    for(auto chain: x.second){
      for(int i = 0; i < chain.size(); i++) {
        ir::instruction *y = (ir::instruction*)chain[i];
        ir::instruction *cloned = y->clone();
        bld.set_insert_point(y);
        bld.insert(cloned);
        if(i > 0)
          chain[i-1]->replace_uses_of_with(y, cloned);
        else
          x.first->replace_uses_of_with(y, cloned);
      }


//      ir::instruction *y = (ir::instruction*)parent;
//      for(ir::user *u: chain){
//        ir::instruction *cloned = y->clone();
//        bld.set_insert_point(y);
//        bld.insert(cloned);
//        std::cout << typeid(*u).name() << std::endl;
//        u->replace_uses_of_with(y, cloned);
//        y = (ir::instruction*)u;
//      }
    }
  }


}


}
}
}
