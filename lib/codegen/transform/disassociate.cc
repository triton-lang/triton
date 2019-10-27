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
                          std::map<int, std::set<ir::user*>>& result,
                          int depth,
                          std::set<ir::value*>& seen) {
  if(!seen.insert(root).second)
    return;
  result[depth].insert(root);
  if(dynamic_cast<ir::make_range*>(root) ||
     dynamic_cast<ir::splat_inst*>(root)){
    return;
  }
  for(ir::value *op: root->ops()){
    ir::user *u = dynamic_cast<ir::user*>(op);
    if(!u)
      continue;
    extract_retile_chain(u, result, depth + 1, seen);
  }
}

void disassociate::run(ir::module &mod) {
  ir::builder &bld = mod.get_builder();

  std::map<ir::user*, std::map<int, std::set<ir::user*>>> clone_info;
  ir::for_each_instruction(mod, [&](ir::instruction *i){
    if(dynamic_cast<ir::reshape_inst*>(i)){
      std::map<int, std::set<ir::user*>> chains;
      std::set<ir::value*> seen;
      if(!dynamic_cast<ir::user*>(i->get_operand(0)))
        return;
      extract_retile_chain(i, chains, 0, seen);
      if(chains.size())
        clone_info[i] = chains;
    }
  });

  for(const auto& x: clone_info){
    int depth = 1;
    std::map<ir::instruction*, ir::instruction*> clone_map;
    while(x.second.find(depth) != x.second.end()){
      // clone all users
      const auto& remat = x.second.at(depth);
      for(ir::user* u: remat){
        ir::instruction *y = (ir::instruction*)u;
        ir::instruction *cloned = y->clone();
        bld.set_insert_point(y);
        bld.insert(cloned);
        clone_map[y] = cloned;
        // replace in above level
        if(depth > 1){
          for(ir::user* ux: x.second.at(depth - 1))
            clone_map.at((ir::instruction*)ux)->replace_uses_of_with(y, cloned);
        }
        else{
          x.first->replace_uses_of_with(y, cloned);
        }
      }
      depth += 1;
    }
  }


}


}
}
}
