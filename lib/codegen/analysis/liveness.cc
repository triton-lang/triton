#include <climits>
#include <iostream>
#include "triton/codegen/analysis/liveness.h"
#include "triton/codegen/analysis/layout.h"
#include "triton/ir/function.h"
#include "triton/ir/module.h"
#include "triton/ir/utils.h"

namespace triton{
namespace codegen{
namespace analysis{


void liveness::run(ir::module &mod) {
  intervals_.clear();

  std::map<ir::value*, std::set<shared_layout*>> layouts_map;
  for(auto &x: layouts_->get_all()){
    shared_layout* layout = x.second->to_shared();
    if(!layout || layout->is_tmp())
      continue;
    for(ir::value* v:layout->get_values()){
      layouts_map[v].insert(layout);
    }
  }



  std::map<ir::user*, std::set<shared_layout*>> live_in;
  while(true){
    bool changed = false;
    ir::instruction* last_inst = nullptr;
    ir::for_each_instruction_backward(mod, [&](ir::instruction* i){
      // gen
      std::set<shared_layout*> gen;
      for(ir::value* v: i->ops())
      for(shared_layout* layout: layouts_map[v])
        gen.insert(layout);
      // kill
      std::set<shared_layout*> kill;
      for(shared_layout* layout: layouts_map[i])
        kill.insert(layout);
      // temporaries are handled separately
      if(layouts_->has_tmp(i)){
        gen.insert(layouts_->get(layouts_->tmp(i))->to_shared());
        kill.insert(layouts_->get(layouts_->tmp(i))->to_shared());
      }
      // live-out
      std::set<shared_layout*> live_out;
      std::vector<ir::instruction*> succs = {last_inst};
      if(i == i->get_parent()->get_inst_list().back())
        for(ir::basic_block* succ: i->get_parent()->get_successors())
          succs.push_back(succ->get_inst_list().front());
      for(ir::instruction* succ: succs)
      for(shared_layout* layout: live_in[succ])
      if(!layout->is_tmp())
        live_out.insert(layout);

      // new sets
      std::set<shared_layout*> live_out_minus_kill;
      std::set_difference(live_out.begin(), live_out.end(), kill.begin(), kill.end(), 
                          std::inserter(live_out_minus_kill, live_out_minus_kill.end()));
      std::set<shared_layout*> new_live_in;
      std::set_union(gen.begin(), gen.end(), live_out_minus_kill.begin(), live_out_minus_kill.end(),
                      std::inserter(new_live_in, new_live_in.end()));
      
      changed = changed || (new_live_in != live_in[i]);
      live_in[i] = new_live_in;
      last_inst = i;
    });
    if(!changed)
      break;
  }
    
  // ir::for_each_instruction(mod, [&](ir::instruction* i){
  //   i->print(std::cout);
  //   std::cout << " live_in: " << live_in[i].size() << std::endl;
  // });



  // Assigns index to each instruction
  std::map<ir::value*, slot_index> indices;
  slot_index index = 0;
  ir::for_each_instruction(mod, [&](ir::instruction* instr){
      index += 1;
      indices.insert({instr, index});
  });
  

  for(auto &x: layouts_->get_all()){
    shared_layout* layout = x.second->to_shared();
    if(layout)
      intervals_[layout] = segment{INT32_MAX, 0};
  }

  for(auto& x: live_in)
  for(shared_layout* layout: x.second)
    intervals_[layout].start = std::min<int>(intervals_[layout].start, indices[x.first]);

  for(auto& x: live_in)
  for(shared_layout* layout: x.second){
    intervals_[layout].end = std::max<int>(intervals_[layout].end, indices[x.first] + 1);
  }

  
  for(auto &x: layouts_->get_all()) {
    shared_layout* layout = x.second->to_shared();
    if(!layout)
      continue;
    // std::cout << intervals_[layout].start << " " << intervals_[layout].end << std::endl;
  }

  

}

}
}
}
