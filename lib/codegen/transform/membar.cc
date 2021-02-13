#include <vector>
#include <set>
#include <algorithm>
#include "triton/codegen/analysis/layout.h"
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/transform/membar.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/utils.h"

namespace triton {

namespace codegen{
namespace transform{


void membar::insert_barrier(ir::instruction *instr, bool read_after_write, ir::builder &builder) {
    builder.set_insert_point(instr);
    if(read_after_write){
      builder.create_async_wait(2);
    }
    else
      builder.create_barrier();
}


int membar::get_req_group_id(ir::value* v, std::vector<ir::instruction *> &async_write) {
  analysis::shared_layout* layout = layouts_->get(v)->to_shared();
  if(!layout)
    return -1;
  if(ir::phi_node* phi = dynamic_cast<ir::phi_node*>(v)){
    analysis::double_buffer_info_t* info = layout->get_double_buffer();
    if(info)
      return get_req_group_id(info->first, async_write);
    int ret = -1;
    for(ir::value* op: phi->ops())
      ret = std::max(ret, get_req_group_id(op, async_write));
    return ret;
  }
  else{
    auto it = std::find(async_write.begin(), async_write.end(), v);
    return std::distance(async_write.begin(), it);
  }
}

void membar::transfer(ir::basic_block *block,
                      vec_inst_t& async_write,
                      vec_inst_t& sync_write,
                      vec_inst_t& sync_read,
                      std::set<ir::value*>& safe_war,
                      bool& inserted, ir::builder& builder) {
  ir::basic_block::inst_list_t instructions = block->get_inst_list();
  for(ir::instruction *i: instructions){
    if(dynamic_cast<ir::phi_node*>(i))
      continue;
    if(dynamic_cast<ir::masked_load_async_inst*>(i))
      async_write.push_back(i);
    if(dynamic_cast<ir::copy_to_shared_inst*>(i))
      sync_write.push_back(i);
    // RAW hazard
    int ret = -1;
    for(ir::value* op: i->ops())
      ret = std::max(ret, get_req_group_id(op, async_write));
    if(ret != -1){
      int N = async_write.size() - 1 - ret;
      builder.create_async_wait(N);
    }
    // WAR hazard
  }
}

void membar::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();
  // extract phi-node associates with double-buffered
  // shared-memory copies. These can be read from and written to
  // without needing synchronization
  std::set<ir::value*> safe_war;
  for(const auto& x: layouts_->get_all()){
    analysis::shared_layout* layout = x.second->to_shared();
    if(!layout || !layout->get_double_buffer())
      continue;
    for(ir::value *v: layout->get_values())
      if(v != layout->get_double_buffer()->phi){
        safe_war.insert(v);
      }
  }

  for(ir::function *fn: mod.get_function_list()){
    std::vector<ir::basic_block*> rpo = ir::cfg::reverse_post_order(fn);
    std::map<ir::basic_block*, vec_inst_t> async_writes;
    std::map<ir::basic_block*, vec_inst_t> sync_writes;
    std::map<ir::basic_block*, vec_inst_t> sync_reads;
    std::vector<ir::instruction*> to_sync;
    std::list<ir::value *> pipelined;
    bool inserted;
    do{
      inserted = false;
      // find barrier location
      for(ir::basic_block *block: rpo){
        // join inputs
        vec_inst_t async_write;
        vec_inst_t sync_write;
        vec_inst_t sync_read;
        for(ir::basic_block* pred: block->get_predecessors()){
          async_write.insert(async_write.end(), async_writes[pred].begin(), async_writes[pred].end());
          async_write.insert(sync_read.end(), sync_reads[pred].begin(), sync_reads[pred].end());
          async_write.insert(sync_write.end(), sync_writes[pred].begin(), async_writes[pred].end());
        }
        transfer(block, async_write, sync_write, sync_read, safe_war, inserted, builder);
        async_writes[block] = async_write;
        sync_writes[block] = sync_write;
        sync_reads[block] = sync_read;
      }
    }while(inserted);
  }
}

}
}
}
