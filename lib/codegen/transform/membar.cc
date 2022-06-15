#include <vector>
#include <set>
#include <algorithm>
#include "triton/codegen/analysis/layout.h"
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/transform/membar.h"
#include "triton/codegen/transform/prefetch.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/utils.h"

namespace triton {

namespace codegen{
namespace transform{



int membar::group_of(ir::value* v, std::vector<ir::value*> &async_write) {
  if(ir::phi_node* phi = dynamic_cast<ir::phi_node*>(v)){
    analysis::shared_layout* layout = layouts_->get(v)->to_shared();
    if (analysis::double_buffer_info_t* info = layout->get_double_buffer())
      return group_of(info->first, async_write);
    else if (analysis::N_buffer_info_t* info = layout->get_N_buffer()) {
      if (v == info->phi)
        return group_of(info->firsts[0], async_write);
      else // prefetched value
        return group_of(info->firsts[1], async_write);
    }
    std::vector<int> groups(phi->get_num_operands());
    std::transform(phi->op_begin(), phi->op_end(), groups.begin(), [&](ir::value* v){ return group_of(v, async_write);});
    return *std::max_element(groups.begin(), groups.end());
  }
  else{
    if(layouts_->has_tmp(v))
      return async_write.size() - 1;
    auto it = std::find(async_write.begin(), async_write.end(), v);
    return std::distance(async_write.begin(), it);
  }
}

inline bool membar::intersect_with(analysis::shared_layout* a_layout, analysis::shared_layout* b_layout) {
  if(!a_layout || !b_layout)
    return false;
  int a_start = alloc_->offset(a_layout);
  int a_end = a_start + a_layout->get_size();
  int b_start = alloc_->offset(b_layout);
  int b_end = b_start + b_layout->get_size();
  if(a_start < b_end || b_start < a_end)
    return true;
  return false;
}

membar::val_set_t membar::intersect_with(const val_set_t& as, const val_set_t& bs) {
  val_set_t ret;
  for(ir::value* a: as){
    if(!a->get_type()->is_block_ty())
      continue;
    analysis::shared_layout* a_layout = layouts_->get(a)->to_shared();
    analysis::shared_layout* a_tmp = layouts_->has_tmp(a) ? layouts_->get(layouts_->tmp(a))->to_shared() : nullptr;
    analysis::shared_layout* a_tmp_index = layouts_->has_tmp_index(a) ? layouts_->get(layouts_->tmp_index(a))->to_shared() : nullptr;
    for(ir::value* b: bs){
      if(!b->get_type()->is_block_ty())
        continue;
      analysis::shared_layout* b_layout = layouts_->get(b)->to_shared();
      analysis::shared_layout* b_tmp = layouts_->has_tmp(b) ? layouts_->get(layouts_->tmp(b))->to_shared() : nullptr;
      analysis::shared_layout* b_tmp_index = layouts_->has_tmp_index(b) ? layouts_->get(layouts_->tmp_index(b))->to_shared() : nullptr;
      if(intersect_with(a_layout, b_layout) ||
         intersect_with(a_layout, b_tmp) ||
         intersect_with(a_layout, b_tmp_index) ||
         intersect_with(a_tmp, b_layout) ||
         intersect_with(a_tmp, b_tmp) ||
         intersect_with(a_tmp, b_tmp_index) ||
         intersect_with(a_tmp_index, b_layout) ||
         intersect_with(a_tmp_index, b_tmp) ||
         intersect_with(a_tmp_index, b_tmp_index))
        ret.insert(b);
    }
  }
  return ret;
}

bool membar::check_safe_war(ir::instruction* i) {
  bool is_i_shared_block = i->get_type()->is_block_ty() &&
                          layouts_->get(i)->to_shared();
  bool is_i_double_buffered = is_i_shared_block &&
                              layouts_->get(i)->to_shared()->get_double_buffer();
  bool is_i_n_buffered = is_i_shared_block && 
                          layouts_->get(i)->to_shared()->get_N_buffer();
  
  if (is_i_double_buffered || is_i_n_buffered) {
    // with async copy & prefetch_s disabled, WARs are not safe
    if (dynamic_cast<ir::masked_load_async_inst*>(i) && !prefetch_->is_prefetched(i))
      return false;
    else
      return true;
  }
  return false;
}

void membar::transfer(ir::basic_block *block,
                      val_vec_t& async_write,
                      val_set_t& sync_write,
                      val_set_t& sync_read,
                      std::set<ir::value*>& safe_war,
                      bool& inserted, ir::builder& builder) {
  std::vector<ir::async_wait_inst*> async_waits;
  ir::basic_block::inst_list_t instructions = block->get_inst_list();
  for(ir::instruction *i: instructions){
    if(dynamic_cast<ir::phi_node*>(i))
      continue;
    if(std::find(async_write.begin(), async_write.end(), i) == async_write.end() &&
       dynamic_cast<ir::masked_load_async_inst*>(i)){
      async_write.push_back(i);
    }
    if(dynamic_cast<ir::copy_to_shared_inst*>(i))
      sync_write.insert(i);
    ir::barrier_inst* barrier = dynamic_cast<ir::barrier_inst*>(i);
    ir::async_wait_inst* async_wait = dynamic_cast<ir::async_wait_inst*>(i);
    // Get shared memory reads
    std::set<ir::value*> read;
    std::copy_if(i->op_begin(), i->op_end(), std::inserter(read, read.begin()),
                 [&](ir::value* i){ return i->get_type()->is_block_ty() && layouts_->get(i)->to_shared();});
    if(layouts_->has_tmp(i))
      read.insert(i);
    // RAW (async)
    val_set_t tmp;
    std::copy(async_write.begin(), async_write.end(), std::inserter(tmp, tmp.begin()));
    if(intersect_with(read, tmp).size()){
      std::vector<int> groups(read.size());
      std::transform(read.begin(), read.end(), groups.begin(), [&](ir::value* v){ return group_of(v, async_write);});
      int N = *std::max_element(groups.begin(), groups.end());
      if(N < async_write.size()){
        builder.set_insert_point(i);
        async_wait = (ir::async_wait_inst*)builder.create_async_wait(async_write.size() - 1 - N);
        barrier = (ir::barrier_inst*)builder.create_barrier();
        inserted = true;
        async_waits.push_back(async_wait);
      }
    }
    // RAW, WAR
    bool is_safe_war = check_safe_war(i);
    // WAR barrier is not required when data is double-buffered
    if(!intersect_with(read, sync_write).empty() || 
       (!intersect_with({i}, sync_read).empty() && !is_safe_war)) {
      builder.set_insert_point(i);
      barrier = (ir::barrier_inst*)builder.create_barrier();
      inserted = true;
    }
    // update state of asynchronous copies
    if(async_wait){
      int N = async_write.size() - async_wait->get_N();
      async_write.erase(async_write.begin(), async_write.begin() + N);
    }
    // all the copy_to_shared and read from shared are synchronized after barrier
    if(barrier){
      sync_write.clear();
      sync_read.clear();
    }
    sync_read.insert(read.begin(), read.end());
  }

  // coalesce barriers
  // fixme: to support more general cases
  if (async_waits.size() == 2) {
    // (aw N; bar; prefetch; aw N-1; bar; prefetch; => aw N-1; bar; 2*prefetch;)
    for (int idx=0; idx<async_waits.size()-1; ++idx) {
      ir::async_wait_inst *first_async_wait = async_waits[idx];
      std::vector<ir::instruction*> to_erase;
      ir::basic_block::inst_list_t instructions = block->get_inst_list();
      for(auto iter = instructions.begin(); iter != instructions.end(); ++iter){
        ir::instruction *i = *iter;
        if (static_cast<ir::instruction*>(first_async_wait) == i) {
          // peak next 5 instructions
          auto peak_iter = std::next(iter);
          if (std::distance(peak_iter, instructions.end()) >= 5) {
            auto first_bar = dynamic_cast<ir::barrier_inst*>(*peak_iter++);
            auto first_pf = dynamic_cast<ir::prefetch_s_inst*>(*peak_iter++);
            auto second_async_wait = dynamic_cast<ir::async_wait_inst*>(*peak_iter++);
            auto second_bar = dynamic_cast<ir::barrier_inst*>(*peak_iter++);
            auto second_pf = dynamic_cast<ir::prefetch_s_inst*>(*peak_iter);
            if (first_bar && first_pf && second_async_wait && second_bar && second_pf) {
              int first_n = first_async_wait->get_N();
              int second_n = second_async_wait->get_N();
              to_erase.push_back(second_async_wait);
              to_erase.push_back(second_bar);
              first_async_wait->set_N(second_n);
            }
          } else 
            break;
          for (ir::instruction *i : to_erase)
            block->erase(i);
        }
      }
    }
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
    if(!layout || !layout->get_double_buffer() || !layout->get_N_buffer())
      continue;
    for(ir::value *v: layout->get_values())
      if(v != layout->get_double_buffer()->phi){
        safe_war.insert(v);
      }
  }

  for(ir::function *fn: mod.get_function_list()){
    std::vector<ir::basic_block*> rpo = ir::cfg::reverse_post_order(fn);
    std::map<ir::basic_block*, val_vec_t> async_writes;
    std::map<ir::basic_block*, val_set_t> sync_writes;
    std::map<ir::basic_block*, val_set_t> sync_reads;
    std::list<ir::value *> pipelined;
    bool inserted;
    do{
      inserted = false;
      // find barrier location
      for(ir::basic_block *block: rpo){
        // join inputs
        val_vec_t async_write;
        val_set_t sync_write;
        val_set_t sync_read;
        val_set_t tmp;
        for(ir::basic_block* pred: block->get_predecessors()){
          for(ir::value* v: async_writes[pred])
            if(tmp.insert(v).second)
              async_write.push_back(v);
          sync_write.insert(sync_writes[pred].begin(), sync_writes[pred].end());
          sync_read.insert(sync_reads[pred].begin(), sync_reads[pred].end());
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
