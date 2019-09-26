#include <vector>
#include <set>
#include <algorithm>

#include "triton/codegen/analysis/liveness.h"
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/instructions.h"
#include "triton/codegen/transform/membar.h"
#include "triton/codegen/transform/cts.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/instructions.h"
#include "triton/ir/utils.h"

namespace triton {

namespace codegen{
namespace transform{

bool membar::intersect(const interval_vec_t &X, interval_t x) {
  return std::any_of(X.begin(), X.end(), [&](const interval_t &y){
    bool left_intersect = y.first <= x.first && x.first < y.second;
    bool right_intersect = y.first <= x.second && x.second < y.second;
    return left_intersect || right_intersect;
  });
}

bool membar::intersect(const interval_vec_t &X, const interval_vec_t &Y) {
  return std::any_of(Y.begin(), Y.end(), [&](const interval_t &y){
    return intersect(X, y);
  });
}

void membar::add_reference(ir::value *v, interval_vec_t &res){
  auto *i = dynamic_cast<ir::instruction*>(v);
  if(!i)
    return;
  if(alloc_->has_offset(v)){
    unsigned offset = alloc_->offset(v);
    unsigned num_bytes = liveness_->num_bytes(v);
    res.push_back(interval_t(offset, offset + num_bytes));
  }
}

void membar::get_read_intervals(ir::instruction *i, interval_vec_t &res){
  for(ir::value *op: i->ops())
    add_reference(op, res);
}

void membar::get_written_intervals(ir::instruction *i, interval_vec_t &res){
  if(!dynamic_cast<ir::phi_node*>(i))
    add_reference(i, res);
}

void membar::insert_barrier(ir::instruction *instr, ir::builder &builder) {
  if(auto *phi = dynamic_cast<ir::phi_node*>(instr)) {
    std::set<ir::value*> incoming;
    for(unsigned n = 0; n < phi->get_num_incoming(); n++){
      ir::instruction *inc_val = dynamic_cast<ir::instruction*>(phi->get_incoming_value(n));
      assert(inc_val);
      if(incoming.insert(inc_val).second){
        ir::basic_block *block = inc_val->get_parent();
        builder.set_insert_point(block->get_inst_list().back());
        builder.create_barrier();
      }
    }
  }
  else {
    builder.set_insert_point(instr);
    builder.create_barrier();
  }
}

membar::interval_vec_t membar::join(const std::vector<interval_vec_t>& intervals) {
  membar::interval_vec_t result;
  for(auto x: intervals)
    for(interval_t i: x)
      result.push_back(i);
  return result;
}

std::pair<membar::interval_vec_t,
          membar::interval_vec_t> membar::transfer(ir::basic_block *block,
                                            const interval_vec_t &written_to,
                                            const interval_vec_t &read_from,
                                            std::set<ir::instruction*>& insert_loc,
                                            std::set<ir::value*>& safe_war) {
  ir::basic_block::inst_list_t instructions = block->get_inst_list();
  interval_vec_t new_written_to = written_to;
  interval_vec_t new_read_from = read_from;

  for(ir::instruction *i: instructions){
    interval_vec_t read, written;
    get_read_intervals(i, read);
    get_written_intervals(i, written);
    bool read_after_write = intersect(new_written_to, read);
    bool write_after_read = intersect(new_read_from, written);
    // double buffering: write and phi-node read won't intersect
    if(safe_war.find(i) != safe_war.end()){
      write_after_read = false;
      read_after_write = false;
    }
    // record hazards
    if(read_after_write || write_after_read) {
      insert_loc.insert(i);
      new_written_to.clear();
      new_read_from.clear();
    }
    std::copy(written.begin(), written.end(), std::back_inserter(new_written_to));
    std::copy(read.begin(), read.end(), std::back_inserter(new_read_from));
  }
  return std::make_pair(new_written_to, new_read_from);
}

void membar::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();
  // extract phi-node associates with double-buffered
  // shared-memory copies. These can be read from and written to
  // without needing synchronization
  std::set<ir::value*> safe_war;
  ir::for_each_instruction(mod, [&](ir::instruction* i){
    if(liveness_->has_double(i)){
      auto info = liveness_->get_double(i);
      safe_war.insert(i);
      safe_war.insert(info.latch);
      auto *trans = dynamic_cast<ir::trans_inst*>(info.latch);
      if(trans)
        safe_war.insert(trans->get_operand(0));
    }
    if(i->get_id() == ir::INST_TRANS)
      safe_war.insert(i);
  });

  for(ir::function *fn: mod.get_function_list()){
    std::vector<ir::basic_block*> rpo = ir::cfg::reverse_post_order(fn);
    std::map<ir::basic_block*, interval_vec_t> written_to;
    std::map<ir::basic_block*, interval_vec_t> read_from;
    std::set<ir::instruction*> insert_locs;
    size_t n_inserted_im1 = 0;
    bool done = false;
    do{
      // find barrier location
      for(ir::basic_block *block: rpo){
        // written to
        std::vector<interval_vec_t> pred_written_to;
        for(ir::basic_block* pred: block->get_predecessors())
          pred_written_to.push_back(written_to[pred]);
        // read from
        std::vector<interval_vec_t> pred_read_from;
        for(ir::basic_block* pred: block->get_predecessors())
          pred_read_from.push_back(read_from[pred]);
        // apply transfer function
        auto result = transfer(block, join(pred_written_to), join(pred_read_from), insert_locs, safe_war);
        written_to[block] = result.first;
        read_from[block] = result.second;
      }
      size_t n_inserted_i = insert_locs.size();
      done = (n_inserted_im1 == n_inserted_i);
      n_inserted_im1 = n_inserted_i;
    }while(!done);
    for(ir::instruction* i: insert_locs)
      insert_barrier(i, builder);
  }
}

}
}
}
