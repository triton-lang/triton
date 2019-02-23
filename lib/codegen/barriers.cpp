#include <algorithm>
#include "codegen/barriers.h"
#include "codegen/allocation.h"
#include "codegen/buffer_info.h"
#include "ir/module.h"
#include "ir/function.h"
#include "ir/basic_block.h"
#include "ir/instructions.h"

namespace tdl {

namespace codegen{

bool barriers::intersect(const interval_vec_t &X, interval_t x) {
  return std::any_of(X.begin(), X.end(), [&](const interval_t &y){
    bool left_intersect = y.first <= x.first && x.first < y.second;
    bool right_intersect = y.first <= x.second && x.second < y.second;
    return left_intersect || right_intersect;
  });
}

bool barriers::intersect(const interval_vec_t &X, const interval_vec_t &Y) {
  return std::any_of(Y.begin(), Y.end(), [&](const interval_t &y){
    return intersect(X, y);
  });
}

void barriers::add_reference(ir::value *v, interval_vec_t &res){
  if(dynamic_cast<ir::copy_to_shared_inst*>(v)){
    unsigned offset = alloc_->get_offset(v);
    unsigned num_bytes = alloc_->get_num_bytes(v);
    res.push_back(interval_t(offset, offset + num_bytes));
  }
}

void barriers::get_read_intervals(ir::instruction *i, interval_vec_t &res){
  for(ir::value *op: i->ops())
    add_reference(op, res);
}

void barriers::get_written_intervals(ir::instruction *i, interval_vec_t &res){
  if(!dynamic_cast<ir::phi_node*>(i))
    add_reference(i, res);
}

void barriers::insert_barrier(ir::instruction *instr, ir::builder &builder) {
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

void barriers::add(ir::basic_block *block, interval_vec_t &not_synced, ir::builder &builder) {
  ir::basic_block::inst_list_t instructions = block->get_inst_list();
  for(ir::instruction *i: instructions){
    interval_vec_t read, written;
    get_read_intervals(i, read);
    get_written_intervals(i, written);
    if(intersect(not_synced, read)) {
      not_synced.clear();
      insert_barrier(i, builder);
    }
    std::copy(written.begin(), written.end(), std::back_inserter(not_synced));
  }
}

void barriers::run(ir::module &mod) {
  ir::builder &builder = mod.get_builder();
  for(ir::function *fn: mod.get_function_list()){
    // find barrier location
    interval_vec_t not_synced;
    for(ir::basic_block *block: fn->blocks())
      add(block, not_synced, builder);
  }
}

}
}
