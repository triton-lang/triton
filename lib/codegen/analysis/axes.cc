#include "triton/codegen/analysis/axes.h"
#include "triton/ir/instructions.h"
#include "triton/ir/type.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/context_impl.h"
#include "triton/ir/constant.h"
#include "triton/driver/device.h"



namespace triton{
namespace codegen{
namespace analysis{

axes::axes() {}

void axes::add_constraint(node_t x, node_t y) {
  size_t shape_x = 1;
  size_t shape_y = 1;
  if(x.first->get_type()->is_tile_ty())
    shape_x = x.first->get_type()->get_tile_shapes()[x.second];
  if(y.first->get_type()->is_tile_ty())
    shape_y = y.first->get_type()->get_tile_shapes()[y.second];
  if(shape_x == 1 && shape_y == 1)
    return;
  dependencies_[x].insert(y);
  dependencies_[y].insert(x);
  nodes_.insert(x);
  nodes_.insert(y);
}

void axes::init_c_graph(ir::instruction *v) {
  // Reference shape
  ir::type::tile_shapes_t shapes;
  if(auto *store = dynamic_cast<ir::store_inst*>(v))
    shapes = store->get_pointer_operand()->get_type()->get_tile_shapes();
  else if(auto *atom = dynamic_cast<ir::atomic_add_inst*>(v))
    shapes = atom->get_operand(0)->get_type()->get_tile_shapes();
  else if(dynamic_cast<ir::downcast_inst*>(v))
    return;
  else if(dynamic_cast<ir::copy_to_shared_inst*>(v))
    return;
  else if(auto *reduce = dynamic_cast<ir::reduce_inst*>(v)) {
    unsigned axis = reduce->get_axis();
    ir::value *arg = reduce->get_operand(0);
    auto in_shapes = arg->get_type()->get_tile_shapes();
    unsigned current = 0;
    for(unsigned i = 0; i < in_shapes.size(); i++){
      if(i == axis)
        continue;
      add_constraint({reduce, current++}, {arg, i});
    }
    return;
  }
  else
    shapes = v->get_type()->get_tile_shapes();
  // Reshape
  if(dynamic_cast<ir::reshape_inst*>(v)) {
    ir::value *op = v->get_operand(0);
    auto op_shapes = op->get_type()->get_tile_shapes();
    unsigned current = 0;
    bool is_skewed = false;
    for(unsigned i = 0; i < shapes.size(); i ++){
      if(shapes[i] == 1){
        add_constraint({v, i}, {v, i});
      }
      else if(!is_skewed &&
        shapes[i] == op_shapes[current])
        add_constraint({v, i}, {op, current++});
      else{
        is_skewed = true;
        add_constraint({v, i}, {v, i});
      }
    }
  }
  // Splat
  else if(dynamic_cast<ir::splat_inst*>(v)){
    return;
  }
  // Trans
  else if(auto *x = dynamic_cast<ir::trans_inst*>(v)){
    ir::value *op = v->get_operand(0);
    auto perm = x->get_perm();
    for(unsigned i = 0; i < perm.size(); i++)
      add_constraint({v, perm[i]->get_value()}, {op, i});
  }
  // Broadcast
  else if(dynamic_cast<ir::broadcast_inst*>(v)){
    ir::value *op = v->get_operand(0);
    ir::type *op_ty = op->get_type();
    const auto& op_shapes = op_ty->get_tile_shapes();
    for(unsigned i = 0; i < shapes.size(); i ++){
      if(op_shapes[i] == shapes[i] && v != op)
        add_constraint({v, i}, {op, i});
    }
  }
  // Matrix multiplication
  else if(dynamic_cast<ir::dot_inst*>(v)){
    ir::value *A = v->get_operand(0);
    ir::value *B = v->get_operand(1);
    ir::value *D = v->get_operand(2);
    for(unsigned i = 0; i < shapes.size(); i++)
      add_constraint({v, i}, {D, i});
    for(unsigned i = 2; i < shapes.size(); i++){
      add_constraint({v, i}, {A, i});
      add_constraint({v, i}, {B, i});
    }
  }
  // Element-wise
  else if(dynamic_cast<ir::user*>(v)) {
    for(unsigned i = 0; i < shapes.size(); i ++){
      std::vector<ir::value*> ops = v->ops();
      for(ir::value* op: ops)
        add_constraint({v, i}, {op, i});
    }
  }
}

void axes::connected_components(node_t x, std::set<node_t> &nodes, graph_t &graph, unsigned group_id) {
  groups_[x.first].insert({x.second, group_id});
  if(nodes.find(x) != nodes.end()){
    nodes.erase(x);
    for(const node_t &y: graph[x])
      connected_components(y, nodes, graph, group_id);
  }
}

unsigned axes::get(ir::value *value, unsigned ax) {
  unsigned result = groups_.at(value).at(ax);
  return result;
}

bool axes::has(ir::value *value, unsigned ax) {
  auto it = groups_.find(value);
  if(it == groups_.end())
    return false;
  auto iit = it->second.find(ax);
  if(iit == it->second.end())
    return false;
  return true;
}


void axes::run(ir::module &mod) {
  nodes_.clear();
  dependencies_.clear();
  groups_.clear();
  // Create graph
  for(ir::function *fn: mod.get_function_list()){
    // Build constraints graph
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *i : block->get_inst_list())
    if(i->has_tile_result_or_op())
      init_c_graph(i);
  }
  // Axes
  unsigned group_id = 0;
  while(!nodes_.empty())
    connected_components(*nodes_.begin(), nodes_, dependencies_, group_id++);
}

}
}

}
