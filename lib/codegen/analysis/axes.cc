#include "triton/codegen/analysis/axes.h"
#include "triton/ir/instructions.h"
#include "triton/ir/utils.h"
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


void axes::update_graph_reduce(ir::instruction *i) {
  auto* red = static_cast<ir::reduce_inst*>(i);
  unsigned axis = red->get_axis();
  ir::value *arg = red->get_operand(0);
  auto in_shapes = arg->get_type()->get_tile_shapes();
  unsigned current = 0;
  for(unsigned d = 0; d < in_shapes.size(); d++){
    if(d == axis)
      continue;
    add_constraint({i, current++}, {arg, d});
  }
}

void axes::update_graph_reshape(ir::instruction *i) {
  auto* reshape = static_cast<ir::reshape_inst*>(i);
  // operands
  ir::value *op = reshape->get_operand(0);
  // shapes
  auto op_shapes = op->get_type()->get_tile_shapes();
  auto res_shapes = reshape->get_type()->get_tile_shapes();
  // construct edges
  unsigned current = 0;
  bool is_skewed = false;
  for(unsigned d = 0; d < res_shapes.size(); d ++){
    bool same_shape = res_shapes[d] == op_shapes[current];
    // either add edge between axis or just add a node in the graph
    if(!is_skewed && same_shape)
      add_constraint({i, d}, {op, current++});
    else
      add_constraint({i, d}, {i, d});
    // reshaping is skewed
    if(res_shapes[d] > 1 && !same_shape)
      is_skewed = true;
  }
}

void axes::update_graph_trans(ir::instruction *i) {
  auto *trans = static_cast<ir::trans_inst*>(i);
  ir::value *op = trans->get_operand(0);
  auto perm = trans->get_perm();
  // add edge between axis perm[d] and axis d
  for(unsigned d = 0; d < perm.size(); d++)
    add_constraint({i, perm[d]->get_value()}, {op, d});
}

void axes::update_graph_broadcast(ir::instruction *i) {
  auto *broadcast = static_cast<ir::broadcast_inst*>(i);
  auto shapes = broadcast->get_type()->get_tile_shapes();
  ir::value *op = broadcast->get_operand(0);
  ir::type *op_ty = op->get_type();
  const auto& op_shapes = op_ty->get_tile_shapes();
  // add edge between non-broadcast axes
  for(unsigned d = 0; d < shapes.size(); d ++)
    if(op_shapes[d] == shapes[d])
      add_constraint({i, d}, {op, d});
}

void axes::update_graph_dot(ir::instruction *i) {
  auto *dot = static_cast<ir::dot_inst*>(i);
  auto shapes = dot->get_type()->get_tile_shapes();
  ir::value *A = dot->get_operand(0);
  ir::value *B = dot->get_operand(1);
  ir::value *D = dot->get_operand(2);
  // add edges between result and accumulator
  for(unsigned d = 0; d < shapes.size(); d++)
    add_constraint({dot, d}, {D, d});
  // add edge for batch dimension
  for(unsigned d = 2; d < shapes.size(); d++){
    add_constraint({dot, d}, {A, d});
    add_constraint({dot, d}, {B, d});
  }
}

void axes::update_graph_elementwise(ir::instruction *i) {
  if(i->get_num_operands() == 0)
    return;
  ir::value *op = i->get_operand(0);
  if(!op->get_type()->is_tile_ty())
    return;
  auto rank = op->get_type()->get_tile_rank();
  for(unsigned d = 0; d < rank; d++)
  for(ir::value* opx: i->ops())
  for(ir::value* opy: i->ops()){
    if(!i->get_type()->is_void_ty())
      add_constraint({i, d}, {opx, d});
    add_constraint({opx, d}, {opy, d});
  }
}


void axes::update_graph(ir::instruction *i) {
  switch (i->get_id()) {
    case ir::INST_REDUCE:         return update_graph_reduce(i);
    case ir::INST_RESHAPE:        return update_graph_reshape(i);
    case ir::INST_SPLAT:          return;
    case ir::INST_TRANS:          return update_graph_trans(i);
    case ir::INST_BROADCAST:      return update_graph_broadcast(i);
    case ir::INST_DOT:            return update_graph_dot(i);
    case ir::INST_COPY_TO_SHARED: return;
    default:                      return update_graph_elementwise(i);
  }
  return;
}

void axes::connected_components(node_t x, std::set<node_t> &nodes, graph_t &graph, unsigned group_id) {
  groups_[x.first].insert({x.second, group_id});
  if(nodes.find(x) != nodes.end()){
    nodes.erase(x);
    for(const node_t &y: graph[x])
      connected_components(y, nodes, graph, group_id);
  }
}

unsigned axes::get_id(ir::value *value, unsigned ax) {
  unsigned result = groups_.at(value).at(ax);
  return result;
}

bool axes::has_id(ir::value *value, unsigned ax) {
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
  // make graph
  ir::for_each_instruction(mod, [this](ir::instruction *x) { update_graph(x); });
  // connected components
  unsigned group_id = 0;
  while(!nodes_.empty())
    connected_components(*nodes_.begin(), nodes_, dependencies_, group_id++);
}

}
}

}
