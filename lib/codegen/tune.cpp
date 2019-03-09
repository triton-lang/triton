#include "triton/codegen/tune.h"
#include "triton/ir/instructions.h"
#include "triton/ir/type.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/context_impl.h"
#include "triton/ir/constant.h"

#include <cstdlib>


namespace triton{
namespace codegen{

tune::tune(): num_global_ranges_(0){ }

void tune::add_constraint(node_t x, node_t y) {
  dependencies_[x].insert(y);
  dependencies_[y].insert(x);
  nodes_.insert(x);
  nodes_.insert(y);
}

void tune::init_c_phi(ir::instruction *v) {
  // Phi Nodes: all the incoming value share the result layout
  if(auto *phi = dynamic_cast<ir::phi_node*>(v))
    for(ir::value *op: phi->ops())
      for(unsigned k = 0; k < phi->get_type()->get_tile_shapes().size(); k++)
        if(dependencies_.find({op, k}) != dependencies_.end()
           || dependencies_.find({phi, k}) != dependencies_.end()){
          add_constraint({phi, k}, {op, k});
        }
}

void tune::init_c_graph(ir::instruction *v) {
  // Reference shape
  ir::type::tile_shapes_t::value_type one = ir::tile_type::make_one(v->get_parent()->get_context());
  ir::type::tile_shapes_t shapes;
  if(auto *store = dynamic_cast<ir::store_inst*>(v))
    shapes = store->get_pointer_operand()->get_type()->get_tile_shapes();
  else
    shapes = v->get_type()->get_tile_shapes();
  // Reshape
  if(dynamic_cast<ir::reshape_inst*>(v)){
    ir::value *op = v->get_operand(0);
    unsigned current = 0;
    for(unsigned i = 0; i < shapes.size(); i ++){
      if(shapes[i] == one)
        static_params_.insert({{v, i}, 1});
      else
        add_constraint({v, i}, {op, current++});
    }
  }
  // Splat
  else if(dynamic_cast<ir::splat_inst*>(v)){

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
  else if(dynamic_cast<ir::matmul_inst*>(v)){
    ir::value *D = v->get_operand(2);
    add_constraint({v, 0}, {D, 0});
    add_constraint({v, 1}, {D, 1});
  }
  // Element-wise
  else if(dynamic_cast<ir::user*>(v)){
    for(unsigned k = 0; k < v->get_num_results(); k++)
      for(unsigned i = 0; i < shapes.size(); i ++)
        for(ir::value* op: v->ops())
          add_constraint({v->get_result(k), i}, {op, i});
  }
}

void tune::connected_components(node_t x, const std::vector<ir::metaparameter *> mps, std::set<node_t> &nodes, graph_t &graph) {
  if(nodes.find(x) != nodes.end()){
    nodes.erase(x);
    std::string suffix = ".d" + std::to_string(x.second);
    params_[x.first].insert({"p0" + suffix, mps[0]});
    params_[x.first].insert({"p1" + suffix, mps[1]});
    params_[x.first].insert({"p2" + suffix, mps[2]});
    ir::type *ty = x.first->get_type();
    if(ty->is_tile_ty()){
      ir::type::tile_shapes_t::value_type shape = ty->get_tile_shapes().at(x.second);
      if(auto mp = dynamic_cast<ir::metaparameter*>(shape))
        params_[x.first].insert({"shape" + suffix, mp});
    }
    if(auto range = dynamic_cast<ir::get_global_range_inst*>(x.first)){
      unsigned ax = range->get_axis();
      global_range_sizes_[ax] = params_[x.first].at("shape.d0");
      num_global_ranges_ = std::max(num_global_ranges_, ax + 1);
    }
    if(static_params_.find(x) != static_params_.end()){
      mps[0]->set_value(static_params_.at(x));
      mps[1]->set_value(static_params_.at(x));
      mps[2]->set_value(static_params_.at(x));
    }
    for(const node_t &y: graph[x])
      connected_components(y, mps, nodes, graph);
  }
}

std::vector<ir::metaparameter *> tune::get_params(ir::module &mod) {
  std::vector<ir::metaparameter*> result;
  std::set<ir::metaparameter*> seen;

  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i : block->get_inst_list())
  for(auto &x: params_[i])
    if(seen.insert(x.second).second && !x.second->has_value()){
      result.push_back(x.second);
    }
  return result;
}

std::map<std::string, ir::metaparameter *> tune::get_params(ir::instruction* i) {
  return params_.at(i);
}


void tune::run(ir::module &mod) {
  ir::context &ctx = mod.get_context();
  // Create metaparameters
  for(ir::function *fn: mod.get_function_list()){
    // Build constraints graph
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *i : block->get_inst_list())
    if(i->has_tile_result_or_op()){
      init_c_graph(i);
    }
    // Build phi constraints
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *i : block->get_inst_list())
    if(i->has_tile_result_or_op())
      init_c_phi(i);
    // Layout parameters
    while(!nodes_.empty()){
      ir::type *ty = mod.get_builder().get_int32_ty();
      ir::metaparameter *mp0 = ir::metaparameter::create(ctx, ty, 1, 4);
      ir::metaparameter *mp1 = ir::metaparameter::create(ctx, ty, 4, 32);
      ir::metaparameter *mp2 = ir::metaparameter::create(ctx, ty, 4, 32);
      connected_components(*nodes_.begin(), {mp0, mp1, mp2}, nodes_, dependencies_);
    }
  }

//  // Get launch info
//  for(ir::function *fn: mod.get_function_list()){
//    std::map<ir::metaparameter*, ir::instruction*> references;
//    std::vector<ir::instruction*> grids;
//    create_grids(grids, references, fn);
//    ir::instruction *first = grids.front();
//    for(unsigned i = 0; i < first->get_type()->get_tile_shapes().size(); i++){
//      std::string suffix = ".d" + std::to_string(i);
//      num_threads_mp_vec_.push_back(params_.at(first).at("p1" + suffix));
//      num_threads_mp_vec_.push_back(params_.at(first).at("p2" + suffix));
//    }
//  }
}

void tune::create_grids(std::vector<ir::instruction*> &grids,
                     std::map<ir::metaparameter*, ir::instruction*> &references,
                     ir::function *fn) {
  // get number of dimensions greater than 1
  auto get_tile_gt1_dim = [&](ir::value *v){
    unsigned result = 0;
    auto one = ir::tile_type::make_one(fn->get_fn_type()->get_context());
    for(ir::constant_int *shape: v->get_type()->get_tile_shapes()) {
      result += (shape != one);
    }
    return result;
  };
  // bind references
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list()){
    if(!i->get_type()->is_tile_ty())
      continue;
    for(auto &param: params_.at(i)){
      if(param.second->get_value() == 1)
        continue;
      ir::instruction *&r = references[param.second];
      if(!r || get_tile_gt1_dim(i) > get_tile_gt1_dim(r))
        r = i;
    }
  }
  // create grid
  for(auto &ref: references)
    if(std::find(grids.begin(), grids.end(), ref.second) == grids.end())
      grids.push_back(ref.second);
}


bool tune::check_constraints(ir::module &mod, std::map<ir::value *, std::vector<std::string>> &errors) {
for(ir::function *fn: mod.get_function_list()){
  using std::to_string;

  // initialize grids
  std::map<ir::metaparameter*, ir::instruction*> references;
  std::vector<ir::instruction*> grids;
  create_grids(grids, references, fn);

  for(unsigned i = 0; i < grids.front()->get_type()->get_tile_shapes().size(); i++){
    std::string suffix = ".d" + std::to_string(i);
    num_threads_mp_vec_.push_back(params_.at(grids.front()).at("p1" + suffix));
    num_threads_mp_vec_.push_back(params_.at(grids.front()).at("p2" + suffix));
  }

  // number of warps
  int num_warps = 1;
  for(size_t k = 0; k < grids.front()->get_type()->get_tile_shapes().size(); k++)
    num_warps *= params_[grids.front()]["p2.d" + to_string(k)]->get_value();

  // check constraints
  for(ir::instruction *i: grids){
    ir::type *ty = i->get_type();
    const auto &shapes = ty->get_tile_shapes();
    // for each dimension, the product of layout components
    // must device the shape
    for(size_t k = 0; k < shapes.size(); k++) {
      std::string strk = to_string(k);
      ir::metaparameter *mp0 = params_[i]["p0.d" + strk];
      ir::metaparameter *mp1 = params_[i]["p1.d" + strk];
      ir::metaparameter *mp2 = params_[i]["p2.d" + strk];
      unsigned multiple = mp0->get_value()*mp1->get_value()*mp2->get_value();
      if(shapes[k]->get_value() % multiple != 0)
        errors[i].push_back("for dim " + strk + ": shape (" + to_string(shapes[k]->get_value()) + ")"
                            " is not a multiple of layout (" + to_string(multiple)  + ")");
    }
    // the number of thread per warp must be 32
    int num_threads = 1;
    for(size_t k = 0; k < shapes.size(); k++)
      num_threads *= params_[i]["p1.d" + to_string(k)]->get_value();
    if(num_threads != 32)
      errors[i].push_back("number of threads per warp (" + to_string(num_threads) + ") must be 32");
    // The number of warps required by the layout is the same
    // for all tiles in the function
    int required_num_warps = 1;
    for(size_t k = 0; k < shapes.size(); k++)
      required_num_warps *= params_[i]["p2.d" + to_string(k)]->get_value();
    if(required_num_warps != num_warps)
      errors[i].push_back("number of warps (" + to_string(required_num_warps) + ") must be " + to_string(num_warps));
  }
  return errors.empty();
}
}

unsigned tune::get_num_global_range() {
  return num_global_ranges_;
}

unsigned tune::get_global_range_size(unsigned axis) {
  return global_range_sizes_.at(axis)->get_value();
}

unsigned tune::get_num_threads() {
  unsigned result = 1;
  for(ir::metaparameter *mp: num_threads_mp_vec_)
    result *= mp->get_value();
  return result;
}


}
}
