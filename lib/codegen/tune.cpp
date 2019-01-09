#include "codegen/tune.h"
#include "ir/instructions.h"
#include "ir/type.h"
#include "ir/module.h"
#include "ir/function.h"
#include <cstdlib>


namespace tdl{
namespace codegen{

void tune::add_constraint(node_t x, node_t y) {
  dependencies_[x].insert(y);
  dependencies_[y].insert(x);
  nodes_.insert(x);
  nodes_.insert(y);
}

void tune::init_c_phi(ir::instruction *v) {
  // Phi Nodes: all the incoming value share the result layout
  if(auto *phi = dynamic_cast<ir::phi_node*>(v))
    for(ir::value *inc: phi->ops())
      for(unsigned k = 0; k < phi->get_type()->get_tile_shapes().size(); k++)
        if(dependencies_.find({inc, k}) != dependencies_.end()
           || dependencies_.find({phi, k}) != dependencies_.end())
          add_constraint({phi, k}, {inc, k});
}

void tune::init_c_graph(ir::instruction *v) {
  unsigned num_dim = v->get_type()->get_tile_shapes().size();
  if(dynamic_cast<ir::reshape_inst*>(v)){

  }
  else if(dynamic_cast<ir::splat_inst*>(v)){

  }
  else if(dynamic_cast<ir::broadcast_inst*>(v)){

  }
  else if(auto *ii = dynamic_cast<ir::matmul_inst*>(v)){
    ir::value *D = ii->get_operand(2);
    add_constraint({v, 0}, {D, 0});
    add_constraint({v, 1}, {D, 1});
  }
  else if(dynamic_cast<ir::user*>(v))
    for(unsigned i = 0; i < num_dim; i ++)
      for(ir::value* op: v->ops())
        add_constraint({v, i}, {op, i});
}

void tune::connected_components(node_t x, const std::vector<unsigned *> vals, std::set<node_t> &nodes, graph_t &graph) {
  if(nodes.find(x) != nodes.end()){
    nodes.erase(x);
    std::string suffix = ".d" + std::to_string(x.second);
    if(auto *instr = dynamic_cast<ir::instruction*>(x.first)){
      params_[instr].insert({"p0" + suffix, vals[0]});
      params_[instr].insert({"p1" + suffix, vals[1]});
      params_[instr].insert({"p2" + suffix, vals[2]});
    }
    for(const node_t &y: graph[x])
      connected_components(y, vals, nodes, graph);
  }
}

void tune::get_params(ir::module &mod, std::vector<unsigned *> &result) {
  result.clear();
  std::set<unsigned*> seen;
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i : block->get_inst_list())
  for(auto &x: params_[i])
    if(seen.insert(x.second).second)
      result.push_back(x.second);
}

void tune::run(ir::module &mod) {
  for(ir::function *fn: mod.get_function_list()){
    // Build constraints graph
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *i : block->get_inst_list())
    if(i->get_type()->is_tile_ty())
      init_c_graph(i);
    // Build phi constraints
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *i : block->get_inst_list())
    if(i->get_type()->is_tile_ty())
      init_c_phi(i);
    // Layout parameters
    while(!nodes_.empty()){
      unsigned *v0 = new unsigned(0);
      unsigned *v1 = new unsigned(0);
      unsigned *v2 = new unsigned(0);
      connected_components(*nodes_.begin(), {v0, v1, v2}, nodes_, dependencies_);
    }
  }
}

bool tune::check_constraints(ir::module &mod, std::map<ir::value *, std::vector<std::string>> &errors) {
for(ir::function *fn: mod.get_function_list()){
  /* grids */
  auto get_tile_gt1_dim = [&](ir::value *v){
    unsigned result = 0;
    for(unsigned shape: v->get_type()->get_tile_shapes()) {
      result += (shape > 1)?shape:0;
    }
    return result;
  };
  using std::to_string;
  std::map<unsigned*, ir::instruction*> references;
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list()){
    if(!i->get_type()->is_tile_ty())
      continue;
    for(auto &param: params_.at(i)){
      if(*param.second == 1)
        continue;
      ir::instruction *&r = references[param.second];
      if(!r && get_tile_gt1_dim(i) > get_tile_gt1_dim(r))
        r = i;
    }
  }

  // extract unique instructions in order
  std::vector<ir::instruction*> grids;
  for(auto &ref: references)
    if(std::find(grids.begin(), grids.end(), ref.second) == grids.end())
      grids.push_back(ref.second);

  // number of warps
  int num_warps = 1;
  for(size_t k = 0; k < grids.front()->get_type()->get_tile_shapes().size(); k++)
    num_warps *= *params_[grids.front()]["p2.d" + to_string(k)];

  // check constraints
  for(ir::instruction *i: grids){
    ir::type *ty = i->get_type();
    const auto &shapes = ty->get_tile_shapes();
    // for each dimension, the product of layout components
    // must device the shape
    for(size_t k = 0; k < shapes.size(); k++) {
      std::string strk = to_string(k);
      unsigned *s0 = params_[i]["p0.d" + strk];
      unsigned *s1 = params_[i]["p1.d" + strk];
      unsigned *s2 = params_[i]["p2.d" + strk];
      unsigned multiple = (*s0)*(*s1)*(*s2);
      if(shapes[k] % multiple != 0)
        errors[i].push_back("for dim " + strk + ": shape (" + to_string(shapes[k]) + ")"
                            " is not a multiple of layout (" + to_string(multiple)  + ")");
    }
    // the number of thread per warp must be 32
    int num_threads = 1;
    for(size_t k = 0; k < shapes.size(); k++)
      num_threads *= *params_[i]["p1.d" + to_string(k)];
    if(num_threads != 32)
      errors[i].push_back("number of threads per warp (" + to_string(num_threads) + ") must be 32");
    // The number of warps required by the layout is the same
    // for all tiles in the function
    int required_num_warps = 1;
    for(size_t k = 0; k < shapes.size(); k++)
      required_num_warps *= *params_[i]["p2.d" + to_string(k)];
    if(required_num_warps != num_warps)
      errors[i].push_back("number of warps (" + to_string(required_num_warps) + ") must be " + to_string(num_warps));
  }
  return errors.empty();
}
}

}
}
