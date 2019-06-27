#include "triton/codegen/tune.h"
#include "triton/ir/instructions.h"
#include "triton/ir/type.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/context_impl.h"
#include "triton/ir/constant.h"
#include "triton/driver/device.h"

#include <cstdlib>


namespace triton{
namespace codegen{

tune::tune(): num_global_ranges_(0){ }

bool is_hmma(ir::value *v){
  bool result = false;
  if(auto *x = dynamic_cast<ir::dot_inst*>(v)){
    ir::value *a = x->get_operand(0);
    ir::type *a_ty = a->get_type();
    ir::value *b = x->get_operand(1);
    ir::type *b_ty = b->get_type();
    // only NT supported
    result = !x->is_a_trans() && x->is_b_trans();
    // inputs have to be FP16
    result = result && a_ty->get_scalar_ty()->is_half_ty() && b_ty->get_scalar_ty()->is_half_ty();
    // reduction has to be multiple of 4
    result = result && ((a_ty->get_tile_shapes()[1]->get_value() % 4) == 0);
  }
  return result;
}

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
  else if(auto *downcast = dynamic_cast<ir::downcast_inst*>(v))
    return;
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
  // Trans
  else if(dynamic_cast<ir::trans_inst*>(v)){
    ir::value *op = v->get_operand(0);
    size_t n_shapes = shapes.size();
    for(unsigned i = 0; i < n_shapes; i++){
      add_constraint({v, (i + 1) % n_shapes}, {op, i});
    }
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
    ir::value *D = v->get_operand(2);
    add_constraint({v, 0}, {D, 0});
    add_constraint({v, 1}, {D, 1});
  }
  // Element-wise
  else if(dynamic_cast<ir::user*>(v)) {
    for(unsigned k = 0; k < v->get_num_results(); k++)
      for(unsigned i = 0; i < shapes.size(); i ++){
        ir::value *result = v->get_result(k);
        for(ir::value* op: v->ops()){
          add_constraint({result, i}, {op, i});
        }
      }
  }
}

tune::fragment_t tune::get_fragmentation_type(node_t x, graph_t &graph){
  std::list<node_t> work;
  std::set<node_t> seen;
  work.push_back(x);
  while(!work.empty()){
    node_t current = work.back();
    if(is_hmma(current.first))
      return HMMA_FRAGMENT_C;
    work.pop_back();
    seen.insert(current);
    for(node_t y: graph[current]){
      if(seen.find(y) == seen.end())
        work.push_back(y);
    }
  }
  return STRIDED_SCAN;
}

void tune::connected_components(node_t x, const std::vector<ir::metaparameter *> mps, const std::vector<std::string> prefixes, std::set<node_t> &nodes, graph_t &graph, unsigned group_id) {
  groups_[x.first][x.second] = group_id;
  if(nodes.find(x) != nodes.end()){
    nodes.erase(x);
    std::string suffix = ".d" + std::to_string(x.second);
    for(int i = 0; i < mps.size(); i++)
      params_[x.first].insert({prefixes[i] + suffix, mps[i]});
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
      for(ir::metaparameter *mp: mps)
        mp->set_value(static_params_.at(x));
    }
    for(const node_t &y: graph[x])
      connected_components(y, mps, prefixes, nodes, graph, group_id);
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
//      std::cout << i->get_name() << " " << x.first << std::endl;
      result.push_back(x.second);
    }

  for(auto x: mod.globals()){
    if(auto mp = dynamic_cast<ir::metaparameter*>(x.second))
      if(seen.insert(mp).second && !mp->has_value())
        result.push_back(mp);
  }

  return result;
}

std::map<std::string, ir::metaparameter *> tune::get_params(ir::instruction* i) {
  return params_.at(i);
}

unsigned tune::get_param_group(ir::value *value, unsigned ax) {
  unsigned result = groups_.at(value).at(ax);
  return result;
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
    unsigned group_id = 0;
    for(auto x: nodes_){
      fragments_[x] = get_fragmentation_type(x, dependencies_);
    }
    while(!nodes_.empty()) {
      ir::type *ty = mod.get_builder().get_int32_ty();
      node_t node = *nodes_.begin();
      if(fragments_[node] == STRIDED_SCAN) {
        ir::metaparameter *nts = ir::metaparameter::create(ctx, ty, 1, 1);
        ir::metaparameter *mts = ir::metaparameter::create(ctx, ty, 4, 32);
        connected_components(node, {nts, mts}, {"nts", "mts"}, nodes_, dependencies_, group_id++);
        nts->set_value(1);
      }
      else {
        ir::metaparameter *fpw = ir::metaparameter::create(ctx, ty, 2, 2);
        ir::metaparameter *wpt = ir::metaparameter::create(ctx, ty, 1, 4);
        connected_components(node, {fpw, wpt}, {"fpw", "wpt"}, nodes_, dependencies_, group_id++);
      }
    }
  }

  // Simplify metaparameters
  for(ir::function *fn: mod.get_function_list())
  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i : block->get_inst_list()){
    if(fragments_.find({i, 0}) != fragments_.end() && fragments_.at({i, 0}) != STRIDED_SCAN)
      continue;
    if(dynamic_cast<ir::load_inst*>(i) && i->get_type()->is_tile_ty()){
      ir::type *ty = mod.get_builder().get_int32_ty();
      std::unique_ptr<ir::metaparameter> tmp(ir::metaparameter::create(ctx, ty, 2, 2));
      *params_.at(i).at("nts.d0") = *tmp;
    }
    if(dynamic_cast<ir::dot_inst*>(i) && i->get_type()->is_tile_ty()){
      ir::type *ty = mod.get_builder().get_int32_ty();
      std::unique_ptr<ir::metaparameter> tmp1(ir::metaparameter::create(ctx, ty, 2, 2));
      std::unique_ptr<ir::metaparameter> tmp2(ir::metaparameter::create(ctx, ty, 2, 2));
      *params_.at(i).at("nts.d0") = *tmp1;
      *params_.at(i).at("nts.d1") = *tmp2;
    }
  }
}

void tune::init(ir::module &mod) {
  for(ir::function *fn: mod.get_function_list()){
    // initialize grids
    std::map<ir::metaparameter*, ir::instruction*> references;
    create_grids(grids_, references, fn);
  }
  // number of threads
  num_threads_ = get_req_num_threads(grids_.front());
}

unsigned tune::get_req_num_threads(ir::instruction *i){
  if(fragments_.at({i, 0}) == STRIDED_SCAN) {
    unsigned result = 1;
    for(unsigned k = 0; k < i->get_type()->get_tile_shapes().size(); k++){
      std::string suffix = ".d" + std::to_string(k);
      result *= params_.at(i).at("mts" + suffix)->get_value();
    }
    return result;
  }
  else {
    unsigned result = 32;
    for(unsigned k = 0; k < i->get_type()->get_tile_shapes().size(); k++){
      std::string suffix = ".d" + std::to_string(k);
      result *= params_.at(i).at("wpt" + suffix)->get_value();
    }
    return result;
  }
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


bool tune::check_constraints(std::map<ir::value *, std::vector<std::string>> &errors) {
  using std::to_string;

  auto get_num_warps = [&](ir::instruction *i, unsigned axis) {
    std::string strk = to_string(axis);
    if(fragments_.at({i, axis}) == STRIDED_SCAN){
      unsigned mts = params_[i]["mts.d" + strk]->get_value();
      unsigned nts = params_[i]["nts.d" + strk]->get_value();
      unsigned shape = i->get_type()->get_tile_shapes()[axis]->get_value();
      return shape / (mts * nts);
    }
    else{
      return (unsigned)params_[i]["wpt.d" + strk]->get_value();
    }
  };

  // number of warps
  ir::instruction *first = grids_.front();
  int num_warps = 1;
  for(size_t k = 0; k < first->get_type()->get_tile_shapes().size(); k++)
    num_warps *= get_num_warps(first, k);

  // check constraints
  for(ir::instruction *i: grids_){
    ir::type *ty = i->get_type();
    const auto &shapes = ty->get_tile_shapes();
    // for each dimension, the product of layout components
    // must device the shape
    for(size_t k = 0; k < shapes.size(); k++) {
      std::string strk = to_string(k);
      unsigned multiple;
      if(fragments_.at({i, 0}) == STRIDED_SCAN) {
        ir::metaparameter *mts = params_[i]["mts.d" + strk];
        ir::metaparameter *nts = params_[i]["nts.d" + strk];
        multiple = mts->get_value()*nts->get_value();
      }
      else {
        ir::metaparameter *fpw = params_[i]["fpw.d" + strk];
        ir::metaparameter *wpt = params_[i]["wpt.d" + strk];
        multiple = fpw->get_value()*wpt->get_value()*8;
      }
      if(shapes[k]->get_value() % multiple != 0)
        errors[i].push_back("for dim " + strk + ": shape (" + to_string(shapes[k]->get_value()) + ")"
                            " is not a multiple of layout (" + to_string(multiple)  + ")");
    }
    // the product of mma fragments per warp must be 4
    if(fragments_.at({i, 0}) == HMMA_FRAGMENT_C){
      unsigned prod = 1;
      for(size_t k = 0; k < shapes.size(); k++){
        prod *= params_[i]["fpw.d" + std::to_string(k)]->get_value();
      }
      if(prod != 4)
        errors[i].push_back("HMMA must have only 4 fragments per warp");
    }
    int num_threads = get_req_num_threads(i);
    if(num_threads % 32 != 0)
      errors[i].push_back("number of threads per block (" + to_string(num_threads) + ") must be multiple of warp size");
    if(num_threads != num_threads_)
      errors[i].push_back("Number of threads must be the same for all tiles (" + to_string(num_threads_) + ")");
  }
  return errors.empty();
}

unsigned tune::get_num_global_range() {
  return num_global_ranges_;
}

unsigned tune::get_global_range_size(unsigned axis) {
  return global_range_sizes_.at(axis)->get_value();
}

unsigned tune::get_num_threads() {
  return num_threads_;
}


}
}
