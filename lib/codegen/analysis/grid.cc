#include <algorithm>
#include <cstdlib>
#include "triton/codegen/transform/coalesce.h"
#include "triton/codegen/analysis/grid.h"
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

grids::grids(size_t num_warps, transform::coalesce *reorder): num_warps_(num_warps), coalesce_(reorder)
{ }

bool is_hmma(ir::value *v){
  bool result = false;
  if(auto *x = dynamic_cast<ir::dot_inst*>(v)){
    ir::value *a = x->get_operand(0);
    ir::type *a_ty = a->get_type();
    ir::value *b = x->get_operand(1);
    ir::type *b_ty = b->get_type();
    // inputs have to be FP16
    result = a_ty->get_scalar_ty()->is_half_ty() && b_ty->get_scalar_ty()->is_half_ty();
    // reduction has to be multiple of 4: TODO
  }
  return result;
}

void grids::add_constraint(node_t x, node_t y) {
  dependencies_[x].insert(y);
  dependencies_[y].insert(x);
  nodes_.insert(x);
  nodes_.insert(y);
}

void grids::init_c_phi(ir::instruction *v) {
  // Phi Nodes: all the incoming value share the result layout
  if(auto *phi = dynamic_cast<ir::phi_node*>(v))
    for(ir::value *op: phi->ops())
      for(unsigned k = 0; k < phi->get_type()->get_tile_shapes().size(); k++)
        if(dependencies_.find({op, k}) != dependencies_.end()
           || dependencies_.find({phi, k}) != dependencies_.end()){
          add_constraint({phi, k}, {op, k});
        }
}

void grids::init_c_graph(ir::instruction *v) {
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

grids::fragment_t grids::get_fragmentation_type(node_t x, graph_t &graph){
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

void grids::connected_components(node_t x, const std::vector<param_ptr_t>& ptr_vec, const std::vector<param_map_t*>& maps,
                                 std::set<node_t> &nodes, graph_t &graph, unsigned group_id) {
  groups_[x.first].insert({x.second, group_id});
  if(nodes.find(x) != nodes.end()){
    nodes.erase(x);
    for(unsigned i = 0; i < ptr_vec.size(); i++)
      (*maps[i])[x.first][x.second] = ptr_vec[i];
    for(const node_t &y: graph[x])
      connected_components(y, ptr_vec, maps, nodes, graph, group_id);
  }
}

unsigned grids::group_of(ir::value *value, unsigned ax) {
  unsigned result = groups_.at(value).at(ax);
  return result;
}

grids::fragment_t grids::fragment_of(ir::value *value, unsigned ax) {
  return fragments_.at({value, ax});
}


//TODO: This shouldn't exist!
void grids::copy(ir::value *dst, ir::value *src) {
  mts_[dst] = mts_[src];
  nts_[dst] = nts_[src];
  fpw_[dst] = fpw_[src];
  wpt_[dst] = wpt_[src];
  groups_[dst] = groups_[src];
  fragments_[{dst, 0}] = fragments_[{src, 0}];
}


void grids::run(ir::module &mod) {
  // Create tiling parameters
  for(ir::function *fn: mod.get_function_list()){
    // Build constraints graph
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *i : block->get_inst_list())
    if(i->has_tile_result_or_op())
      init_c_graph(i);
    // Build phi constraints
    for(ir::basic_block *block: fn->blocks())
    for(ir::instruction *i : block->get_inst_list())
    if(i->has_tile_result_or_op())
      init_c_phi(i);
    // Layout parameters
    unsigned group_id = 0;
    for(auto x: nodes_)
      fragments_[x] = get_fragmentation_type(x, dependencies_);
    while(!nodes_.empty()) {
      node_t node = *nodes_.begin();
      if(fragments_[node] == STRIDED_SCAN) {
        param_ptr_t nts(new int(-1));
        param_ptr_t mts(new int(-1));
        connected_components(node, {nts, mts}, {&nts_, &mts_}, nodes_, dependencies_, group_id++);
      }
      else {
        param_ptr_t fpw(new int(-1));
        param_ptr_t wpt(new int(-1));
        connected_components(node, {fpw, wpt}, {&fpw_, &wpt_}, nodes_, dependencies_, group_id++);
      }
    }
  }

  for(ir::function *fn: mod.get_function_list()){
    std::map<unsigned, ir::value*> references;
    create_grids(grids_, references, fn);
  }


  unsigned num_threads = num_warps_*32;
  auto clamp = [&](unsigned x, unsigned lo, unsigned hi) { return std::min(std::max(x, lo), hi); };

  for(ir::value *i: grids_){
    if(!i->get_type()->is_tile_ty())
      continue;
    auto order = coalesce_->get_order(i);
    auto shapes = i->get_type()->get_tile_shapes();
    unsigned size = i->get_type()->get_tile_num_elements();
    /* HMMA parameters*/
    if(fragments_.at({i, 0}) == HMMA_FRAGMENT_C){
      unsigned shape_0 = shapes[order[0]];
      unsigned shape_1 = shapes[order[1]];
      /* fragments per warp */
      // try to make things as square as possible to maximize data re-use
      std::vector<unsigned> fpw = {1, 1, 1};
      std::vector<unsigned> fpw_nm1;
      unsigned num_fragments = std::min<unsigned>((shape_0/8)*(shape_1/8), 4);
      do {
        fpw_nm1 = fpw;
        if(fpw[0]*fpw[1] < num_fragments)
          fpw[0] = clamp(fpw[0]*2, 1, shape_0 / 8);
        if(fpw[0]*fpw[1] < num_fragments)
          fpw[1] = clamp(fpw[1]*2, 1, shape_1 / 8);
      }while(fpw_nm1 != fpw);
      // store parameters
      for(unsigned d = 0; d < shapes.size(); d++)
        *fpw_[i][d] = fpw[d];
      /* warps per tile */
      // try to make things as square as possible to maximize data re-use
      std::vector<unsigned> wpt = {1, 1, 1};
      std::vector<unsigned> wpt_nm1;
      do{
        wpt_nm1 = wpt;
        if(wpt[0] * wpt[1] * wpt[2] < num_warps_)
          wpt[0] = clamp(wpt[0]*2, 1, shape_0 / (fpw[0]*8));
        if(wpt[0] * wpt[1] * wpt[2] < num_warps_)
          wpt[1] = clamp(wpt[1]*2, 1, shape_1 / (fpw[1]*8));
      }while(wpt_nm1 != wpt);
      // store parameters
      for(unsigned d = 0; d < shapes.size(); d++)
        *wpt_[i][d] = wpt[d];
      /* sanity check */
      unsigned effective_num_warps = 1;
      for(size_t d = 0; d < shapes.size(); d++)
        effective_num_warps *= *wpt_[i][d];
      if(num_warps_ != effective_num_warps)
        throw std::runtime_error("cannot create a kernel with this amount of warps");
    }

    /* Scan-line */
    else{
      unsigned ld = order[0];
      unsigned current = num_threads;
      *nts_[i][ld] = clamp(size / num_threads, 1, 4);
      *mts_[i][ld] = clamp(current, 1, shapes[ld] / *nts_[i][ld]);
      current = current / *mts_[i][ld];
      for(size_t d = 1; d < shapes.size(); d++){
        ld = order[d];
        *nts_[i][ld] = 1;
        *mts_[i][ld] = clamp(current, 1, shapes[ld]);
        current = current / *mts_[i][ld];
      }
      /* sanity check */
      unsigned effective_num_threads = 1;
      for(size_t d = 0; d < shapes.size(); d++)
        effective_num_threads *= *mts_[i][d];
      if(num_threads != effective_num_threads)
        throw std::runtime_error("cannot create a kernel with this amount of warps");
    }
  }

}


void grids::create_grids(std::vector<ir::value*> &grids,
                             std::map<unsigned, ir::value*> &references,
                             ir::function *fn) {
  // get number of dimensions greater than 1
  auto get_tile_gt1_dim = [&](ir::value *v){
    unsigned result = 0;
    for(auto shape: v->get_type()->get_tile_shapes()) {
      result += (shape > 1)? shape : 0;
    }
    return result;
  };
  // bind references
  std::set<ir::value*> seen;
  std::function<void(ir::value*)> bind_references = [&](ir::value *v)
  {
    // skip
    if(!v->get_type()->is_tile_ty() || !seen.insert(v).second)
      return;
    // recurse
    if(auto *user = dynamic_cast<ir::user*>(v))
      for(ir::value *op: user->ops())
        bind_references(op);
    // bind
    const auto& shapes = v->get_type()->get_tile_shapes();
    for(size_t d = 0; d < shapes.size(); d++){
      if(shapes[d] == 1)
        continue;
      unsigned x = group_of(v, d);
      ir::value *&r = references[x];
      if(!r || get_tile_gt1_dim(v) > get_tile_gt1_dim(r))
        r = v;
    }
  };

  for(ir::basic_block *block: fn->blocks())
  for(ir::instruction *i: block->get_inst_list())
    bind_references(i);

  // create grid
  for(auto &ref: references)
    if(std::find(grids.begin(), grids.end(), ref.second) == grids.end())
      grids.push_back(ref.second);
}

int grids::mts(ir::value *value, unsigned ax) {
  return *mts_.at(value).at(ax);
}

int grids::nts(ir::value *value, unsigned ax) {
  return *nts_.at(value).at(ax);
}

int grids::fpw(ir::value *value, unsigned ax) {
  return *fpw_.at(value).at(ax);
}

int grids::wpt(ir::value *value, unsigned ax) {
  return *wpt_.at(value).at(ax);
}

}
}
}
