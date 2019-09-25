#include <algorithm>
#include <climits>
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/analysis/liveness.h"
#include "triton/codegen/transform/cts.h"
#include "triton/codegen/analysis/tiles.h"
#include "triton/ir/basic_block.h"
#include "triton/ir/type.h"
#include "triton/ir/value.h"
#include "triton/ir/function.h"
#include "triton/ir/instructions.h"
#include "triton/ir/utils.h"

namespace triton{
namespace codegen{
namespace analysis{

unsigned allocation::is_ld_padded(ir::value *x) {
  if(auto *trans = dynamic_cast<ir::trans_inst*>(x)){
    if(trans->get_perm()[0]->get_value() != 0)
      return 4;
  }
  auto order = tiles_->order(x);
  bool is_col_major = order[0] == 0;
  if(tiles_->hmma(x) == HMMA_A_ROW)
    return is_col_major ? 16 : 8;
  if(tiles_->hmma(x) == HMMA_A_COL)
    return is_col_major ? 8 : 16;
  if(tiles_->hmma(x) == HMMA_B_COL)
    return is_col_major ? 16 : 8;
  if(tiles_->hmma(x) == HMMA_B_ROW)
    return is_col_major ? 8 : 16;
  if(auto* phi = dynamic_cast<ir::phi_node*>(x)) {
    unsigned result = 0;
    for(unsigned i = 0; i < phi->get_num_incoming(); i++)
      result = std::max(result, is_ld_padded(phi->get_incoming_value(i)));
    return result;
  }
  return 0;
}

unsigned allocation::num_bytes(ir::value *x) {
  if(auto *red = dynamic_cast<ir::reduce_inst*>(x)){
    unsigned num_bytes = x->get_type()->get_scalar_ty()->get_primitive_size_in_bits() / 8;
    size_t axis = red->get_axis();
    ir::value *op = red->get_operand(0);
    auto shapes = op->get_type()->get_tile_shapes();
    shapes.erase(shapes.begin() + axis);
    size_t num_elements = 1;
    for(auto x: shapes)
      num_elements *= x;
    size_t depth;
    if(tiles_->hmma(x))
      depth = tiles_->wpt(op, axis);
    else
      depth = tiles_->mts(op, axis);
    return num_elements * num_bytes * depth;
  }
  unsigned num_bytes = x->get_type()->get_primitive_size_in_bits() / 8;
  unsigned pad = is_ld_padded(x);
  if(pad > 0){
    unsigned ld = x->get_type()->get_tile_shapes()[tiles_->order(x)[0]];
    num_bytes += pad * num_bytes / ld;
  }
  if(liveness_->has_double(x))
    num_bytes *= 2;
  return num_bytes;
}

void allocation::run(ir::module &mod) {
  using std::max;
  using std::min;
  typedef std::multimap<unsigned, segment> triples_map_type;

  std::vector<ir::value *> I;
  for(auto x: liveness_->intervals())
    I.push_back(x.first);
  std::vector<ir::value *> J = I;

  triples_map_type H;
  H.insert({0, segment{0, INT_MAX}});

  std::vector<ir::value *> V;
  std::map<ir::value *, unsigned> starts;
  while(!J.empty()){
    auto h_it = H.begin();
    unsigned w = h_it->first;
    segment xh = h_it->second;
    H.erase(h_it);
    auto j_it = std::find_if(J.begin(), J.end(), [&](ir::value *JJ){
      segment xj = liveness_->get_interval(JJ);
      bool res = xj.intersect(xh);
      for(auto val: H)
        res = res && !val.second.intersect(xj);
      return res;
    });
    if(j_it != J.end()){
      unsigned size = num_bytes(*j_it);
      segment xj = liveness_->get_interval(*j_it);
      starts[*j_it] = w;
      H.insert({w + size, segment{max(xh.start, xj.start), min(xh.end, xj.end)}});
      if(xh.start < xj.start)
        H.insert({w, segment{xh.start, xj.end}});
      if(xj.end < xh.end)
        H.insert({w, segment{xj.start, xh.end}});
      V.push_back(*j_it);
      J.erase(j_it);
    }
  }

  // Build interference graph
  std::map<ir::value*, std::set<ir::value *>> interferences;
  for(ir::value *x: V)
  for(ir::value *y: V){
    if(x == y)
      continue;
    unsigned X0 = starts[x], Y0 = starts[y];
    unsigned NX = num_bytes(x);
    unsigned NY = num_bytes(y);
    segment XS = {X0, X0 + NX};
    segment YS = {Y0, Y0 + NY};
    if(liveness_->get_interval(x).intersect(liveness_->get_interval(y))
        && XS.intersect(YS))
      interferences[x].insert(y);
  }

  // Initialize colors
  std::map<ir::value *, int> colors;
  for(ir::value *X: V)
    colors[X] = (X==V[0])?0:-1;


  // First-fit graph coloring
  std::vector<bool> available(V.size());
  for(ir::value *x: V){
    // Non-neighboring colors are available
    std::fill(available.begin(), available.end(), true);
    for(ir::value *Y: interferences[x]){
      int color = colors[Y];
      if(color >= 0)
        available[color] = false;
    }
    // Assigns first available color
    auto It = std::find(available.begin(), available.end(), true);
    colors[x] = std::distance(available.begin(), It);
  }

  // Finalize allocation
  for(ir::value *x: V){
    unsigned Adj = 0;
    for(ir::value *y: interferences[x])
      Adj = std::max(Adj, starts[y] + num_bytes(y));
    offsets_[x] = starts[x] + colors[x] * Adj;
    if(liveness_->has_double(x)){
      auto info = liveness_->get_double(x);
      offsets_[info.latch] = offsets_[x] + num_bytes(x) / 2;
    }
  }

  // Save maximum size of induced memory space
  allocated_size_ = 0;
  for(auto &x: offsets_){
    allocated_size_ = std::max<size_t>(allocated_size_, x.second + num_bytes(x.first));
  }
}

}
}
}
