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


void allocation::run(ir::module &mod) {
  using std::max;
  using std::min;
  typedef std::multimap<unsigned, segment> triples_map_type;

  std::vector<buffer_t*> I;
  for(auto x: liveness_->intervals())
    I.push_back(x.first);
  std::vector<buffer_t*> J = I;

  triples_map_type H;
  H.insert({0, segment{0, INT_MAX}});

  std::vector<buffer_t*> V;
  std::map<buffer_t*, unsigned> starts;
  while(!J.empty()){
    auto h_it = H.begin();
    unsigned w = h_it->first;
    segment xh = h_it->second;
    H.erase(h_it);
    auto j_it = std::find_if(J.begin(), J.end(), [&](buffer_t* JJ){
      segment xj = liveness_->get_interval(JJ);
      bool res = xj.intersect(xh);
      for(auto val: H)
        res = res && !val.second.intersect(xj);
      return res;
    });
    if(j_it != J.end()){
      unsigned size = (*j_it)->size;
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
  std::map<buffer_t*, std::set<buffer_t*>> interferences;
  for(buffer_t* x: V)
  for(buffer_t* y: V){
    if(x->id == y->id)
      continue;
    unsigned X0 = starts[x], Y0 = starts[y];
    unsigned NX = x->size;
    unsigned NY = y->size;
    segment XS = {X0, X0 + NX};
    segment YS = {Y0, Y0 + NY};
    if(liveness_->get_interval(x).intersect(liveness_->get_interval(y))
        && XS.intersect(YS))
      interferences[x].insert(y);
  }

  // Initialize colors
  std::map<buffer_t*, int> colors;
  for(buffer_t* X: V)
    colors[X] = (X->id==V[0]->id)?0:-1;


  // First-fit graph coloring
  std::vector<bool> available(V.size());
  for(buffer_t* x: V){
    // Non-neighboring colors are available
    std::fill(available.begin(), available.end(), true);
    for(buffer_t* Y: interferences[x]){
      int color = colors[Y];
      if(color >= 0)
        available[color] = false;
    }
    // Assigns first available color
    auto It = std::find(available.begin(), available.end(), true);
    colors[x] = std::distance(available.begin(), It);
  }

  // Finalize allocation
  for(buffer_t* x: V){
    unsigned Adj = 0;
    for(buffer_t* y: interferences[x])
      Adj = std::max<unsigned>(Adj, starts[y] + y->size);
    // create offsets
    for(ir::value *v: liveness_->get_values(x)){
      offsets_[v] = starts[x] + colors[x] * Adj;
      if(liveness_->has_double(v)){
        auto info = liveness_->get_double(v);
        offsets_[info.latch] = offsets_[v] + x->size / 2;
      }
    }
  }

  // Save maximum size of induced memory space
  allocated_size_ = 0;
  for(buffer_t* x: V)
    allocated_size_ = std::max<size_t>(allocated_size_, starts[x] + x->size);
}

}
}
}
