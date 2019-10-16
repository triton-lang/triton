#include <algorithm>
#include <climits>
#include "triton/codegen/analysis/layout.h"
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/analysis/liveness.h"
#include "triton/codegen/transform/cts.h"
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

  std::vector<layout_t*> I;
  for(auto x: liveness_->get())
    I.push_back(x.first);
  std::vector<layout_t*> J = I;

  triples_map_type H;
  H.insert({0, segment{0, INT_MAX}});

  std::vector<layout_t*> V;
  std::map<layout_t*, unsigned> starts;
  while(!J.empty()){
    auto h_it = H.begin();
    unsigned w = h_it->first;
    segment xh = h_it->second;
    H.erase(h_it);
    auto j_it = std::find_if(J.begin(), J.end(), [&](layout_t* JJ){
      segment xj = liveness_->get(JJ);
      bool res = xj.intersect(xh);
      for(auto val: H)
        res = res && !val.second.intersect(xj);
      return res;
    });
    if(j_it != J.end()){
      unsigned size = (*j_it)->size;
      segment xj = liveness_->get(*j_it);
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
  std::map<layout_t*, std::set<layout_t*>> interferences;
  for(layout_t* x: V)
  for(layout_t* y: V){
    if(x->id == y->id)
      continue;
    unsigned X0 = starts[x], Y0 = starts[y];
    unsigned NX = x->size;
    unsigned NY = y->size;
    segment XS = {X0, X0 + NX};
    segment YS = {Y0, Y0 + NY};
    if(liveness_->get(x).intersect(liveness_->get(y))
        && XS.intersect(YS))
      interferences[x].insert(y);
  }

  // Initialize colors
  std::map<layout_t*, int> colors;
  for(layout_t* X: V)
    colors[X] = (X->id==V[0]->id)?0:-1;


  // First-fit graph coloring
  std::vector<bool> available(V.size());
  for(layout_t* x: V){
    // Non-neighboring colors are available
    std::fill(available.begin(), available.end(), true);
    for(layout_t* Y: interferences[x]){
      int color = colors[Y];
      if(color >= 0)
        available[color] = false;
    }
    // Assigns first available color
    auto It = std::find(available.begin(), available.end(), true);
    colors[x] = std::distance(available.begin(), It);
  }

  // Finalize allocation
  for(layout_t* x: V){
    unsigned Adj = 0;
    for(layout_t* y: interferences[x])
      Adj = std::max<unsigned>(Adj, starts[y] + y->size);
    offsets_[x] = starts[x] + colors[x] * Adj;
  }


  // Save maximum size of induced memory space
  allocated_size_ = 0;
  for(layout_t* x: V)
    allocated_size_ = std::max<size_t>(allocated_size_, starts[x] + x->size);
}

}
}
}
