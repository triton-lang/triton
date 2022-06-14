#include <algorithm>
#include <climits>
#include "triton/codegen/analysis/layout.h"
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/analysis/liveness.h"
#include "triton/ir/utils.h"

namespace triton{
namespace codegen{
namespace analysis{


void allocation::run(ir::module &mod) {
  using std::max;
  using std::min;
  typedef std::multimap<unsigned, segment> triples_map_type;

  std::vector<shared_layout*> I;
  for(auto x: liveness_->get())
    I.push_back(x.first);
  std::vector<shared_layout*> J = I;

  triples_map_type H;
  H.insert({0, segment{0, INT_MAX}});

  std::vector<shared_layout*> V;
  std::map<shared_layout*, unsigned> starts;
  while(!J.empty()){
    auto h_it = H.begin();
    unsigned w = h_it->first;
    segment xh = h_it->second;
    H.erase(h_it);
    auto j_it = std::find_if(J.begin(), J.end(), [&](shared_layout* JJ){
      segment xj = liveness_->get(JJ);
      bool res = xj.intersect(xh);
      for(auto val: H)
        res = res && !val.second.intersect(xj);
      return res;
    });
    if(j_it != J.end()){
      unsigned size = (*j_it)->get_size();
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
  std::map<shared_layout*, std::set<shared_layout*>> interferences;
  for(shared_layout* x: V)
  for(shared_layout* y: V){
    if(x == y)
      continue;
    unsigned X0 = starts[x], Y0 = starts[y];
    unsigned NX = x->get_size();
    unsigned NY = y->get_size();
    segment XS = {X0, X0 + NX};
    segment YS = {Y0, Y0 + NY};
    if(liveness_->get(x).intersect(liveness_->get(y))
        && XS.intersect(YS))
      interferences[x].insert(y);
  }
  // Initialize colors
  std::map<shared_layout*, int> colors;
  for(shared_layout* X: V)
    colors[X] = (X==V[0])?0:-1;
  // First-fit graph coloring
  std::vector<bool> available(V.size());
  for(shared_layout* x: V){
    // Non-neighboring colors are available
    std::fill(available.begin(), available.end(), true);
    for(shared_layout* Y: interferences[x]){
      int color = colors[Y];
      if(color >= 0)
        available[color] = false;
    }
    // Assigns first available color
    auto It = std::find(available.begin(), available.end(), true);
    colors[x] = std::distance(available.begin(), It);
  }
  // Finalize allocation
  for(shared_layout* x: V){
    unsigned Adj = 0;
    for(shared_layout* y: interferences[x])
      Adj = std::max<unsigned>(Adj, starts[y] + y->get_size());
    offsets_[x] = starts[x] + colors[x] * Adj;
  }
  // Save maximum size of induced memory space
  allocated_size_ = 0;
  for(shared_layout* x: V){
    allocated_size_ = std::max<size_t>(allocated_size_, starts[x] + x->get_size());
    std::cout << "start: " << starts[x] << " | end: " << starts[x] + x->get_size() << std::endl;
  }
}

}
}
}
