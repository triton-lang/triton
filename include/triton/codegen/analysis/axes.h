#ifndef _TRITON_CODEGEN_ANALYSIS_AXES_H_
#define _TRITON_CODEGEN_ANALYSIS_AXES_H_

#include <map>
#include <set>
#include <vector>
#include <memory>

namespace triton{

namespace ir{
  class value;
  class module;
  class instruction;
}

namespace codegen{
namespace analysis{

class axes {
  typedef std::pair<ir::value*, unsigned> node_t;
  typedef std::map <node_t, std::set<node_t>> graph_t;

private:
  void add_constraint(node_t x, node_t y);
  void init_c_phi(ir::instruction *i);
  void init_c_graph(ir::instruction *v);
  void connected_components(node_t x, std::set<node_t> &nodes, graph_t &graph, unsigned group_id);

public:
  axes();
  void run(ir::module &mod);
  unsigned get(ir::value *value, unsigned ax);
  bool has(ir::value *value, unsigned ax);

private:
  // constraints graph
  graph_t dependencies_;
  std::set<node_t> nodes_;
  // parameter groups
  std::map<ir::value*, std::map<unsigned, unsigned>> groups_;
};

}
}

}

#endif
