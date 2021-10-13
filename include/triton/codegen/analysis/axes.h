#ifndef _TRITON_CODEGEN_ANALYSIS_AXES_H_
#define _TRITON_CODEGEN_ANALYSIS_AXES_H_

#include "triton/tools/graph.h"
#include <map>
#include <vector>

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

private:
  // update graph
  void update_graph_store(ir::instruction *i);
  void update_graph_reduce(ir::instruction *i);
  void update_graph_reshape(ir::instruction *i);
  void update_graph_trans(ir::instruction *i);
  void update_graph_broadcast(ir::instruction *i);
  void update_graph_dot(ir::instruction *i);
  void update_graph_elementwise(ir::instruction *i,
                                bool is_masked_load_async=false);
  void update_graph_no_edge(ir::instruction *i);
  void update_graph(ir::instruction *i);

public:
  axes();
  void run(ir::module &mod);
  // accessors
  int get(ir::value *value, unsigned dim);
  std::vector<int> get(ir::value *value);

private:
  tools::graph<node_t> graph_;
  std::map<node_t, size_t> axes_;
};

}
}

}

#endif
