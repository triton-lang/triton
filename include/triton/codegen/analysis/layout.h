#ifndef _TRITON_CODEGEN_ANALYSIS_GRID_H_
#define _TRITON_CODEGEN_ANALYSIS_GRID_H_

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

class axes;

class layout {
  typedef ir::value* node_t;
  typedef std::map <node_t, std::set<node_t>> graph_t;

private:
  // connected components
  void connected_components(node_t x, std::set<node_t> &nodes, graph_t &graph, unsigned id);
  // list the axes of the given value
  std::set<int> axes_of(ir::value *value);

public:
  // constructor
  layout(analysis::axes *axes);
  // run the passes
  void run(ir::module &mod);
  // get the layout ID of the given value
  unsigned id(ir::value *value) const;
  // get the values associates with the given ID
  const std::vector<ir::value*>& values(unsigned id) const;
  // get number of groups
  size_t get_num_groups() const;

private:
  analysis::axes* axes_;
  graph_t dependencies_;
  std::set<node_t> nodes_;
  std::map<ir::value*, unsigned> groups_;
  std::map<unsigned, std::vector<ir::value*>> values_;
};

}
}

}

#endif
