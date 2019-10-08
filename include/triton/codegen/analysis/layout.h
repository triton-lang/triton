#ifndef _TRITON_CODEGEN_ANALYSIS_GRID_H_
#define _TRITON_CODEGEN_ANALYSIS_GRID_H_

#include <map>
#include <set>
#include <vector>
#include <memory>
#include "triton/tools/graph.h"

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
  // graph creation
  void connect(ir::value *x, ir::value *y);
  void make_graph(ir::instruction *i);

public:
  // constructor
  layout(analysis::axes *axes);
  // accessors
  unsigned layout_of(ir::value *value) const;
  const std::vector<ir::value*>& values_of(unsigned id) const;
  size_t num_layouts() const;
  // execution
  void run(ir::module &mod);

private:
  analysis::axes* axes_;
  tools::graph<ir::value*> graph_;
  std::map<ir::value*, size_t> groups_;
  std::map<size_t, std::vector<ir::value*>> values_;
};

}
}

}

#endif
