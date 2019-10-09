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
class align;

enum layout_type_t {
  HMMA_884,
  SCANLINE
};

struct layout_t {
  layout_type_t type;
  std::vector<int> axes;
  std::vector<unsigned> shapes;
  std::vector<int> order;
  std::map<int, int> mts;
  std::map<int, int> nts;
  std::map<int, int> fpw;
  std::map<int, int> wpt;
};

class layout {
  typedef ir::value* node_t;
  typedef std::map <node_t, std::set<node_t>> graph_t;

private:
  // graph creation
  void connect(ir::value *x, ir::value *y);
  void make_graph(ir::instruction *i);

  void init_hmma_tile(layout_t& layout);
  void init_scanline_tile(layout_t &layout);

public:
  // constructor
  layout(analysis::axes *axes, analysis::align *align, size_t num_warps);
  // accessors
  unsigned layout_of(ir::value *value) const;
  const std::vector<ir::value*>& values_of(unsigned id) const;
  size_t num_layouts() const;
  const layout_t& get(ir::value *v) const;
  std::map<size_t, layout_t> &get_all();

  // execution
  void run(ir::module &mod);

private:
  analysis::axes* axes_;
  analysis::align* align_;
  size_t num_warps_;
  tools::graph<ir::value*> graph_;
  std::map<ir::value*, size_t> groups_;
  std::map<size_t, std::vector<ir::value*>> values_;
  std::map<size_t, layout_t> layouts_;
};

}
}

}

#endif
