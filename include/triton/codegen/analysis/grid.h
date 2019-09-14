#ifndef TDL_INCLUDE_IR_CODEGEN_TUNE_H
#define TDL_INCLUDE_IR_CODEGEN_TUNE_H

#include <map>
#include <set>
#include <vector>
#include <memory>

namespace triton{

namespace ir{
  class value;
  class module;
  class instruction;
  class function;
  class metaparameter;
  class constant_int;
}

namespace codegen{

namespace transform{
class coalesce;
}

namespace analysis{

class grids {
  typedef std::pair<ir::value*, unsigned> node_t;
  typedef std::map <node_t, std::set<node_t>> graph_t;
  typedef std::shared_ptr<int> param_ptr_t;
  typedef std::map<ir::value*, std::map<int, param_ptr_t>> param_map_t;

public:
  enum fragment_t{
    STRIDED_SCAN,
    HMMA_FRAGMENT_C
  };

private:
  void add_constraint(node_t x, node_t y);
  void init_c_phi(ir::instruction *i);
  void init_c_graph(ir::instruction *v);
  fragment_t get_fragmentation_type(node_t x, graph_t &graph);
  void connected_components(node_t x, const std::vector<param_ptr_t>& params, const std::vector<param_map_t*>& maps, std::set<node_t> &nodes, graph_t &graph, unsigned group_id);
  void create_grids(std::vector<ir::value*> &grids,
                    std::map<unsigned, triton::ir::value *> &references,
                    ir::function *fn);


public:
  grids(size_t num_warps, transform::coalesce* reorder);
  fragment_t get_fragment(ir::value *value, unsigned ax);
  void copy(ir::value *dst, ir::value *src);
  void run(ir::module &mod);
  unsigned get_param_group(ir::value *value, unsigned ax);
  const std::vector<ir::value*> get_grids() const { return grids_; }
  int mts(ir::value *value, unsigned ax);
  int nts(ir::value *value, unsigned ax);
  int fpw(ir::value *value, unsigned ax);
  int wpt(ir::value *value, unsigned ax);

private:

  transform::coalesce* reorder_;
  // number of warps
  size_t num_warps_;
  // grids
  std::vector<ir::value*> grids_;
  // grid parameters
  param_map_t fpw_;
  param_map_t wpt_;
  param_map_t mts_;
  param_map_t nts_;
  // constraints graph
  graph_t dependencies_;
  std::set<node_t> nodes_;
  // fragments
  std::map<node_t, fragment_t> fragments_;
  // parameter groups
  std::map<ir::value*, std::map<unsigned, unsigned>> groups_;
};


}
}
}

#endif
