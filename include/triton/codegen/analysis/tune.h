#ifndef TDL_INCLUDE_IR_CODEGEN_TUNE_H
#define TDL_INCLUDE_IR_CODEGEN_TUNE_H

#include <map>
#include <set>
#include <vector>

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
namespace analysis{

class grids {
  typedef std::pair<ir::value*, unsigned> node_t;
  typedef std::map <node_t, std::set<node_t>> graph_t;

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
  void connected_components(node_t x, const std::vector<ir::metaparameter *> mps, const std::vector<std::string> prefixes, std::set<node_t> &nodes, graph_t &graph, unsigned group_id);
  void create_grids(std::vector<ir::value*> &grids,
                    std::map<unsigned, ir::value*> &references,
                    ir::function *fn);


public:
  grids(size_t num_warps);
  ir::metaparameter* get_param(ir::value *value, const std::string &key) { return params_[value][key]; }
  unsigned get_param_group(ir::value *value, unsigned ax);
  fragment_t get_fragment(ir::value *value, unsigned ax) { return fragments_.at({value, ax}); }
  void copy(ir::value *dst, ir::value *src);
  void run(ir::module &mod);
  unsigned get_num_threads();

private:
  std::vector<unsigned*> pool_;
  graph_t dependencies_;
  std::set<node_t> nodes_;
  std::map<node_t, fragment_t> fragments_;
  std::map<node_t, unsigned> static_params_;
  std::map<ir::value*, std::map<std::string, ir::metaparameter*>> params_;
  std::map<unsigned, ir::metaparameter*> global_range_sizes_;
  std::vector<ir::value*> grids_;
  std::map<ir::value*, std::map<unsigned, unsigned>> groups_;
  size_t num_warps_;
};


}
}
}

#endif
