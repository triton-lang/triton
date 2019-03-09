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
}

namespace codegen{

class tune {
  typedef std::pair<ir::value*, unsigned> node_t;
  typedef std::map <node_t, std::set<node_t>> graph_t;

private:
  void add_constraint(node_t x, node_t y);
  void init_c_phi(ir::instruction *i);
  void init_c_graph(ir::instruction *v);
  void connected_components(node_t x, const std::vector<ir::metaparameter *> mps, std::set<node_t> &nodes, graph_t &graph);
  void create_grids(std::vector<ir::instruction*> &grids, std::map<ir::metaparameter *, ir::instruction *> &references, ir::function *fn);


public:
  std::vector<ir::metaparameter *> get_params(ir::module& mod);
  std::map<std::string, ir::metaparameter *> get_params(ir::instruction* i);
  ir::metaparameter* get_param(ir::value *value, const std::string &key) { return params_[value][key]; }
  void copy(ir::value *dst, ir::value *src) { params_[dst] = params_[src]; }
  bool check_constraints(ir::module &fn, std::map<ir::value *, std::vector<std::string>> &errors);
  void run(ir::module &mod);
  ir::metaparameter* get_num_threads();
  ir::metaparameter* get_global_range_size(unsigned axis);

private:
  std::vector<unsigned*> pool_;
  graph_t dependencies_;
  std::set<node_t> nodes_;
  std::map<node_t, unsigned> static_params_;
  std::map<ir::value*, std::map<std::string, ir::metaparameter*>> params_;
  ir::metaparameter *num_threads_;
  std::vector<ir::metaparameter*> global_range_sizes_;
};


}
}

#endif
