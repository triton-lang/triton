#ifndef TDL_INCLUDE_IR_CODEGEN_TUNE_H
#define TDL_INCLUDE_IR_CODEGEN_TUNE_H

#include <map>
#include <set>
#include <vector>

namespace tdl{

namespace ir{
  class value;
  class module;
  class instruction;
}

namespace codegen{

class tune {
  typedef std::pair<ir::value*, unsigned> node_t;
  typedef std::map <node_t, std::set<node_t>> graph_t;

private:
  void add_constraint(node_t x, node_t y);
  void init_c_phi(ir::instruction *i);
  void init_c_graph(ir::instruction *v);
  void connected_components(node_t x, const std::vector<unsigned*> vals, std::set<node_t> &nodes, graph_t &graph);


public:
  unsigned *get_param(ir::value *value);
  bool check_constraints(ir::module &fn, std::map<ir::value *, std::vector<std::string>> &errors);
  void run(ir::module &mod);

private:
  std::map<ir::value*, std::map<std::string, unsigned*>> params_;
  std::vector<unsigned*> pool_;
  graph_t dependencies_;
  std::set<node_t> nodes_;
};


}
}

#endif
