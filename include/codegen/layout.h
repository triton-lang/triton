#ifndef TDL_INCLUDE_IR_CODEGEN_LAYOUT_H
#define TDL_INCLUDE_IR_CODEGEN_LAYOUT_H

#include <vector>
#include <map>

namespace tdl {

namespace ir {
  class function;
  class instruction;
  class value;
}

namespace codegen{

struct shared_view_info{
  ir::value *usr;
  bool has_dedicated_storage;
};

class layout {
private:
  typedef std::vector<shared_view_info> shared_view_val_t;

  void add_phi_nodes(ir::value *v);
  void add_shared_views(ir::value *v);

public:
  // accessors
  unsigned get_num_shared_views(ir::value *v);
  shared_view_info get_shared_view(ir::value *v, unsigned idx);

  // run
  bool run(ir::function &fn);

private:
  std::map<ir::value*, shared_view_val_t> shared_views_;
};


}
}

#endif
