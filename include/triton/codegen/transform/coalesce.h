#ifndef TDL_INCLUDE_CODEGEN_OPTIMIZE_REORDER_H
#define TDL_INCLUDE_CODEGEN_OPTIMIZE_REORDER_H

#include <map>
#include <vector>

namespace triton {

namespace ir {
  class module;
  class value;
}

namespace codegen{

namespace analysis{
  class align;
  class meminfo;
}

namespace transform{

class coalesce {
public:
  coalesce(analysis::align* algin, analysis::meminfo* mem);
  std::vector<unsigned> get_order(ir::value* v);
  void run(ir::module &mod);

private:
  analysis::align* align_;
  analysis::meminfo* mem_;
  std::map<ir::value*, std::vector<unsigned>> order_;
};

}
}
}

#endif
