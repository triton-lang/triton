#ifndef TDL_INCLUDE_CODEGEN_OPTIMIZE_REORDER_H
#define TDL_INCLUDE_CODEGEN_OPTIMIZE_REORDER_H

#include <map>
#include <set>
#include <vector>

namespace triton {

namespace ir {
  class module;
  class value;
  class io_inst;
  class instruction;
  class builder;
}

namespace codegen{

namespace analysis{
  class align;
  class layouts;
  class cts;
}

namespace transform{

class coalesce {
private:
  void extract_io_use(ir::value *v, std::set<ir::io_inst*>& result);
  void extract_ld(ir::io_inst *i, std::map<int, std::vector<triton::ir::io_inst *> > &result);
  ir::value* rematerialize(ir::value *v, ir::builder& builder, std::map<ir::value*, ir::value*>& seen);

public:
  coalesce(analysis::align* align, triton::codegen::analysis::layouts *layouts);
  void run(ir::module &mod);

private:
  analysis::align* align_;
  analysis::layouts* layout_;
};

}
}
}

#endif
