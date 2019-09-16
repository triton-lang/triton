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
}

namespace codegen{

namespace analysis{
  class align;
  class layout;
  class meminfo;
}

namespace transform{

class coalesce {
private:
  void extract_io_use(ir::value *v, std::set<ir::io_inst*>& result);
  void extract_ld(ir::io_inst *i, std::map<int, std::vector<triton::ir::io_inst *> > &result);

public:
  coalesce(analysis::align* align, triton::codegen::analysis::layout *layouts, analysis::meminfo* mem);
  void run(ir::module &mod);

private:
  analysis::align* align_;
  analysis::layout* layout_;
  analysis::meminfo* mem_;
};

}
}
}

#endif
