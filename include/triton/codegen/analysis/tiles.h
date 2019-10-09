#ifndef _TRITON_CODEGEN_ANALYSIS_TILES_H_
#define _TRITON_CODEGEN_ANALYSIS_TILES_H_

#include <map>
#include <set>
#include <vector>
#include <memory>
#include "triton/codegen/analysis/layout.h"

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

class axes;
class layout;
class align;


class tiles {
  typedef std::map<ir::value*, std::map<int, int>> param_map_t;
private:
  void init_hmma_tile(const layout_t& layout);
  void init_scanline_tile(const layout_t& layout);
  bool is_trans(ir::value *i);

public:
  tiles(size_t num_warps, analysis::align* align, analysis::axes* axes, analysis::layout* layout);
  void run(ir::module &mod);
  int mts(ir::value *value, unsigned ax);
  int nts(ir::value *value, unsigned ax);
  int fpw(ir::value *value, unsigned ax);
  int wpt(ir::value *value, unsigned ax);


private:
  // dependencies
  analysis::align* align_;
  analysis::layout* layout_;
  analysis::axes* axes_;
  // number of warps
  size_t num_warps_;
  // tile properties
  std::map<int, int> fpw_;
  std::map<int, int> wpt_;
  std::map<int, int> mts_;
  std::map<int, int> nts_;
};


}
}
}

#endif
