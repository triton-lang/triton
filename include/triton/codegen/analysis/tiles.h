#ifndef _TRITON_CODEGEN_ANALYSIS_TILES_H_
#define _TRITON_CODEGEN_ANALYSIS_TILES_H_

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

namespace analysis{

class axes;
class layout;
class align;

enum layout_t {
  SCANLINE,
  HMMA_C,
  HMMA_A_COL,
  HMMA_A_ROW,
  HMMA_B_COL,
  HMMA_B_ROW
};

class tiles {
  typedef std::map<ir::value*, std::map<int, int>> param_map_t;
private:
  void init_hmma_tile(ir::value *i);
  void init_scanline_tile(ir::value *i);
  bool is_trans(ir::value *i);

public:
  tiles(size_t num_warps, analysis::align* align, analysis::axes* axes, analysis::layout* layout);
  void run(ir::module &mod);
  layout_t hmma(ir::value *value);
  int mts(ir::value *value, unsigned ax);
  int nts(ir::value *value, unsigned ax);
  int fpw(ir::value *value, unsigned ax);
  int wpt(ir::value *value, unsigned ax);
  std::vector<int> order(ir::value *v);
  const std::map<int, ir::value*>& largest();

private:
  // dependencies
  analysis::align* align_;
  analysis::layout* layout_;
  analysis::axes* axes_;
  // number of warps
  size_t num_warps_;
  // tile properties
  std::map<int, ir::value*> largest_;
  std::map<int, std::vector<int>> order_;
  std::map<int, layout_t> hmma_;
  std::map<int, int> fpw_;
  std::map<int, int> wpt_;
  std::map<int, int> mts_;
  std::map<int, int> nts_;
};


}
}
}

#endif
