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

namespace transform{
class coalesce;
}

namespace analysis{

class axes;
class layout;

class tiles {
  typedef std::map<ir::value*, std::map<int, int>> param_map_t;
private:
  void init_hmma_tile(ir::value *i);
  void init_scanline_tile(ir::value *i);

public:
  tiles(size_t num_warps, transform::coalesce* coalesce, analysis::axes* axes, analysis::layout* layout);
  void run(ir::module &mod);
  bool hmma(ir::value *value);
  int mts(ir::value *value, unsigned ax);
  int nts(ir::value *value, unsigned ax);
  int fpw(ir::value *value, unsigned ax);
  int wpt(ir::value *value, unsigned ax);
  const std::map<int, ir::value*>& largest();

private:
  // dependencies
  analysis::layout* layout_;
  analysis::axes* axes_;
  transform::coalesce* coalesce_;
  // number of warps
  size_t num_warps_;
  // tile properties
  std::map<int, bool> hmma_;
  std::map<int, ir::value*> largest_;
  std::map<int, int> fpw_;
  std::map<int, int> wpt_;
  std::map<int, int> mts_;
  std::map<int, int> nts_;
};


}
}
}

#endif
