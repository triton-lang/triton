#ifndef TDL_INCLUDE_CODEGEN_BARRIERS_H
#define TDL_INCLUDE_CODEGEN_BARRIERS_H

#include <vector>
#include <map>
#include <list>
#include <set>

namespace triton {

namespace ir {
  class module;
  class basic_block;
  class instruction;
  class masked_load_async_inst;
  class value;
  class builder;
}

namespace codegen{

namespace analysis{

class allocation;
class liveness;
class layouts;
class cts;

}

namespace transform{

class membar {
private:
  typedef std::pair<unsigned, unsigned> interval_t;
  typedef std::set<ir::value*> val_set_t;
  typedef std::vector<ir::value*> val_vec_t;

private:
  bool intersect(const val_set_t &X, const val_set_t &Y);
  int group_of(triton::ir::value *i, std::vector<triton::ir::value *> &async_write);
  val_set_t intersect_with(const val_set_t& as, const val_set_t& bs);
  void transfer(ir::basic_block *block, val_vec_t &async_write, val_set_t &sync_write, val_set_t &sync_read,
                std::set<triton::ir::value *> &safe_war, bool &inserted, ir::builder &builder);

public:
  membar(analysis::liveness *liveness, analysis::layouts *layouts, analysis::allocation *alloc):
    liveness_(liveness), layouts_(layouts), alloc_(alloc) {}
  void run(ir::module &mod);

private:
  analysis::liveness *liveness_;
  analysis::layouts *layouts_;
  analysis::allocation *alloc_;
};


}
}
}

#endif
