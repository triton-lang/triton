#ifndef TDL_INCLUDE_CODEGEN_SHARED_COPY_H
#define TDL_INCLUDE_CODEGEN_SHARED_COPY_H

#include <tuple>
#include <vector>

namespace tdl {

namespace ir {
  class module;
  class value;
  class builder;
  class basic_block;
}

namespace codegen{

class place_shared_copy {
private:
  typedef std::pair<unsigned, unsigned> interval_t;
  typedef std::vector<interval_t> interval_vec_t;

private:
  bool intersect(const interval_vec_t &I, interval_t i);
  void add_copies(ir::value *x, ir::builder &builder);

public:
  void run(ir::module &mod);
};


}
}

#endif
