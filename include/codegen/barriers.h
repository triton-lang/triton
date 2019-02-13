#ifndef TDL_INCLUDE_CODEGEN_BARRIERS_H
#define TDL_INCLUDE_CODEGEN_BARRIERS_H

#include <tuple>
#include <vector>
#include <set>

namespace tdl {

namespace ir {
  class module;
  class basic_block;
  class instruction;
  class value;
  class builder;
}

namespace codegen{

class allocation;
class buffer_info_pass;

class barriers {
private:
  typedef std::pair<unsigned, unsigned> interval_t;
  typedef std::vector<interval_t> interval_vec_t;

private:
  void insert_barrier(ir::instruction *instr, ir::builder &builder);
  bool intersect(const interval_vec_t &X, interval_t x);
  bool intersect(const interval_vec_t &X, const interval_vec_t &Y);
  void add_reference(ir::value *v, interval_vec_t &res);
  void get_read_intervals(ir::instruction *i, interval_vec_t &res);
  void get_written_intervals(ir::instruction *i, interval_vec_t &res);
  void add(ir::basic_block *block, interval_vec_t &not_synced, std::set<ir::instruction *> &insert_pts);

public:
  barriers(allocation *alloc, buffer_info_pass *buffer_info): alloc_(alloc), buffer_info_(buffer_info) {}
  void run(ir::module &mod);

private:
  allocation *alloc_;
  buffer_info_pass *buffer_info_;
};


}
}

#endif
