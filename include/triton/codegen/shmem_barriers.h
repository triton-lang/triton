#ifndef TDL_INCLUDE_CODEGEN_BARRIERS_H
#define TDL_INCLUDE_CODEGEN_BARRIERS_H

#include <tuple>
#include <vector>
#include <set>

namespace triton {

namespace ir {
  class module;
  class basic_block;
  class instruction;
  class value;
  class builder;
}

namespace codegen{

class shmem_allocation;
class shmem_info;

class shmem_barriers {
private:
  typedef std::pair<unsigned, unsigned> interval_t;
  typedef std::vector<interval_t> interval_vec_t;

private:
  interval_vec_t join(const std::vector<interval_vec_t>& intervals);
  void insert_barrier(ir::instruction *instr, ir::builder &builder);
  bool intersect(const interval_vec_t &X, interval_t x);
  bool intersect(const interval_vec_t &X, const interval_vec_t &Y);
  void add_reference(ir::value *v, interval_vec_t &res);
  void get_read_intervals(ir::instruction *i, interval_vec_t &res);
  void get_written_intervals(ir::instruction *i, interval_vec_t &res);
  std::pair<interval_vec_t, interval_vec_t> transfer(ir::basic_block *block, const interval_vec_t &written_to, const interval_vec_t &read_from, std::set<ir::instruction *> &insert_loc);

public:
  shmem_barriers(shmem_allocation *alloc, shmem_info *buffer_info): alloc_(alloc), buffer_info_(buffer_info) {}
  void run(ir::module &mod);

private:
  shmem_allocation *alloc_;
  shmem_info *buffer_info_;
};


}
}

#endif
