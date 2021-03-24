#pragma once

#ifndef _TRITON_RUNTIME_FUNCTION_H_
#define _TRITON_RUNTIME_FUNCTION_H_

#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <memory>
#include <functional>
// codegen
#include "triton/ir/function.h"
#include "triton/ir/context.h"
#include "triton/runtime/arg.h"
#include "triton/runtime/error.h"

// driver forward declaration
namespace triton {
namespace driver{
  class module;
  class stream;
  class kernel;
  class context;
  class device;
}
}
// ir forward declaration
namespace triton{
namespace ir {
class module;
class function;
class context;
}
}

namespace triton{
namespace runtime{


/* ------------------------- */
/* Compilation options       */
/* ------------------------- */

struct options_t {
  template<class T>
  T D(const std::string& name) const {
    return std::stoi(defines.at(name));
  }
  std::unordered_map<std::string, std::string> defines;
  int num_warps;
};

/* ------------------------- */
/* Runtime arguments         */
/* ------------------------- */

enum arg_type {
  INT1_T,
  INT8_T,
  INT16_T,
  INT32_T,
  INT64_T,
  HALF_T,
  FLOAT_T,
  DOUBLE_T,
  BUFFER_T
};

inline size_t size_of(arg_type ty){
  switch(ty){
  case INT1_T  : return 1;
  case INT8_T  : return 1;
  case INT16_T : return 2;
  case INT32_T : return 4;
  case INT64_T : return 8;
  case HALF_T  : return 2;
  case FLOAT_T : return 4;
  case DOUBLE_T: return 8;
  case BUFFER_T: return 8;
  default: throw std::runtime_error("unknown type");
  }
}

template<class T>
void add_arg(std::stringstream& ss, T arg) {
  ss.write((char*)&arg, sizeof(T));
}


/* ------------------------- */
/* ------------------------- */

class kernel{
public:
  typedef std::vector<size_t> grid_t;

public:
  static std::shared_ptr<ir::module> src_to_ir(const std::string& src, const options_t& opt);
  static std::tuple<std::shared_ptr<driver::module>,
                    std::shared_ptr<driver::kernel>,
                    size_t> ir_to_bin(ir::module& ir, driver::device *dev, const options_t &opt);

public:
  kernel(const std::string& src, const options_t& opt, driver::device *device, const std::map<int, triton::ir::attribute> &attrs = {});
  void operator()(const std::string& args, driver::stream *stream, const grid_t& grid) const;
  std::string get_asm(const std::string &mode);

public:
  const options_t opt;

private:
  driver::device* dev_;
  // handles
  std::shared_ptr<ir::module> ir_;
  std::shared_ptr<driver::module> mod_;
  std::shared_ptr<driver::kernel> ker_;
  // shared mem
  size_t shared_mem_;
};

struct config {
  std::map<std::string, std::string> defines;
  int num_warps;
};

class function {
public:
  typedef std::function<kernel::grid_t(const options_t&)> grid_fn_ty;
  typedef std::pair<options_t, std::shared_ptr<kernel>> kernel_pair_t;
  typedef std::map<std::vector<uint64_t>, kernel*> cache_t;
  typedef std::vector<config> autotune_confs_t;

public:
  function(const std::string& src, const options_t& opt, driver::device *device,
           const std::vector<config>& tune_confs = {}, const std::vector<std::string> &tune_key = {});
  kernel* autotune(const std::string& args, const grid_fn_ty& grid, driver::stream *stream);
  void operator()(const std::string& args, const grid_fn_ty& grid, driver::stream *stream);
  const std::vector<arg_type> get_signature() { return sig_; }

private:
  std::map<std::vector<uint64_t>, std::vector<std::shared_ptr<kernel>>> kernels_;
  std::map<std::vector<uint64_t>, kernel*> cache_;
  std::vector<arg_type> sig_;
  std::vector<int> align_idxs_;
  std::vector<int> int_idxs_;
  std::vector<int> key_idxs_;
  std::vector<int> arg_size_;
  std::vector<int> arg_off_;
  std::vector<options_t> opts_;
  std::string src_;
  driver::device* device_;
};

}
}

#endif
