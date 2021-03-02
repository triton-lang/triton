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

typedef std::vector<size_t> grid_t;
typedef std::map<std::string, size_t> params_t;
template<typename T> inline T convert(const std::string& name);
template<> inline long convert<long>(const std::string& name) { return std::stol(name); }
template<> inline int convert<int>(const std::string& name) { return std::stoi(name); }

template<class T>
void add_arg(std::stringstream& ss, T arg) {
  ss.write((char*)&arg, sizeof(T));
}

enum asm_mode_t {
  ASM_LLIR,
  ASM_NV_PTX,
  ASM_NV_SASS
};

struct options_t {
  template<class T>
  T D(const std::string& name) const {
    return convert<T>(defines.at(name));
  }
  std::unordered_map<std::string, std::string> defines;
  int num_warps;
};


/* ------------------------- */

class kernel{
private:
  static std::string preheader();
  static arg_type convert(ir::type *ty);

public:
  kernel(const std::string& src, const options_t& opt, driver::device *device);
  void operator()(const std::string& args, driver::stream *stream, const std::vector<size_t>& grid) const;
  // getters
  const std::vector<arg_type>& get_sig() const { return sig_; }
  const std::vector<std::string>& get_arg_names() const { return arg_names_; }
  std::string get_asm(asm_mode_t mode);

private:
  void init_ir (const std::string &src);
  void init_ker();
  void init_sig();

public:
  const options_t opt;

private:
  driver::device* dev_;
  // signature
  std::vector<arg_type> sig_;
  std::vector<std::string> arg_names_;
  // triton context for parsing
  ir::context ctx_;
  // handles
  std::shared_ptr<ir::module> ir_;
  std::shared_ptr<driver::module> mod_;
  std::shared_ptr<driver::kernel> ker_;
  // shared mem
  size_t shared_mem_;
};

class function {
public:
  typedef std::function<grid_t(const options_t&)> grid_fn_ty;
  typedef std::pair<options_t, std::shared_ptr<kernel>> kernel_pair_t;
  typedef std::map<std::vector<uint64_t>, kernel*> cache_t;
  typedef std::vector<std::pair<std::map<std::string, std::string>, int>> autotune_vals_t;

private:
  static void do_loop_nest(std::vector<size_t> const & ranges,
                           std::function<void(std::vector<size_t> const &)> const & f);
public:
  function(const std::string& src, const options_t& opt, driver::device *device,
           const autotune_vals_t& autotune_vals = {}, const std::vector<std::string> &autotune_key = {});
  void operator()(const std::string& args, const grid_fn_ty& grid, driver::stream *stream);
  // auto-tuning
  cache_t::iterator find_in_cache(void* args, size_t args_size);
  kernel* autotune(const std::string& args, const grid_fn_ty& grid, driver::stream *stream);
  // getters
  const std::vector<kernel_pair_t> get_kernels() { return kernels_; }
  const std::vector<arg_type> get_signature() { return kernels_[0].second->get_sig(); }

private:
  void init_kernels(const std::string& src, const options_t& opt, const autotune_vals_t& autotune_vals, driver::device *device);

private:
  std::vector<kernel_pair_t> kernels_;
  std::map<std::vector<uint64_t>, kernel*> cache_;
  std::vector<int> key_idxs_;
  std::vector<int> arg_size_;
  std::vector<int> arg_off_;
};

}
}

#endif
