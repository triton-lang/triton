#include <sstream>
#include <unordered_map>
#include "triton/dnn/base.h"
#include "triton/runtime/jit.h"
#include "triton/tools/bench.hpp"

namespace triton{
namespace dnn{

namespace rt = triton::runtime;


void base::set_ld(const std::vector<int32_t>& shapes,
                  std::vector<int32_t>& ld) {
  size_t size = shapes.size();
  ld.resize(size);
  ld[size - 1] = 1;
  for(int i = size - 1; i >= 1; i--)
    ld[i - 1] = shapes[i] * ld[i];
}


base::base(const std::string& name)
  : name_(name) { }

std::vector<params_t> base::search_space() const {
  return {};
}

params_t base::heuristics() const {
  return *search_space().begin();
}

std::pair<base*, rt::jit*> base::get_profile_impl(driver::stream *stream, std::vector<driver::buffer *> args, autotuning_t autotune) {
  static std::unordered_map<base*, std::unique_ptr<rt::jit>, recompile_hash, recompile_equal> m_jit;
  driver::context* ctx = stream->context();
  rt::jit* jit;
  /* the current template has not already been compiled */
  if(m_jit.find(this) == m_jit.end()) {
    base* clone = this->clone();
    jit = m_jit.emplace(clone, std::unique_ptr<rt::jit>(new rt::jit(ctx, 8))).first->second.get();
    std::ostringstream oss;
    clone->triton_c_src(oss);
    std::string src = oss.str();
    auto benchmark = [&](triton::driver::kernel* kernel,
                         rt::launch_information info) {
      // launch info
      clone->init_impl(stream, (triton::driver::cu_module*)kernel->module(), info);
      clone->enqueue_impl(stream, kernel, args, info);
      stream->synchronize();
      double ts = triton::tools::bench([&](){ clone->enqueue_impl(stream, kernel, args, info); }, stream);
      clone->deinit_impl();
//      std::cout << ts * 1e-6 << std::endl;
      return num_flops() / ts * 1e-3;
    };
    // auto-tune and save result
    if(autotune == FULL_TUNING || autotune == PARTIAL_TUNING) {
      std::vector<params_t> space = {};
      if(autotune == PARTIAL_TUNING)
        space = search_space();
      rt::jit::tune_res_t best = jit->autotune(name_.c_str(), src.c_str(), benchmark, space);
      jit->add_module(name_.c_str(), src.c_str(), best.params);
    }
    else{
//      params_t params = heuristics();
//      params_t params = jit->get_valid(name_.c_str(), src.c_str());
//      params_t params = {4, 1, 32, 4, 1, 32, 4, 4, 4, 1, 1, 16, 32, 16, 4, 4, 4, 4, 1}; //NT
//      params_t params = {4, 1, 32, 4, 32, 4, 4, 4, 1, 1, 16, 32, 16, 1, 4, 4, 4, 4, 4, 1}; //NN
      params_t params = {4, 16, 4, 2, 16, 4, 8, 2, 2, 8, 2, 32, 8, 1}; // TT
      jit->add_module(name_.c_str(), src.c_str(), params);
    }
    triton::driver::kernel* kernel = jit->get_function(name_.c_str());
    rt::launch_information info = jit->get_launch_info(name_.c_str());
    clone->init_impl(stream, (triton::driver::cu_module*)kernel->module(), info);
  }
  /* retrieved compiled template */
  else {
    jit = m_jit.at(this).get();
  }
  auto it = m_jit.find(this);
  return {it->first, jit};
}

base* base::enqueue(driver::stream *stream, std::vector<driver::buffer *> args, autotuning_t autotune) {
  launch_context_t info = get_launch_context(stream, args, autotune);
  info.op->enqueue_impl(stream, info.kernel, args, info.info);
  return info.op;
}

launch_context_t base::get_launch_context(driver::stream *stream, std::vector<driver::buffer *> args, autotuning_t autotune) {
  std::pair<base*, rt::jit*> profile = get_profile_impl(stream, args, autotune);
  driver::kernel* kernel = profile.second->get_function(name_.c_str());
  rt::launch_information info = profile.second->get_launch_info(name_.c_str());
  return {profile.first, kernel, info};
}

}
}
