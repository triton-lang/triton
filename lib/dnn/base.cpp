#include <sstream>
#include "triton/dnn/base.h"
#include "triton/runtime/jit.h"
#include "triton/tools/bench.hpp"

namespace triton{
namespace dnn{




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

void base::enqueue(driver::stream *stream, std::vector<driver::buffer *> args, bool autotune) {
  namespace rt = triton::runtime;
  static std::map<base*, std::unique_ptr<rt::jit>, cmp_recompile> m_jit;
  driver::context* ctx = stream->context();
  rt::jit* jit;
  /* the current template has not already been compiled */
  if(m_jit.find(this) == m_jit.end()) {
    base* clone = this->clone();
    jit = m_jit.emplace(clone, std::unique_ptr<rt::jit>(new rt::jit(ctx))).first->second.get();
    std::ostringstream oss;
    clone->triton_c_src(oss);
    std::string src = oss.str();
    auto benchmark = [&](triton::driver::kernel* kernel,
                         rt::launch_information info) {
      // launch info
      clone->init_impl(stream, (triton::driver::cu_module*)kernel->module());
      clone->enqueue_impl(stream, kernel, args, info);
      stream->synchronize();
      double ts = triton::tools::bench([&](){ clone->enqueue_impl(stream, kernel, args, info); },
                        [&](){ stream->synchronize(); }, ctx->device());
      clone->deinit_impl();
      return num_flops() / ts * 1e-3;
    };
    // auto-tune and save result
    if(autotune) {
      rt::jit::tune_res_t best = jit->autotune(name_.c_str(), src.c_str(), benchmark);
      jit->add_module(name_.c_str(), src.c_str(), best.params);
    }
    else {
      jit->add_module(name_.c_str(), src.c_str(), {16, 4, 128, 16, 4, 128, 2, 2, 2, 2, 8, 16, 8, 1});
    }
    triton::driver::kernel* kernel = jit->get_function(name_.c_str());
    clone->init_impl(stream, (triton::driver::cu_module*)kernel->module());
  }
  /* retrieved compiled template */
  else
    jit = m_jit.at(this).get();

  /* get launch parameters */
  driver::kernel* kernel = jit->get_function(name_.c_str());
  rt::launch_information info = jit->get_launch_info(name_.c_str());
  /* launch */
  auto it = m_jit.find(this);
  it->first->enqueue_impl(stream, kernel, args, info);
}

}
}
