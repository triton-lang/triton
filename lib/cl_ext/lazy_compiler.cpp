#include "atidlas/cl_ext/lazy_compiler.h"

namespace atidlas
{

namespace cl_ext
{

lazy_compiler::lazy_compiler(cl::Context const & ctx, std::string const & name, std::string const & src, bool force_recompilation) :
  context_(ctx), pname_(name), src_(src), force_recompilation_(force_recompilation)
{ }

lazy_compiler::lazy_compiler(cl::Context const & ctx, std::string const & name, bool force_recompilation) :
  context_(ctx), pname_(name), force_recompilation_(force_recompilation)
{ }

void lazy_compiler::add(std::string const & src)
{  src_+=src; }

std::string const & lazy_compiler::src() const
{ return src_;  }

cl::Program & lazy_compiler::program()
{
  if(program_()==0)
  {
    if(force_recompilation_)
      pmap.erase(context_, pname_);
    program_ = pmap.add(context_, pname_, src_);
  }
  return program_;
}

}

}
