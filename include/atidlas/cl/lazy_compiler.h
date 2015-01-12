#ifndef ATIDLAS_CL_LAZY_COMPILER_H
#define ATIDLAS_CL_LAZY_COMPILER_H

#include "atidlas/cl/cl.hpp"
#include "atidlas/cl/program_map.h"

namespace atidlas
{

namespace cl
{

class lazy_compiler
{
public:
  lazy_compiler(cl::Context const & ctx, std::string const & name, std::string const & src, bool force_recompilation);
  lazy_compiler(cl::Context const & ctx, std::string const & name, bool force_recompilation);
  void add(std::string const & src);
  std::string const & src() const;
  cl::Program & program();

private:
  cl::Context context_;
  cl::Program program_;
  std::string pname_;
  std::string src_;
  bool force_recompilation_;
};

}

}
#endif
