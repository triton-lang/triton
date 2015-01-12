#ifndef ATIDLAS_CL_PROGRAM_MAP_H
#define ATIDLAS_CL_PROGRAM_MAP_H

#include <map>
#include "atidlas/cl/cl.hpp"

namespace atidlas
{

namespace cl
{

class program_map
{
  typedef std::map<cl::Context, std::map<std::string, cl::Program> > container_type;

public:
  program_map();
  cl::Program add(cl::Context & context, std::string const & pname, std::string const & source);
  cl::Program & at(cl::Context const & context, std::string const & key);
  void erase(cl::Context const & context, std::string const & pname);

private:
  std::map<cl_context, std::map<std::string, cl::Program> > data_;
  std::string cache_path_;
};

extern program_map pmap;

}

}

#endif
