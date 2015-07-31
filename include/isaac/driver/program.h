#ifndef ISAAC_DRIVER_PROGRAM_H
#define ISAAC_DRIVER_PROGRAM_H

#include <map>

#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/handle.h"

namespace isaac
{

namespace driver
{

class Context;
class Device;

class ISAACAPI Program
{
  friend class Kernel;
public:
  Program(Context const & context, std::string const & source);
  Context const & context() const;
private:
  backend_type backend_;
  Context const & context_;
  std::string source_;
  HANDLE_TYPE(cl_program, CUmodule) h_;
};

class ISAACAPI ProgramsHandler
{
public:
    static Program const & add(Context const & scontext, std::string const & name, std::string const & src);
    static Program const * find(Context const & context, std::string const & name);
private:
    static std::map<driver::Context const *, std::map<std::string, Program> > programs_;
};

}

}

#endif
