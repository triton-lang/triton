#ifndef ISAAC_DRIVER_PROGRAM_H
#define ISAAC_DRIVER_PROGRAM_H

#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/context.h"
#include "isaac/driver/handle.h"

namespace isaac
{

namespace driver
{

class Context;

class ISAACAPI Program
{
  friend class Kernel;
public:
  Program(Context const & context, std::string const & source);
  Context const & context() const;
private:
  backend_type backend_;
  Context context_;
  std::string source_;
  HANDLE_TYPE(cl_program, CUmodule) h_;
};

}

}

#endif
