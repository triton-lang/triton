#ifndef ISAAC_DRIVER_PROGRAM_H
#define ISAAC_DRIVER_PROGRAM_H

#include <map>

#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/handle.h"
#include "isaac/driver/context.h"
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
DISABLE_MSVC_WARNING_C4251
  backend_type backend_;
  Context context_;
  std::string source_;
  HANDLE_TYPE(cl_program, CUmodule) h_;
RESTORE_MSVC_WARNING_C4251
};


}

}

#endif
