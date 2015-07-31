#ifndef ISAAC_DRIVER_CONTEXT_H
#define ISAAC_DRIVER_CONTEXT_H

#include <map>
#include <memory>
#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/device.h"
#include "isaac/driver/handle.h"
#include "isaac/driver/program.h"

namespace isaac
{

namespace driver
{

class Program;

class ISAACAPI Context
{
  friend class Program;
  friend class CommandQueue;
  friend class Buffer;

  class ISAACAPI ProgramsHandler
  {
  public:
      ProgramsHandler(Context const & context);
      Program const & add(std::string const & name, std::string const & src);
      Program const * find(std::string const & name);
  private:
      Context const & context_;
      std::map<std::string, Program> programs_;
  };

public:
  explicit Context(cl_context const & context, bool take_ownership = true);
  explicit Context(Device const & device);
  Context(Context const &) = delete;

  backend_type backend() const;
  Device const & device() const;
  bool operator==(Context const &) const;
  bool operator<(Context const &) const;

  HANDLE_TYPE(cl_context, CUcontext) const & handle() const { return h_; }
  ProgramsHandler & programs() { return programs_; }

private:
  backend_type backend_;
  Device device_;

  std::string cache_path_;
  HANDLE_TYPE(cl_context, CUcontext) h_;
  ProgramsHandler programs_;
};

}
}

#endif
