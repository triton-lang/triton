#ifndef ISAAC_DRIVER_CONTEXT_H
#define ISAAC_DRIVER_CONTEXT_H

#include <map>
#include <memory>
#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/device.h"
#include "isaac/driver/handle.h"

namespace isaac
{

namespace driver
{

class ISAACAPI Context
{
  friend class Program;
  friend class CommandQueue;
  friend class Buffer;

public:
  explicit Context(cl::Context const & context);
  explicit Context(Device const & device);
  backend_type backend() const;
  Device const & device() const;
  bool operator==(Context const &) const;
  bool operator<(Context const &) const;
private:
  backend_type backend_;
  Device device_;

  std::string cache_path_;
  HANDLE_TYPE(cl::Context, CUcontext) h_;
};

}
}

#endif
