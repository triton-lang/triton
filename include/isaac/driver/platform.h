#ifndef ISAAC_DRIVER_PLATFORM_H
#define ISAAC_DRIVER_PLATFORM_H

#include <vector>
#include <string>

#include "isaac/defines.h"
#include "isaac/driver/common.h"

namespace isaac
{

namespace driver
{

class Device;

class ISAACAPI Platform
{
private:
public:
#ifdef ISAAC_WITH_CUDA
  Platform(backend_type);
#endif
  Platform(cl_platform_id const &);
  std::string name() const;
  std::string version() const;
  void devices(std::vector<Device> &) const;
private:
  backend_type backend_;
  cl_platform_id cl_platform_;
};

}

}

#endif
