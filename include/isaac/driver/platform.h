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
  Platform(backend_type);
  Platform(cl_platform_id const &);
  std::string name() const;
  std::string version() const;
  void devices(std::vector<Device> &) const;
  cl_platform_id cl_id() const;
private:
  backend_type backend_;
  cl_platform_id cl_platform_;
};

}

}

#endif
