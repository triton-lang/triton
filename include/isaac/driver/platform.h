#ifndef ISAAC_DRIVER_PLATFORM_H
#define ISAAC_DRIVER_PLATFORM_H

#include "isaac/driver/common.h"

namespace isaac
{

namespace driver
{

class Device;

class Platform
{
private:
public:
#ifdef ISAAC_WITH_CUDA
  Platform(backend_type);
#endif
  Platform(cl::Platform const &);
  std::string name() const;
  std::string version() const;
  std::vector<Device> devices() const;
  static std::vector<Platform> get();

private:
  backend_type backend_;
  cl::Platform cl_platform_;
};

}

}

#endif
