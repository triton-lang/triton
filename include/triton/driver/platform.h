#pragma once

#ifndef _TRITON_DRIVER_PLATFORM_H_
#define _TRITON_DRIVER_PLATFORM_H_

#include <vector>
#include <string>

#include "triton/driver/handle.h"

namespace triton
{

namespace driver
{

class device;

class platform
{
public:
  // Constructor
  platform(const std::string& name): name_(name){ }
  // Accessors
  std::string name() const { return name_; }
  // Virtual methods
  virtual std::string version() const = 0;
  virtual void devices(std::vector<driver::device *> &devices) const = 0;
private:
  std::string name_;
};

// Host
class host_platform: public platform
{
public:
  host_platform(): platform("CPU") { }
  std::string version() const;
  void devices(std::vector<driver::device*> &devices) const;
};

// CUDA
class cu_platform: public platform
{
public:
  cu_platform(): platform("CUDA") { }
  std::string version() const;
  void devices(std::vector<driver::device*> &devices) const;
private:
  handle<CUPlatform> cu_;
};

}

}

#endif
