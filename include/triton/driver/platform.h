/* Copyright 2015-2017 Philippe Tillet
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef TDL_INCLUDE_DRIVER_PLATFORM_H
#define TDL_INCLUDE_DRIVER_PLATFORM_H

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

class cu_platform: public platform
{
public:
  cu_platform(): platform("CUDA") { }
  std::string version() const;
  void devices(std::vector<driver::device*> &devices) const;

private:
  handle<CUPlatform> cu_;
};

class cl_platform: public platform
{
public:
  cl_platform(cl_platform_id cl): platform("OpenCL"), cl_(cl) { }
  std::string version() const;
  void devices(std::vector<driver::device*> &devices) const;

private:
  handle<cl_platform_id> cl_;
};

}

}

#endif
