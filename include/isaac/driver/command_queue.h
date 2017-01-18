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

#ifndef ISAAC_DRIVER_COMMAND_QUEUE_H
#define ISAAC_DRIVER_COMMAND_QUEUE_H

#include <map>
#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/context.h"
#include "isaac/driver/device.h"
#include "isaac/driver/handle.h"

namespace isaac
{

namespace driver
{

class Kernel;
class Event;
class NDRange;
class Buffer;

// Command Queue
class ISAACAPI CommandQueue: public has_handle_comparators<CommandQueue>
{
public:
  typedef Handle<cl_command_queue, CUstream> handle_type;

public:
  //Constructors
  CommandQueue(cl_command_queue const & queue, bool take_ownership = true);
  CommandQueue(Context const & context, Device const & device, cl_command_queue_properties properties = 0);
  //Accessors
  handle_type & handle();
  handle_type const & handle() const;
  backend_type backend() const;
  Context const & context() const;
  Device const & device() const;
  //Synchronize
  void synchronize();
  //Profiling
  void enable_profiling();
  void disable_profiling();
  //Enqueue calls
  void enqueue(Kernel const & kernel, NDRange global, driver::NDRange local, std::vector<Event> const *, Event *event);
  void write(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void const* ptr);
  void read(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void* ptr);

private:
  backend_type backend_;
  Context context_;
  Device device_;
  handle_type h_;
};


}

}

#endif
