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
class ISAACAPI CommandQueue
{
public:
  CommandQueue(cl::CommandQueue const & queue);
  CommandQueue(Context const & context, Device const & device, cl_command_queue_properties properties = 0);
  Context const & context() const;
  Device const & device() const;
  void synchronize();
  Event enqueue(Kernel const & kernel, NDRange global, driver::NDRange local, std::vector<Event> const *);
  void write(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void const* ptr);
  void read(Buffer const & buffer, bool blocking, std::size_t offset, std::size_t size, void* ptr);
  bool operator==(CommandQueue const & other) const;
  bool operator<(CommandQueue const & other) const;
  HANDLE_TYPE(cl::CommandQueue, CUstream)& handle();
private:
  backend_type backend_;
  Context context_;
  Device device_;
  HANDLE_TYPE(cl::CommandQueue, CUstream) h_;
};


}

}

#endif
