#ifndef ISAAC_DRIVER_EVENT_H
#define ISAAC_DRIVER_EVENT_H

#include "isaac/driver/common.h"
#include "isaac/driver/handle.h"
namespace isaac
{

namespace driver
{

// Event
class Event
{
  friend class CommandQueue;
public:
  Event(backend_type backend);
  long elapsed_time() const;
private:
  backend_type backend_;
#ifdef ISAAC_WITH_CUDA
  typedef std::pair<CUevent, CUevent> cu_event_t;
#endif
  HANDLE_TYPE(cl::Event, cu_event_t) h_;
};

}

}

#endif
