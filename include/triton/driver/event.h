#pragma once

#ifndef _TRITON_DRIVER_EVENT_H_
#define _TRITON_DRIVER_EVENT_H_

#include "triton/driver/handle.h"

namespace triton
{

namespace driver
{

// event
class event
{
public:
  float elapsed_time() const;
  handle<cu_event_t> const & cu() const;

private:
  handle<cu_event_t> cu_;
};

}

}

#endif
