#pragma once

#ifndef _TRITON_TOOLS_BENCH_H_
#define _TRITON_TOOLS_BENCH_H_

#include <chrono>
#include <functional>
#include <algorithm>
#include "triton/driver/device.h"
#include "triton/driver/stream.h"

namespace triton{
namespace tools{

class timer{
    typedef std::chrono::high_resolution_clock high_resolution_clock;
    typedef std::chrono::nanoseconds nanoseconds;

public:
    explicit timer(bool run = false)
    { if (run) start(); }

    void start()
    { _start = high_resolution_clock::now(); }

    nanoseconds get() const
    { return std::chrono::duration_cast<nanoseconds>(high_resolution_clock::now() - _start); }

private:
    high_resolution_clock::time_point _start;
};

inline double bench(std::function<void()> const & op, driver::stream * stream, bool normalize = false)
{
//  const driver::device * device = stream->context()->device();
  timer tmr;
  std::vector<size_t> times;
  double total_time = 0;
  op();
  stream->synchronize();
  tmr.start();
  size_t repeat = 100;
  for(size_t i = 0; i < repeat; i++){
    op();
  }
  stream->synchronize();
  return (float)tmr.get().count() / repeat;

//  return *std::min_element(times.begin(), times.end());
}

}
}

#endif
