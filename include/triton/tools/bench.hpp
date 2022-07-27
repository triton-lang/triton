#pragma once

#ifndef _TRITON_TOOLS_BENCH_H_
#define _TRITON_TOOLS_BENCH_H_

#include "triton/driver/device.h"
#include "triton/driver/stream.h"
#include <algorithm>
#include <chrono>
#include <functional>

namespace triton {
namespace tools {

class timer {
  typedef std::chrono::high_resolution_clock high_resolution_clock;
  typedef std::chrono::nanoseconds nanoseconds;

public:
  explicit timer(bool run = false) {
    if (run)
      start();
  }

  void start() { _start = high_resolution_clock::now(); }

  nanoseconds get() const {
    return std::chrono::duration_cast<nanoseconds>(
        high_resolution_clock::now() - _start);
  }

private:
  high_resolution_clock::time_point _start;
};

inline double bench(std::function<void()> const &op, driver::stream *stream,
                    size_t warmup = 10, size_t repeat = 200) {
  timer tmr;
  std::vector<size_t> times;
  double total_time = 0;
  for (size_t i = 0; i < warmup; i++)
    op();
  stream->synchronize();
  tmr.start();
  for (size_t i = 0; i < repeat; i++) {
    op();
  }
  stream->synchronize();
  return (float)tmr.get().count() / repeat;

  //  return *std::min_element(times.begin(), times.end());
}

} // namespace tools
} // namespace triton

#endif
