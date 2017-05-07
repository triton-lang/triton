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

#ifndef BENCH_HPP
#define BENCH_HPP

#include <chrono>
#include <algorithm>
#include <isaac/driver/device.h>
#include <iostream>
#include <iomanip>
#include <iterator>

class Timer
{
    typedef std::chrono::high_resolution_clock high_resolution_clock;
    typedef std::chrono::nanoseconds nanoseconds;

public:
    explicit Timer(bool run = false)
    { if (run) start(); }

    void start()
    { _start = high_resolution_clock::now(); }

    nanoseconds get() const
    { return std::chrono::duration_cast<nanoseconds>(high_resolution_clock::now() - _start); }

private:
    high_resolution_clock::time_point _start;
};

template<class T>
T min(std::vector<T> x)
{ return *std::min_element(x.begin(), x.end()); }


template<class OP, class SYNC>
double bench(OP const & op, SYNC const & sync, isaac::driver::Device const & device)
{
  Timer tmr;
  std::vector<size_t> times;
  double total_time = 0;
  op();
  sync();
  while(total_time*1e-9 < 1e-3){
    float norm = (float)device.current_sm_clock()/device.max_sm_clock();
    tmr.start();
    op();
    sync();
    times.push_back(norm*tmr.get().count());
    total_time+=times.back();
  }
  return min(times);
}



enum Code {
  RESET = 0,
  BOLD = 1,
  ITALIC = 3,
  FG_RED = 31,
  FG_GREEN = 32,
  FG_YELLOW = 33,
  FG_BLUE = 34,
  FG_MAGENTA = 35,
  FG_CYAN = 36,
  FG_LIGHT_GRAY = 37,
  FG_DARK_GRAY = 90,
  FG_LIGHT_RED = 91,
  FG_LIGHT_GREEN = 92,
  FG_LIGHT_YELLOW = 93,
  FG_LIGHT_BLUE = 94,
  FG_LIGHT_MAGENTA = 95,
  FG_LIGHT_CYAN = 96,
  FG_WHITE = 97
};
class color_stream {
    Code code;
public:
    color_stream(Code pCode) : code(pCode) {}
    friend std::ostream&
    operator<<(std::ostream& os, const color_stream& mod) {
        return os << "\033[" << mod.code << "m";
    }
};

template<class T>
std::string str(T const & x){ return std::to_string(x); }

#endif
