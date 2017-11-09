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
  while(total_time*1e-9 < 1e-1){
    float norm = (float)device.current_sm_clock()/device.max_sm_clock();
    tmr.start();
    op();
    sync();
    times.push_back(norm*tmr.get().count());
    total_time+=times.back();
  }
  return min(times);
}


template<class T>
std::string str(T const & x){ return std::to_string(x); }

#endif
