#ifndef _VIENNACL_TOOLS_TIMER_HPP_
#define _VIENNACL_TOOLS_TIMER_HPP_

/* =========================================================================
   Copyright (c) 2010-2015, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */


/** @file   viennacl/tools/timer.hpp
    @brief  A simple, yet (mostly) sufficiently accurate timer for benchmarking and profiling. */

#include <iostream>


#ifdef _WIN32

#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max

namespace viennacl
{
namespace tools
{

/** @brief Simple timer class based on gettimeofday (POSIX) or QueryPerformanceCounter (Windows).
  *
  * Avoids messing with Boost and should be sufficient for benchmarking purposes.
  */
class timer
{
public:

  timer()
  {
    QueryPerformanceFrequency(&freq);
  }

  void start()
  {
    QueryPerformanceCounter((LARGE_INTEGER*) &start_time);
  }

  double get() const
  {
    LARGE_INTEGER  elapsed;
    QueryPerformanceCounter((LARGE_INTEGER*) &end_time);
    elapsed.QuadPart = end_time.QuadPart - start_time.QuadPart;
    return elapsed.QuadPart / static_cast<double>(freq.QuadPart);
  }


private:
  LARGE_INTEGER freq;
  LARGE_INTEGER start_time;
  LARGE_INTEGER end_time;
};

}

}

#else

#include <sys/time.h>

namespace viennacl
{
namespace tools
{

/** @brief Simple timer class based on gettimeofday (POSIX) or QueryPerformanceCounter (Windows).
  *
  * Avoids messing with Boost and should be sufficient for benchmarking purposes.
  */
class timer
{
public:

  timer() : ts(0)
  {}

  void start()
  {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    ts = static_cast<double>(tval.tv_sec * 1000000 + tval.tv_usec);
  }

  double get() const
  {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    double end_time = static_cast<double>(tval.tv_sec * 1000000 + tval.tv_usec);

    return static_cast<double>(end_time-ts) / 1000000.0;
  }

private:
  double ts;
};

}
}



#endif
#endif
