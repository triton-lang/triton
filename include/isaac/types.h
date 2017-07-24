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

#ifndef ISAAC_TYPES_H
#define ISAAC_TYPES_H

#include <algorithm>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include "isaac/defines.h"

namespace isaac
{

typedef long long int_t;
static const int_t start = 0;
static const int_t end = -1;
static const int_t newaxis = 1;

struct slice
{
//  slice(int_t _start) : start(_start), end(_start + 1), stride(1){}
  slice(int_t _start, int_t _end, int_t _stride = 1) : start(_start), end(_end), stride(_stride) { }

  int_t size(int_t bound) const
  {
    int_t effective_end = (end < 0)?bound - std::abs(end + 1):end;
    return (effective_end - start)/stride;
  }

  int_t start;
  int_t end;
  int_t stride;
};
static const slice all = slice{start, end, 1};

}
#endif
