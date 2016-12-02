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

#include <vector>
#include "isaac/random/rand.h"

namespace isaac
{
namespace random
{

template<typename T>
array rand(tuple const & shape, driver::Context const & context)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<> rng(0, 1);

  std::vector<T> data(prod(shape));
  for(size_t i = 0 ; i < data.size() ; ++i)
    data[i] = rng(mt);

  return array(shape, data, context);
}

#define INSTANTIATE(T) template array rand<T>(tuple const &, driver::Context const & context);
INSTANTIATE(half)
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

}
}
