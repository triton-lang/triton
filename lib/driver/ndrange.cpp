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

#include <algorithm>
#include "isaac/driver/ndrange.h"

namespace isaac
{

namespace driver
{

NDRange::NDRange(size_t size0)
{
    sizes_[0] = size0;
    sizes_[1] = 1;
    sizes_[2] = 1;
    dimension_ = 1;
}

NDRange::NDRange(size_t size0, size_t size1)
{
    sizes_[0] = size0;
    sizes_[1] = size1;
    sizes_[2] = 1;
    dimension_ = 2;
}

NDRange::NDRange(size_t size0, size_t size1, size_t size2)
{
    sizes_[0] = size0;
    sizes_[1] = size1;
    sizes_[2] = size2;
    dimension_ = 3;
}

int NDRange::dimension() const
{
 return dimension_;
}

NDRange::operator const size_t*() const
{
  return (const size_t*) sizes_;
}

}

}
