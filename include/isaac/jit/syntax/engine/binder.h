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

#ifndef ISAAC_BACKEND_BINDER_H
#define ISAAC_BACKEND_BINDER_H

#include <map>
#include "isaac/driver/buffer.h"
#include "isaac/jit/syntax/expression/expression.h"

namespace isaac
{

class array_base;


class symbolic_binder
{
  class cmp
  {
  public:
    cmp(driver::backend_type backend) : backend_(backend) {}

    bool operator()(handle_t const & x, handle_t const & y) const
    {
      if(backend_==driver::OPENCL)
        return x.cl < y.cl;
      else
        return x.cu < y.cu;
    }

  private:
    driver::backend_type backend_;
  };

public:
  symbolic_binder(driver::backend_type backend);
  virtual ~symbolic_binder();
  virtual bool bind(handle_t const &, bool) = 0;
  virtual unsigned int get(handle_t const &, bool) = 0;
  unsigned int get();
protected:
  unsigned int current_arg_;
  std::map<handle_t,unsigned int, cmp> memory;
};


class bind_sequential : public symbolic_binder
{
public:
  bind_sequential(driver::backend_type backend);
  bool bind(handle_t const & a, bool);
  unsigned int get(handle_t const & a, bool);
};

class bind_independent : public symbolic_binder
{
public:
  bind_independent(driver::backend_type backend);
  bool bind(handle_t const & a, bool);
  unsigned int get(const handle_t &a, bool);
};

}

#endif
