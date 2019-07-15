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

#ifndef TDL_INCLUDE_DNN_BASE_H
#define TDL_INCLUDE_DNN_BASE_H

#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"
#include "triton/runtime/launch_info.h"

namespace triton{
namespace dnn{



class base {
  friend class cmp_recompile;

protected:
  // leading dimensions
  static void set_ld(const std::vector<int32_t>& shapes,
                     std::vector<int32_t>& ld);

private:
  // initialize
  virtual void init_impl(driver::stream *, driver::cu_module *){ }
  // enqueue
  virtual void enqueue_impl(driver::stream *stream, driver::kernel *kernel,
                    std::vector<driver::buffer*> args,
                    triton::runtime::launch_information info) = 0;
  // number of flops
  virtual size_t num_flops() const = 0;
  // comparison for maps
  virtual bool operator<(const base& other) const = 0;

public:
  // constructor
  base(const std::string& name);
  // triton-c source
  virtual void triton_c_src(std::ostream &os) const = 0;
  // clone
  virtual base* clone() const = 0;
  // enqueue
  void enqueue(driver::stream* stream, std::vector<driver::buffer*> args, bool autotune = false);

private:
  std::string name_;
};

struct cmp_recompile{
  bool operator()(base* x, base* y) const{
    return *x < *y;
  }
};

}
}

#endif
