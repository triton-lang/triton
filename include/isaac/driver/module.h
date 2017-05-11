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

#ifndef ISAAC_DRIVER_MODULE_H
#define ISAAC_DRIVER_MODULE_H

#include <map>
#include "isaac/driver/handle.h"
#include "isaac/driver/context.h"

namespace isaac
{

namespace driver
{

class Context;
class Device;

class Module: public HandleInterface<Module, CUmodule>
{
public:
  Module(Context const & context, std::string const & source, bool is_ir = true);
  Context const & context() const;
  Handle<CUmodule> const & cu() const;

private:
  Handle<CUmodule> cu_;
  Context context_;
  std::string source_;
};


}

}

#endif
