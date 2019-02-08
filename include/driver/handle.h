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

#ifndef TDL_INCLUDE_DRIVER_HANDLE_H
#define TDL_INCLUDE_DRIVER_HANDLE_H

#include <memory>
#include <iostream>
#include <functional>
#include <type_traits>
#include "driver/dispatch.h"

namespace tdl
{

namespace driver
{

struct cu_event_t{
  operator bool() const { return first && second; }
  CUevent first;
  CUevent second;
};

struct cu_platform{
  cu_platform() : status_(dispatch::cuInit(0)) { }
  operator bool() const { return status_; }
private:
  CUresult status_;
};

template<class T, class CUType>
class HandleInterface{
public:
    //Accessors
    operator CUType() const { return *(((T*)this)->cu().h_); }
    //Comparison
    bool operator==(HandleInterface const & y) { return (CUType)(*this) == (CUType)(y); }
    bool operator!=(HandleInterface const & y) { return (CUType)(*this) != (CUType)(y); }
    bool operator<(HandleInterface const & y) { return (CUType)(*this) < (CUType)(y); }
};

template<class CUType>
class Handle{
public:
  template<class, class> friend class HandleInterface;
public:
  //Constructors
  Handle(CUType cu = CUType(), bool take_ownership = true);
  ~Handle();
  CUType& operator*() { return *h_; }
  CUType const & operator*() const { return *h_; }
  CUType* operator->() const { return h_.get(); }

protected:
  std::shared_ptr<CUType> h_;
  bool has_ownership_;
};

}
}

#endif
