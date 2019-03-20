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
#include "triton/driver/dispatch.h"

namespace triton
{

namespace driver
{

struct cu_event_t{
  operator bool() const { return first && second; }
  CUevent first;
  CUevent second;
};

struct CUPlatform{
  CUPlatform() : status_(dispatch::cuInit(0)) { }
  operator bool() const { return status_; }
private:
  CUresult status_;
};

template<class T, class CUType>
class handle_interface{
public:
    //Accessors
    operator CUType() const { return *(((T*)this)->cu().h_); }
    //Comparison
    bool operator==(handle_interface const & y) { return (CUType)(*this) == (CUType)(y); }
    bool operator!=(handle_interface const & y) { return (CUType)(*this) != (CUType)(y); }
    bool operator<(handle_interface const & y) { return (CUType)(*this) < (CUType)(y); }
};

template<class CUType>
class handle{
public:
  template<class, class> friend class handle_interface;
public:
  //Constructors
  handle(CUType cu = CUType(), bool take_ownership = true);
  ~handle();
  CUType& operator*() { return *h_; }
  CUType const & operator*() const { return *h_; }
  CUType* operator->() const { return h_.get(); }

protected:
  std::shared_ptr<CUType> h_;
  bool has_ownership_;
};

template<class CUType, class CLType>
class polymorphic_resource {
public:
  polymorphic_resource(CUType cu, bool take_ownership): cu_(cu, take_ownership){}
  polymorphic_resource(CLType cl, bool take_ownership): cl_(cl, take_ownership){}
  virtual ~polymorphic_resource() { }

  handle<CUType> cu() { return cu_; }
  handle<CLType> cl() { return cl_; }
  const handle<CUType>& cu() const { return cu_; }
  const handle<CLType>& cl() const { return cl_; }

protected:
  handle<CLType> cl_;
  handle<CUType> cu_;
};

}
}

#endif
