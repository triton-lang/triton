#ifndef ATIDLAS_TOOLS_SHARED_PTR_HPP
#define ATIDLAS_TOOLS_SHARED_PTR_HPP

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file tools/shared_ptr.hpp
    @brief Implementation of a shared pointer class (cf. std::shared_ptr, boost::shared_ptr). Will be used until C++11 is widely available.

    Contributed by Philippe Tillet.
*/

#include <cstdlib>
#include <algorithm>

namespace atidlas
{
namespace tools
{
namespace detail
{

  /** @brief Reference counting class for the shared_ptr implementation */
  class count
  {
  public:
    count(unsigned int val) : val_(val){ }
    void dec(){ --val_; }
    void inc(){ ++val_; }
    bool is_null(){ return val_ == 0; }
    unsigned int val(){ return val_; }
  private:
    unsigned int val_;
  };

  /** @brief Interface for the reference counter inside the shared_ptr */
  struct aux
  {
    detail::count count;

    aux() :count(1) {}
    virtual void destroy()=0;
    virtual ~aux() {}
  };

  /** @brief Implementation helper for the reference counting mechanism inside shared_ptr. */
  template<class U, class Deleter>
  struct auximpl: public detail::aux
  {
    U* p;
    Deleter d;

    auximpl(U* pu, Deleter x) :p(pu), d(x) {}
    virtual void destroy() { d(p); }
  };

  /** @brief Default deleter class for a pointer. The default is to just call 'delete' on the pointer. Provide your own implementations for 'delete[]' and 'free'. */
  template<class U>
  struct default_deleter
  {
    void operator()(U* p) const { delete p; }
  };

}

class shared_ptr_base
{
protected:
  detail::aux* pa;
public:
  unsigned int count() { return pa->count.val(); }
};

/** @brief A shared pointer class similar to boost::shared_ptr. Reimplemented in order to avoid a Boost-dependency. Will be replaced by std::shared_ptr as soon as C++11 is widely available. */
template<class T>
class shared_ptr : public shared_ptr_base
{
  template<class U>
  friend class shared_ptr;

  detail::aux* pa;
  T* pt;

public:

  shared_ptr() :pa(NULL), pt(NULL) {}

  template<class U, class Deleter>
  shared_ptr(U* pu, Deleter d) : pa(new detail::auximpl<U, Deleter>(pu, d)), pt(pu) {}

  template<class U>
  explicit shared_ptr(U* pu) : pa(new detail::auximpl<U, detail::default_deleter<U> >(pu, detail::default_deleter<U>())), pt(pu) {}

  template<class U>
  shared_ptr(const shared_ptr<U>& s) :pa(s.pa), pt(s.pt)  { inc(); }

  shared_ptr(const shared_ptr& s) :pa(s.pa), pt(s.pt)  { inc(); }
  ~shared_ptr() { dec(); }

  T* get() const {  return pt; }
  T* operator->() const {  return pt; }
  T& operator*() const { return *pt; }

  void reset() { shared_ptr<T>().swap(*this); }
  void reset(T * ptr) { shared_ptr<T>(ptr).swap(*this); }

  void swap(shared_ptr<T> & other)
  {
    std::swap(pt,other.pt);
    std::swap(pa, other.pa);
  }

  shared_ptr& operator=(const shared_ptr& s)
  {
    if (this!=&s)
    {
      dec();
      pa = s.pa;
      pt = s.pt;
      inc();
    }
    return *this;
  }

  void inc()
  {
    if (pa) pa->count.inc();
  }

  void dec()
  {
    if (pa)
    {
      pa->count.dec();
      if (pa->count.is_null())
      {
        pa->destroy();
        delete pa;
        pa = NULL;
      }
    }
  }
};

}
}

#endif
