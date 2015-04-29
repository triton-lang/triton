#ifndef ISAAC_BACKEND_BINDER_H
#define ISAAC_BACKEND_BINDER_H

#include <map>
#include "isaac/driver/buffer.h"

namespace isaac
{

enum binding_policy_t
{
  BIND_ALL_UNIQUE,
  BIND_TO_HANDLE
};


class symbolic_binder
{
public:
  symbolic_binder();
  virtual ~symbolic_binder();
  virtual bool bind(driver::Buffer const &) = 0;
  virtual unsigned int get(driver::Buffer const &) = 0;
  virtual unsigned int get();
protected:
  unsigned int current_arg_;
  std::map<driver::Buffer,unsigned int> memory;
};


class bind_to_handle : public symbolic_binder
{
public:
  bind_to_handle();
  bool bind(driver::Buffer const &);
  unsigned int get(driver::Buffer const &);
};

class bind_all_unique : public symbolic_binder
{
public:
  bind_all_unique();
  bool bind(driver::Buffer const &);
  unsigned int get(driver::Buffer const &);
};

}

#endif
