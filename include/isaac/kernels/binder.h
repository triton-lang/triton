#ifndef ISAAC_BACKEND_BINDER_H
#define ISAAC_BACKEND_BINDER_H

#include <map>
#include "isaac/driver/buffer.h"

namespace isaac
{

enum binding_policy_t
{
  BIND_INDEPENDENT,
  BIND_SEQUENTIAL
};


class symbolic_binder
{
public:
  symbolic_binder();
  virtual ~symbolic_binder();
  virtual bool bind(driver::Buffer const &, bool) = 0;
  virtual unsigned int get(driver::Buffer const &, bool) = 0;
  unsigned int get();
protected:
  unsigned int current_arg_;
  std::map<driver::Buffer,unsigned int> memory;
};


class bind_sequential : public symbolic_binder
{
public:
  bind_sequential();
  bool bind(driver::Buffer const &, bool);
  unsigned int get(driver::Buffer const &, bool);
};

class bind_independent : public symbolic_binder
{
public:
  bind_independent();
  bool bind(driver::Buffer const &, bool);
  unsigned int get(driver::Buffer const &, bool);
};

}

#endif
