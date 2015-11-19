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

class array_base;

class symbolic_binder
{
public:
  symbolic_binder();
  virtual ~symbolic_binder();
  virtual bool bind(array_base const * a, bool) = 0;
  virtual unsigned int get(array_base const * a, bool) = 0;
  unsigned int get();
protected:
  unsigned int current_arg_;
  std::map<array_base const *,unsigned int> memory;
};


class bind_sequential : public symbolic_binder
{
public:
  bind_sequential();
  bool bind(array_base const * a, bool);
  unsigned int get(array_base const * a, bool);
};

class bind_independent : public symbolic_binder
{
public:
  bind_independent();
  bool bind(array_base const * a, bool);
  unsigned int get(array_base const * a, bool);
};

}

#endif
