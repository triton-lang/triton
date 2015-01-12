#ifndef ATIDLAS_BACKEND_BINDER_H
#define ATIDLAS_BACKEND_BINDER_H

#include <map>
#include "atidlas/cl/cl.hpp"

namespace atidlas
{

enum binding_policy_t
{
  BIND_ALL_UNIQUE,
  BIND_TO_HANDLE
};


class symbolic_binder
{
public:
  virtual ~symbolic_binder();
  virtual bool bind(cl::Buffer const * ph) = 0;
  virtual unsigned int get(cl::Buffer const * ph) = 0;
};


class bind_to_handle : public symbolic_binder
{
public:
  bind_to_handle();
  bool bind(cl::Buffer const * ph);
  unsigned int get(cl::Buffer const * ph);
private:
  unsigned int current_arg_;
  std::map<void*,unsigned int> memory;
};

class bind_all_unique : public symbolic_binder
{
public:
  bind_all_unique();
  bool bind(cl::Buffer const *);
  unsigned int get(cl::Buffer const *);
private:
  unsigned int current_arg_;
  std::map<void*,unsigned int> memory;
};

}

#endif
