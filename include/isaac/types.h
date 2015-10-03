#ifndef ISAAC_TYPES_H
#define ISAAC_TYPES_H

#include <list>
#include <cstddef>
#include "isaac/defines.h"

namespace isaac
{

typedef long long int_t;

struct ISAACAPI size4
{
  size4(int_t s0, int_t s1 = 1, int_t s2 = 1, int_t s3 = 1)
  {
    data_[0] = s0;
    data_[1] = s1;
    data_[2] = s2;
    data_[3] = s3;
  }

  bool operator==(size4 const & other) const { return (*this)[0]==other[0] && (*this)[1]==other[1]; }
  int_t operator[](size_t i) const { return data_[i]; }
  int_t & operator[](size_t i) { return data_[i]; }
private:
  int_t data_[4];
};

inline int_t prod(size4 const & s) { return s[0]*s[1]; }

static const int_t start = 0;
static const int_t end = -1;
struct slice
{
  slice(int_t _start, int_t _end, int_t _stride = 1) : start(_start), end(_end), stride(_stride) { }

  int_t size(int_t bound) const
  {
    int_t effective_end = (end < 0)?bound - (end + 1):end;
    return (effective_end - start)/stride;
  }

  int_t start;
  int_t end;
  int_t stride;
};



}
#endif
