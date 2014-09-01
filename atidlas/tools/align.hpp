#ifndef ATIDLAS_TOOLS_ALIGN_HPP
#define ATIDLAS_TOOLS_ALIGN_HPP

#include <string>

namespace atidlas
{
namespace tools
{

template<class IntT>
IntT align_to_previous_multiple(IntT to_round, IntT base)
{
  return to_round/base * base;
}

template<class IntT>
IntT align_to_next_multiple(IntT to_round, IntT base)
{
  if (to_round % base == 0)
    return to_round;
  return (to_round + base - 1)/base * base;
}

}
}

#endif
