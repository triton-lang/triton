#ifndef ISAAC_TOOLS_MISCELLANEOUS_HPP
#define ISAAC_TOOLS_MISCELLANEOUS_HPP

#include <string>
#include <sstream>

namespace isaac
{
namespace tools
{

inline unsigned int align(unsigned int to_round, unsigned int base)
{
  if (to_round % base == 0)
    return to_round;
  return (to_round + base - 1)/base * base;
}

}
}

#endif
