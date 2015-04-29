#ifndef ISAAC_TOOLS_TO_STRING_HPP
#define ISAAC_TOOLS_TO_STRING_HPP

#include <string>
#include <sstream>

namespace isaac
{
namespace tools
{

template<class T>
inline std::string to_string ( T const t )
{
  std::stringstream ss;
  ss << t;
  return ss.str();
}

}
}

#endif
