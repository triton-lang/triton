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

#if defined(ANDROID) || defined(__CYGWIN__)
  std::stringstream ss;
  ss << t;
  return ss.str();
#else
  return  std::to_string(t);
#endif
}

}
}

#endif
