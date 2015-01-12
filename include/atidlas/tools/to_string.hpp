#ifndef ATIDLAS_TOOLS_TO_STRING_HPP
#define ATIDLAS_TOOLS_TO_STRING_HPP

#include <string>

namespace atidlas
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
