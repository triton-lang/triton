#ifndef ATIDLAS_TOOLS_FIND_AND_REPLACE_HPP
#define ATIDLAS_TOOLS_FIND_AND_REPLACE_HPP

#include <string>

namespace atidlas
{
namespace tools
{

int inline find_and_replace(std::string & source, std::string const & find, std::string const & replace)
{
  int num=0;
  size_t fLen = find.size();
  size_t rLen = replace.size();
  for (size_t pos=0; (pos=source.find(find, pos))!=std::string::npos; pos+=rLen)
  {
    num++;
    source.replace(pos, fLen, replace);
  }
  return num;
}

}
}

#endif
