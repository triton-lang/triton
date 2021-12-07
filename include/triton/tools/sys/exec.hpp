#ifndef TRITON_TOOLS_SYS_EXEC_HPP
#define TRITON_TOOLS_SYS_EXEC_HPP

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace triton
{
namespace tools
{


#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

#ifndef WEXITSTATUS
#define WEXITSTATUS(stat_val) ((unsigned)(stat_val) & 255)
#endif

int exec(const std::string& cmd, std::string& result) {
  char buffer[128];
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe)
    return 0;
  result.clear();
  try {
    while (fgets(buffer, sizeof buffer, pipe) != NULL)
      result += buffer;
  } catch (...) {
    pclose(pipe);
    return 0;
  }
  int status = pclose(pipe);
  return WEXITSTATUS(status);

}

}
}

#endif
