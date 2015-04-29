#ifndef ISAAC_BACKEND_STREAM_H
#define ISAAC_BACKEND_STREAM_H

#include <sstream>

namespace isaac
{

class kernel_generation_stream : public std::ostream
{
  class kgenstream : public std::stringbuf
  {
  public:
    kgenstream(std::ostringstream& oss,unsigned int const & tab_count) ;
    int sync();
    ~kgenstream();
  private:
    std::ostream& oss_;
    unsigned int const & tab_count_;
  };

public:
  kernel_generation_stream();
  ~kernel_generation_stream();

  std::string str();
  void inc_tab();
  void dec_tab();
private:
  unsigned int tab_count_;
  std::ostringstream oss;
};

}

#endif
