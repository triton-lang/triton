#include "atidlas/backend/stream.h"

namespace atidlas
{

kernel_generation_stream::kgenstream::kgenstream(std::ostringstream& oss,unsigned int const & tab_count) :
  oss_(oss), tab_count_(tab_count)
{ }

int kernel_generation_stream::kgenstream::sync()
{
  for (unsigned int i=0; i<tab_count_;++i)
    oss_ << "    ";
  oss_ << str();
  str("");
  return !oss_;
}

kernel_generation_stream::kgenstream:: ~kgenstream()
{  pubsync(); }

kernel_generation_stream::kernel_generation_stream() : std::ostream(new kgenstream(oss,tab_count_)), tab_count_(0)
{ }

kernel_generation_stream::~kernel_generation_stream()
{ delete rdbuf(); }

std::string kernel_generation_stream::str()
{ return oss.str(); }

void kernel_generation_stream::inc_tab()
{ ++tab_count_; }

void kernel_generation_stream::dec_tab()
{ --tab_count_; }

}

