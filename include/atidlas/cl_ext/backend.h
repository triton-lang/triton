#ifndef ATIDLAS_CL_QUEUES_H
#define ATIDLAS_CL_QUEUES_H

#include <map>
#include <list>
#include <CL/cl.hpp>

namespace atidlas
{
namespace cl_ext
{

class queues_type
{
private:
  void append(cl::Context const &);
  void init();
public:
  typedef std::list<std::pair<cl::Context, std::vector<cl::CommandQueue> > > data_type;
  std::vector<cl::CommandQueue> & operator[](cl::Context const &);
  cl::Context default_context();
  data_type const & data();
private:
  data_type data_;
};

typedef std::map<std::pair<cl_program, unsigned int>, cl::Kernel> kernels_type;

extern kernels_type kernels;
extern queues_type queues;
extern unsigned int default_context_idx;
extern cl_command_queue_properties queue_properties;

void synchronize(cl::Context const & context);
cl::Context default_context();

}
}

#endif
