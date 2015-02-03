#ifndef ATIDLAS_CL_QUEUES_H
#define ATIDLAS_CL_QUEUES_H

#include <map>
#include <CL/cl.hpp>
#include "atidlas/cl/compare.hpp"

namespace atidlas
{
namespace cl_ext
{

typedef std::map<std::pair<cl_program, unsigned int>, cl::Kernel> kernels_t;
typedef std::vector<std::pair<cl::Context, std::vector<cl::CommandQueue> > > queues_t;

extern kernels_t kernels;
extern queues_t queues;
extern unsigned int default_context_idx;
extern cl_command_queue_properties queue_properties;


void synchronize(cl::Context const & context);
cl::Context default_context();
cl::CommandQueue & get_queue(cl::Context const &, std::size_t);
cl::Device get_device(cl::CommandQueue &);
std::vector<cl::CommandQueue> & get_queues(cl::Context const & ctx);

}
}

#endif
