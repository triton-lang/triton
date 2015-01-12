#ifndef ATIDLAS_CL_QUEUES_H
#define ATIDLAS_CL_QUEUES_H

#include <map>
#include "atidlas/cl/cl.hpp"
#include "atidlas/cl/compare.hpp"

namespace atidlas
{

namespace cl
{

typedef std::map<cl::Program, cl::Kernel, cl::compare> kernels_t;
typedef std::map<cl::Context, std::vector<cl::CommandQueue>, cl::compare> queues_t;

queues_t init_queues();
void synchronize(cl::Context const & context);
cl::Context default_context();
cl::CommandQueue & get_queue(cl::Context const &, std::size_t);
cl::Device get_device(cl::CommandQueue &);
extern kernels_t kernels;
extern queues_t queues;

}

}

#endif
