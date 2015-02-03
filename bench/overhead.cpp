#include "atidlas/array.h"
#include "atidlas/tools/timer.hpp"

#include <vector>

namespace ad = atidlas;

#ifdef BENCH_CUBLAS
__global__ void dummy(){}
#endif


int main()
{
  for(ad::cl_ext::queues_t::iterator it = ad::cl_ext::queues.begin() ; it != ad::cl_ext::queues.end() ; ++it)
  {
    cl::CommandQueue queue = it->second[0];
    cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
    cl::Program program("__kernel void dummy(){}");
                program.build();
    cl::Kernel kernel(program, "dummy");

    cl::NDRange offset = cl::NullRange;
    cl::NDRange global(1);
    cl::NDRange local(1);

    cl::Event event;
    std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "-------------------------" << std::endl;

    queue.enqueueNDRangeKernel(kernel, offset, global, local, NULL, &event);
    queue.flush();
    queue.finish();

    float time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>() - event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    std::cout << "Kernel launch overhead: " << time << std::endl;

#ifdef BENCH_CUBLAS
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    dummy<<1, 1>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "CUDA Kernel launch overhead: " << time << std::endl;
#endif
    std::cout << "-------------------------" << std::endl;
  }

}
