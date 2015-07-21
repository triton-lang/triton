#include "isaac/array.h"
#include "isaac/tools/timer.hpp"

#include <vector>

namespace isc = isaac;

#ifdef BENCH_CUBLAS
__global__ void dummy(){}
#endif


int main()
{
  for(isc::driver::queues_type::data_type::const_iterator it = isc::driver::queues.data().begin() ; it != isc::driver::queues.data().end() ; ++it)
  {
    cl::CommandQueue queue = it->second[0];
    cl::Context context = it->first;
    cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
    cl::Program program(context,"__kernel void dummy(){}");
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

    {
    long time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    std::cout << "Kernel launch overhead: " << time << std::endl;
    }

#ifdef BENCH_CUBLAS
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    dummy<<<1, 1>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "CUDA Kernel launch overhead: " << time << std::endl;
#endif
    std::cout << "-------------------------" << std::endl;
  }

}
