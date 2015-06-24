#include "CL/cl.hpp"

#include "isaac/wrap/clBLAS.h"
#include "isaac/array.h"

namespace is = isaac;

extern "C"
{

    clblasStatus clblasSetup()
    {
        return clblasSuccess;
    }

    void clblasTeardown()
    {

    }

    clblasStatus clblasSgemm(clblasOrder order, clblasTranspose transA,  clblasTranspose transB,
                            size_t M, size_t N, size_t K,
                            cl_float alpha, const cl_mem mA, size_t offA, size_t lda,
                            const cl_mem mB, size_t offB, size_t ldb, cl_float beta,
                            cl_mem mC, size_t offC, size_t ldc,
                            cl_uint numCommandQueues, cl_command_queue *commandQueues,
                            cl_uint numEventsInWaitList, const cl_event *eventWaitList, cl_event *events)
    {
        is::int_t As1 = M, As2 = K;
        is::int_t Bs1 = K, Bs2 = N;

        if(transA==clblasTrans) std::swap(As1, As2);
        if(transB==clblasTrans) std::swap(Bs1, Bs2);

        is::array A(is::FLOAT_TYPE, cl::Buffer(mA), is::_(offA%lda, As1, 1), is::_(offA/lda, As2, 1), lda);
        is::array B(is::FLOAT_TYPE, cl::Buffer(mB), is::_(offB%ldb, Bs1, 1), is::_(offB/ldb, Bs2, 1), ldb);
        is::array C(is::FLOAT_TYPE, cl::Buffer(mC), is::_(offC%ldc, M, 1), is::_(offC/ldc, N, 1), ldc);

        std::vector<is::driver::Event> waitlist;
        for(cl_uint i = 0 ; i < numEventsInWaitList ; ++i)
            waitlist.push_back(cl::Event(eventWaitList[i]));

        std::list<is::driver::Event> levents;

        for(cl_uint i = 0 ; i < numCommandQueues ; ++i)
        {
            clRetainCommandQueue(commandQueues[i]);
            cl::CommandQueue clqueue(commandQueues[i]);
            is::driver::CommandQueue queue(clqueue);


            is::execution_options_type opt(queue, &levents, &waitlist);

            if(transA==clblasTrans && transB==clblasTrans)
                C = is::control(alpha*dot(A.T(), B.T()) + beta*C, opt);
            else if(transA==clblasTrans && transB==clblasNoTrans)
                C = is::control(alpha*dot(A.T(), B) + beta*C, opt);
            else if(transA==clblasNoTrans && transB==clblasTrans)
                C = is::control(alpha*dot(A, B.T()) + beta*C, opt);
            else
                C = is::control(alpha*dot(A, B) + beta*C, opt);
        }

        if(events)
            *events = static_cast<cl::Event>(levents.front())();
        std::cout << events << std::endl;

        return clblasSuccess;
    }

}
