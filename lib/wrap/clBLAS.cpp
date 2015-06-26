#include "CL/cl.hpp"

#include "isaac/wrap/clBLAS.h"
#include "isaac/array.h"
#include "isaac/symbolic/execute.h"

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

    void execute(is::array_expression const & operation, is::driver::Context const & context,
                 cl_uint numCommandQueues, cl_command_queue *commandQueues,
                 cl_uint numEventsInWaitList, const cl_event *eventWaitList,
                 cl_event *events)
    {
        std::vector<is::driver::Event> waitlist;
        for(cl_uint i = 0 ; i < numEventsInWaitList ; ++i)
            waitlist.push_back(cl::Event(eventWaitList[i]));
        for(cl_uint i = 0 ; i < numCommandQueues ; ++i)
        {
            std::list<is::driver::Event> levents;
            is::driver::CommandQueue queue(cl::CommandQueue(commandQueues[i]));
            clRetainCommandQueue(commandQueues[i]);
            is::execution_options_type options(queue, &levents, &waitlist);
            is::execute(is::control(operation, options), is::models(options.queue(context)));
            if(events)
            {
                events[i] = static_cast<cl::Event>(levents.front())();
                clRetainEvent(events[i]);
            }
        }
    }

    //*****************
    //BLAS1
    //*****************

    //AXPY
    #define MAKE_AXPY(TYPE_CHAR, TYPE_ISAAC, TYPE_CL) \
    clblasStatus clblas ## TYPE_CHAR ## axpy(size_t N, TYPE_CL alpha, \
                            const cl_mem mx,  size_t offx, int incx, \
                            cl_mem my, size_t offy, int incy, \
                            cl_uint numCommandQueues, cl_command_queue *commandQueues, \
                            cl_uint numEventsInWaitList, const cl_event *eventWaitList, \
                            cl_event *events) \
    { \
        is::array x(N, TYPE_ISAAC, cl::Buffer(mx), offx, incx); \
        clRetainMemObject(mx); \
        is::array y(N, TYPE_ISAAC, cl::Buffer(my), offy, incy); \
        clRetainMemObject(my); \
        execute(is::detail::assign(y, x + alpha*y), y.context(), numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events); \
        return clblasSuccess; \
    }

    MAKE_AXPY(S, is::FLOAT_TYPE, cl_float)
    MAKE_AXPY(D, is::DOUBLE_TYPE, cl_double)

    //SCAL
    #define MAKE_SCAL(TYPE_CHAR, TYPE_ISAAC, TYPE_CL) \
    clblasStatus clblas ## TYPE_CHAR ## scal(size_t N, TYPE_CL alpha,\
                             cl_mem mx, size_t offx, int incx,\
                             cl_uint numCommandQueues, cl_command_queue *commandQueues,\
                             cl_uint numEventsInWaitList, const cl_event *eventWaitList, cl_event *events)\
    {\
        is::array x(N, TYPE_ISAAC, cl::Buffer(mx), offx, incx);\
        clRetainMemObject(mx);\
        execute(is::detail::assign(x, alpha*x), x.context(), numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);\
        return clblasSuccess;\
    }

    MAKE_SCAL(S, is::FLOAT_TYPE, cl_float)
    MAKE_SCAL(D, is::DOUBLE_TYPE, cl_double)

    //COPY
    #define MAKE_COPY(TYPE_CHAR, TYPE_ISAAC, TYPE_CL)\
    clblasStatus clblas ## TYPE_CHAR ## copy(size_t N,\
                             const cl_mem mx, size_t offx, int incx,\
                             cl_mem my, size_t offy, int incy,\
                             cl_uint numCommandQueues, cl_command_queue *commandQueues,\
                             cl_uint numEventsInWaitList, const cl_event *eventWaitList, cl_event *events)\
    {\
        const is::array x(N, TYPE_ISAAC, cl::Buffer(mx), offx, incx);\
        clRetainMemObject(mx);\
        is::array y(N, TYPE_ISAAC, cl::Buffer(my), offy, incy);\
        clRetainMemObject(my);\
        execute(is::detail::assign(y, x), y.context(), numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);\
        return clblasSuccess;\
    }

    MAKE_COPY(S, is::FLOAT_TYPE, cl_float)
    MAKE_COPY(D, is::DOUBLE_TYPE, cl_double)

    //DOT
    #define MAKE_DOT(TYPE_CHAR, TYPE_ISAAC, TYPE_CL) \
    clblasStatus clblas ## TYPE_CHAR ## dot(size_t N, cl_mem dotProduct, size_t offDP, \
               const cl_mem mx, size_t offx, int incx, \
               const cl_mem my, size_t offy, int incy, \
               cl_mem /*scratchBuff*/, cl_uint numCommandQueues, \
               cl_command_queue *commandQueues, cl_uint numEventsInWaitList, \
               const cl_event *eventWaitList, cl_event *events) \
    { \
        is::array x(N, TYPE_ISAAC, cl::Buffer(mx), offx, incx); \
        clRetainMemObject(mx); \
        is::array y(N, TYPE_ISAAC, cl::Buffer(my), offy, incy); \
        clRetainMemObject(my); \
        is::scalar s(TYPE_ISAAC, cl::Buffer(dotProduct), offDP); \
        clRetainMemObject(dotProduct); \
        execute(is::detail::assign(s, dot(x,y)), s.context(), numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events); \
        return clblasSuccess; \
    }

    MAKE_DOT(S, is::FLOAT_TYPE, cl_float)
    MAKE_DOT(D, is::DOUBLE_TYPE, cl_double)

    //ASUM
    #define MAKE_ASUM(TYPE_CHAR, TYPE_ISAAC, TYPE_CL) \
    clblasStatus clblas ## TYPE_CHAR ## asum(size_t N, cl_mem asum, size_t offAsum, \
                             const cl_mem mx, size_t offx, int incx,\
                             cl_mem /*scratchBuff*/, cl_uint numCommandQueues, cl_command_queue *commandQueues,\
                             cl_uint numEventsInWaitList, const cl_event *eventWaitList, cl_event *events)\
    {\
        is::array x(N, TYPE_ISAAC, cl::Buffer(mx), offx, incx);\
        clRetainMemObject(mx);\
        is::scalar s(TYPE_ISAAC, cl::Buffer(asum), offAsum);\
        clRetainMemObject(asum);\
        execute(is::detail::assign(s, sum(abs(x))), s.context(), numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);\
        return clblasSuccess;\
    }

    MAKE_ASUM(S, is::FLOAT_TYPE, cl_float)
    MAKE_ASUM(D, is::DOUBLE_TYPE, cl_double)

    //*****************
    //BLAS2
    //*****************
    #define MAKE_GEMV(TYPE_CHAR, TYPE_ISAAC, TYPE_CL) \
    clblasStatus clblas ## TYPE_CHAR ## gemv(clblasOrder order, clblasTranspose transA,\
                             size_t M, size_t N,\
                             TYPE_CL alpha, const cl_mem mA, size_t offA, size_t lda,\
                             const cl_mem mx, size_t offx, int incx,\
                             TYPE_CL beta, cl_mem my, size_t offy, int incy,\
                             cl_uint numCommandQueues, cl_command_queue *commandQueues,\
                             cl_uint numEventsInWaitList, const cl_event *eventWaitList, cl_event *events)\
    {\
        is::int_t As1 = M, As2 = N;\
        if(transA==clblasTrans) std::swap(As1, As2);\
        is::array A(As1, As2, TYPE_ISAAC, cl::Buffer(mA), offA, lda);\
        clRetainMemObject(mA);\
        \
        is::array x(N, TYPE_ISAAC, cl::Buffer(mx), offx, incx);\
        clRetainMemObject(mx);\
        \
        is::array y(N, TYPE_ISAAC, cl::Buffer(my), offy, incy);\
        clRetainMemObject(my);\
        \
        is::driver::Context const & context = A.context();\
        if((transA==clblasTrans) ^ (order==clblasRowMajor))\
            execute(is::detail::assign(y, alpha*dot(A.T(), x) + beta*y), context, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);\
        else\
            execute(is::detail::assign(y, alpha*dot(A, x) + beta*y), context, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);\
        return clblasSuccess;\
    }

    MAKE_GEMV(S, is::FLOAT_TYPE, cl_float)
    MAKE_GEMV(D, is::DOUBLE_TYPE, cl_double)

    //*****************
    //BLAS3
    //*****************
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
        //Trans
        if(transA==clblasTrans) std::swap(As1, As2);
        if(transB==clblasTrans) std::swap(Bs1, Bs2);
        //Struct
        is::array A(As1, As2, is::FLOAT_TYPE, cl::Buffer(mA), offA, lda);
        clRetainMemObject(mA);
        is::array B(Bs1, Bs2, is::FLOAT_TYPE, cl::Buffer(mB), offB, ldb);
        clRetainMemObject(mB);
        is::array C(M, N, is::FLOAT_TYPE, cl::Buffer(mC), offC, ldc);
        clRetainMemObject(mC);
        is::driver::Context const & context = C.context();
        //Operation
        if(transA==clblasTrans && transB==clblasTrans)
            execute(is::detail::assign(C, alpha*dot(A.T(), B.T()) + beta*C), context, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
        else if(transA==clblasTrans && transB==clblasNoTrans)
            execute(is::detail::assign(C, alpha*dot(A.T(), B) + beta*C), context, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
        else if(transA==clblasNoTrans && transB==clblasTrans)
            execute(is::detail::assign(C, alpha*dot(A, B.T()) + beta*C), context, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);
        else
            execute(is::detail::assign(C, alpha*dot(A, B) + beta*C), context, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);

        return clblasSuccess;
    }

}
