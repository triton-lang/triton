#include "isaac/array.h"
#include "isaac/symbolic/execute.h"
#include "clBLAS.h"

namespace sc = isaac;

extern "C"
{

    clblasStatus clblasSetup()
    {
        return clblasSuccess;
    }

    void clblasTeardown()
    {
        isaac::profiles::release();
        isaac::driver::backend::release();
    }

    void execute(sc::math_expression const & operation, sc::driver::Context const & context,
                 cl_uint numCommandQueues, cl_command_queue *commandQueues,
                 cl_uint numEventsInWaitList, const cl_event *eventWaitList,
                 cl_event *events)
    {
        std::vector<sc::driver::Event> waitlist;
        for(cl_uint i = 0 ; i < numEventsInWaitList ; ++i)
            waitlist.push_back(eventWaitList[i]);
        for(cl_uint i = 0 ; i < numCommandQueues ; ++i)
        {
            std::list<sc::driver::Event> levents;
            sc::execution_options_type options(sc::driver::CommandQueue(commandQueues[i],false), &levents, &waitlist);
            sc::execute(sc::execution_handler(operation, options), sc::profiles::get(options.queue(context)));
            if(events)
            {
                events[i] = levents.front().handle().cl();
                sc::driver::dispatch::clRetainEvent(events[i]);
            }
            sc::driver::dispatch::clFlush(commandQueues[i]);
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
        sc::array x((sc::int_t)N, TYPE_ISAAC, sc::driver::Buffer(mx,false), (sc::int_t)offx, incx); \
        sc::array y((sc::int_t)N, TYPE_ISAAC, sc::driver::Buffer(my,false), (sc::int_t)offy, incy); \
        execute(sc::assign(y, alpha*x + y), y.context(), numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events); \
        return clblasSuccess; \
    }

    MAKE_AXPY(S, sc::FLOAT_TYPE, cl_float)
    MAKE_AXPY(D, sc::DOUBLE_TYPE, cl_double)

    //SCAL
    #define MAKE_SCAL(TYPE_CHAR, TYPE_ISAAC, TYPE_CL) \
    clblasStatus clblas ## TYPE_CHAR ## scal(size_t N, TYPE_CL alpha,\
                             cl_mem mx, size_t offx, int incx,\
                             cl_uint numCommandQueues, cl_command_queue *commandQueues,\
                             cl_uint numEventsInWaitList, const cl_event *eventWaitList, cl_event *events)\
    {\
        sc::array x((sc::int_t)N, TYPE_ISAAC, sc::driver::Buffer(mx,false), (sc::int_t)offx, incx);\
        execute(sc::assign(x, alpha*x), x.context(), numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);\
        return clblasSuccess;\
    }

    MAKE_SCAL(S, sc::FLOAT_TYPE, cl_float)
    MAKE_SCAL(D, sc::DOUBLE_TYPE, cl_double)

    //COPY
    #define MAKE_COPY(TYPE_CHAR, TYPE_ISAAC, TYPE_CL)\
    clblasStatus clblas ## TYPE_CHAR ## copy(size_t N,\
                             const cl_mem mx, size_t offx, int incx,\
                             cl_mem my, size_t offy, int incy,\
                             cl_uint numCommandQueues, cl_command_queue *commandQueues,\
                             cl_uint numEventsInWaitList, const cl_event *eventWaitList, cl_event *events)\
    {\
        const sc::array x((sc::int_t)N, TYPE_ISAAC, sc::driver::Buffer(mx, false), (sc::int_t)offx, incx);\
        sc::array y((sc::int_t)N, TYPE_ISAAC, sc::driver::Buffer(my, false), (sc::int_t)offy, incy);\
        execute(sc::assign(y, x), y.context(), numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);\
        return clblasSuccess;\
    }

    MAKE_COPY(S, sc::FLOAT_TYPE, cl_float)
    MAKE_COPY(D, sc::DOUBLE_TYPE, cl_double)

    //DOT
    #define MAKE_DOT(TYPE_CHAR, TYPE_ISAAC, TYPE_CL) \
    clblasStatus clblas ## TYPE_CHAR ## dot(size_t N, cl_mem dotProduct, size_t offDP, \
               const cl_mem mx, size_t offx, int incx, \
               const cl_mem my, size_t offy, int incy, \
               cl_mem /*scratchBuff*/, cl_uint numCommandQueues, \
               cl_command_queue *commandQueues, cl_uint numEventsInWaitList, \
               const cl_event *eventWaitList, cl_event *events) \
    { \
        sc::array x((sc::int_t)N, TYPE_ISAAC, sc::driver::Buffer(mx, false), (sc::int_t)offx, incx); \
        sc::array y((sc::int_t)N, TYPE_ISAAC, sc::driver::Buffer(my, false), (sc::int_t)offy, incy); \
        sc::scalar s(TYPE_ISAAC, sc::driver::Buffer(dotProduct, false), (sc::int_t)offDP); \
        execute(sc::assign(s, dot(x,y)), s.context(), numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events); \
        return clblasSuccess; \
    }

    MAKE_DOT(S, sc::FLOAT_TYPE, cl_float)
    MAKE_DOT(D, sc::DOUBLE_TYPE, cl_double)

    //ASUM
    #define MAKE_ASUM(TYPE_CHAR, TYPE_ISAAC, TYPE_CL) \
    clblasStatus clblas ## TYPE_CHAR ## asum(size_t N, cl_mem asum, size_t offAsum, \
                             const cl_mem mx, size_t offx, int incx,\
                             cl_mem /*scratchBuff*/, cl_uint numCommandQueues, cl_command_queue *commandQueues,\
                             cl_uint numEventsInWaitList, const cl_event *eventWaitList, cl_event *events)\
    {\
        sc::array x((sc::int_t)N, TYPE_ISAAC, sc::driver::Buffer(mx, false), (sc::int_t)offx, incx);\
        sc::scalar s(TYPE_ISAAC, sc::driver::Buffer(asum, false), (sc::int_t)offAsum);\
        execute(sc::assign(s, sum(abs(x))), s.context(), numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);\
        return clblasSuccess;\
    }

    MAKE_ASUM(S, sc::FLOAT_TYPE, cl_float)
    MAKE_ASUM(D, sc::DOUBLE_TYPE, cl_double)


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
        if(order==clblasRowMajor){\
            std::swap(M, N);\
            transA = (transA==clblasTrans)?clblasNoTrans:clblasTrans;\
        }\
        sc::array A((sc::int_t)M, (sc::int_t)N, TYPE_ISAAC, sc::driver::Buffer(mA, false), (sc::int_t)offA, (sc::int_t)lda);\
        \
        sc::int_t sx = (sc::int_t)N, sy = (sc::int_t)M;\
        if(transA) std::swap(sx, sy);\
        sc::array x(sx, TYPE_ISAAC, sc::driver::Buffer(mx, false), (sc::int_t)offx, incx);\
        sc::array y(sy, TYPE_ISAAC, sc::driver::Buffer(my, false), (sc::int_t)offy, incy);\
        \
        sc::driver::Context const & context = A.context();\
        if(transA==clblasTrans)\
            execute(sc::assign(y, alpha*dot(A.T, x) + beta*y), context, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);\
        else\
            execute(sc::assign(y, alpha*dot(A, x) + beta*y), context, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);\
        return clblasSuccess;\
    }

    MAKE_GEMV(S, sc::FLOAT_TYPE, cl_float)
    MAKE_GEMV(D, sc::DOUBLE_TYPE, cl_double)

    //*****************
    //BLAS3
    //*****************

    #define MAKE_GEMM(TYPE_CHAR, TYPE_ISAAC, TYPE_CL) \
    clblasStatus clblas ## TYPE_CHAR ## gemm(clblasOrder order, clblasTranspose transA,  clblasTranspose transB,\
                            size_t M, size_t N, size_t K,\
                            TYPE_CL alpha, const cl_mem cmA, size_t offA, size_t lda,\
                            const cl_mem cmB, size_t offB, size_t ldb, TYPE_CL beta,\
                            cl_mem mC, size_t offC, size_t ldc,\
                            cl_uint numCommandQueues, cl_command_queue *commandQueues,\
                            cl_uint numEventsInWaitList, const cl_event *eventWaitList, cl_event *events)\
    {\
        cl_mem mA = cmA;\
        cl_mem mB = cmB;\
        if(order==clblasRowMajor){\
            std::swap(mA, mB);\
            std::swap(offA, offB);\
            std::swap(lda, ldb);\
            std::swap(M, N);\
            std::swap(transA, transB);\
        }\
        sc::int_t As1 = (sc::int_t)M, As2 = (sc::int_t)K;\
        sc::int_t Bs1 = (sc::int_t)K, Bs2 = (sc::int_t)N;\
        if(transA==clblasTrans) std::swap(As1, As2);\
        if(transB==clblasTrans) std::swap(Bs1, Bs2);\
        /*Struct*/\
        sc::array A(As1, As2, TYPE_ISAAC, sc::driver::Buffer(mA, false), (sc::int_t)offA, (sc::int_t)lda);\
        sc::array B(Bs1, Bs2, TYPE_ISAAC, sc::driver::Buffer(mB, false), (sc::int_t)offB, (sc::int_t)ldb);\
        sc::array C((sc::int_t)M, (sc::int_t)N, TYPE_ISAAC, sc::driver::Buffer(mC, false), (sc::int_t)offC, (sc::int_t)ldc);\
        sc::driver::Context const & context = C.context();\
        /*Operation*/\
        if((transA==clblasTrans) && (transB==clblasTrans))\
            execute(sc::assign(C, alpha*dot(A.T, B.T) + beta*C), context, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);\
        else if((transA==clblasTrans) && (transB==clblasNoTrans))\
            execute(sc::assign(C, alpha*dot(A.T, B) + beta*C), context, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);\
        else if((transA==clblasNoTrans) && (transB==clblasTrans))\
            execute(sc::assign(C, alpha*dot(A, B.T) + beta*C), context, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);\
        else\
            execute(sc::assign(C, alpha*dot(A, B) + beta*C), context, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events);\
        return clblasSuccess;\
    }

    MAKE_GEMM(S, sc::FLOAT_TYPE, cl_float)
    MAKE_GEMM(D, sc::DOUBLE_TYPE, cl_double)

#undef DOT

}
