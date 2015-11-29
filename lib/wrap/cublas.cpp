#include "isaac/array.h"
#include "isaac/symbolic/execute.h"
#include "cublas.h"

namespace sc = isaac;

extern "C"
{

//    sc::driver::Context current_context()
//    {
//        CUcontext ctx;
//        CUdevice dev;
//        cuCtxGetCurrent(&ctx);
//        cuCtxGetDevice(&dev);
//        return sc::driver::Context(ctx, dev, false);
//    }

    struct cublasContext
    {

    };

    cublasStatus_t cublasCreate_v2 (cublasHandle_t *handle)
    {
        *handle = new cublasContext();
        return CUBLAS_STATUS_SUCCESS;
    }

    cublasStatus_t cublasDestroy_v2 (cublasHandle_t handle)
    {
        delete handle;
        return cublasShutdown();
    }

    cublasStatus cublasInit()
    {
        return CUBLAS_STATUS_SUCCESS;
    }

    cublasStatus cublasShutdown()
    {
        isaac::profiles::release();
        isaac::driver::backend::release();
        return CUBLAS_STATUS_SUCCESS;
    }




    //*****************
    //BLAS1
    //*****************

    //AXPY
    #define MAKE_AXPY(TYPE_CHAR, TYPE_ISAAC, TYPE_CU) \
    void cublas ## TYPE_CHAR ## axpy (int n, TYPE_CU alpha, const TYPE_CU *x, int incx, TYPE_CU *y, int incy)\
    {\
        sc::array dx((sc::int_t)n, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)x,false), 0, incx); \
        sc::array dy((sc::int_t)n, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)y,false), 0, incy); \
        sc::execute(sc::assign(dy, alpha*dx + dy));\
    }\
    cublasStatus_t cublas ## TYPE_CHAR ## axpy_v2 (cublasHandle_t, int n, const TYPE_CU *alpha,\
                                   const TYPE_CU *x, int incx, TYPE_CU *y, int incy)\
    {\
        cublas ## TYPE_CHAR ## axpy(n, *alpha, x, incx, y, incy);\
        return CUBLAS_STATUS_SUCCESS;\
    }

    MAKE_AXPY(S, sc::FLOAT_TYPE, float)
    MAKE_AXPY(D, sc::DOUBLE_TYPE, double)

    //COPY
    #define MAKE_COPY(TYPE_CHAR, TYPE_ISAAC, TYPE_CU) \
    void cublas ## TYPE_CHAR ## copy (int n, const TYPE_CU *x, int incx, TYPE_CU *y, int incy)\
    {\
        sc::array dx((sc::int_t)n, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)x,false), 0, incx); \
        sc::array dy((sc::int_t)n, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)y,false), 0, incy); \
        sc::execute(sc::assign(dy,dx));\
    }\
    cublasStatus_t cublas ## TYPE_CHAR ## copy_v2 (cublasHandle_t, int n, const TYPE_CU *x, int incx, TYPE_CU *y, int incy)\
    {\
        cublas ## TYPE_CHAR ## copy(n, x, incx, y, incy);\
        return CUBLAS_STATUS_SUCCESS;\
    }

    MAKE_COPY(S, sc::FLOAT_TYPE, float)
    MAKE_COPY(D, sc::DOUBLE_TYPE, double)

    //SCAL
    #define MAKE_SCAL(TYPE_CHAR, TYPE_ISAAC, TYPE_CU) \
    void cublas ## TYPE_CHAR ## scal (int n, TYPE_CU alpha, TYPE_CU *x, int incx)\
    {\
        sc::array dx((sc::int_t)n, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)x,false), 0, incx); \
        sc::execute(sc::assign(dx,alpha*dx));\
    }\
    cublasStatus_t cublas ## TYPE_CHAR ## scal_v2 (cublasHandle_t, int n, const TYPE_CU * alpha, TYPE_CU *x, int incx)\
    {\
        cublas ## TYPE_CHAR ## scal(n, *alpha, x, incx);\
        return CUBLAS_STATUS_SUCCESS;\
    }

    MAKE_SCAL(S, sc::FLOAT_TYPE, float)
    MAKE_SCAL(D, sc::DOUBLE_TYPE, double)

    //DOT
    #define MAKE_DOT(TYPE_CHAR, TYPE_ISAAC, TYPE_CU) \
    TYPE_CU cublas ## TYPE_CHAR ## dot (int n, const TYPE_CU *x, int incx, const TYPE_CU *y, int incy)\
    {\
        sc::array dx((sc::int_t)n, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)x,false), 0, incx); \
        sc::array dy((sc::int_t)n, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)y,false), 0, incy); \
        return sc::value_scalar(sc::dot(dx,dy));\
    }\
    cublasStatus_t cublas ## TYPE_CHAR ## dot_v2 (cublasHandle_t, int n, const TYPE_CU *x, int incx, const TYPE_CU *y, int incy, TYPE_CU* result)\
    {\
        *result = cublas ## TYPE_CHAR ## dot(n, x, incx, y, incy);\
        return CUBLAS_STATUS_SUCCESS;\
    }

    MAKE_DOT(S, sc::FLOAT_TYPE, float)
    MAKE_DOT(D, sc::DOUBLE_TYPE, double)

    //ASUM
    #define MAKE_ASUM(TYPE_CHAR, TYPE_ISAAC, TYPE_CU) \
    TYPE_CU cublas ## TYPE_CHAR ## asum (int n, const TYPE_CU *x, int incx)\
    {\
        sc::array dx((sc::int_t)n, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)x,false), 0, incx); \
        return sc::value_scalar(sum(abs(dx)));\
    }\
    cublasStatus_t cublas ## TYPE_CHAR ## asum_v2 (cublasHandle_t, int n, const TYPE_CU *x, int incx, TYPE_CU* result)\
    {\
        *result = cublas ## TYPE_CHAR ## asum(n, x, incx);\
        return CUBLAS_STATUS_SUCCESS;\
    }

    MAKE_ASUM(S, sc::FLOAT_TYPE, float)
    MAKE_ASUM(D, sc::DOUBLE_TYPE, double)

    //*****************
    //BLAS2
    //*****************

    #define MAKE_GEMV(TYPE_CHAR, TYPE_ISAAC, TYPE_CU) \
    void cublas ## TYPE_CHAR ## gemv (char trans, int m, int n, TYPE_CU alpha,\
                                   const TYPE_CU *A, int lda, const TYPE_CU *x, int incx,\
                                   TYPE_CU beta, TYPE_CU *y, int incy)\
    {\
        sc::array dA((sc::int_t)m, (sc::int_t)n, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)A, false), 0, (sc::int_t)lda);\
        \
        sc::int_t sx = (sc::int_t)n, sy = (sc::int_t)m;\
        if(trans=='T') std::swap(sx, sy);\
        sc::array dx(sx, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)x, false), 0, incx);\
        sc::array dy(sy, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)y, false), 0, incy);\
        \
        if(trans=='T')\
            sc::execute(sc::assign(dy, alpha*dot(dA.T, dx) + beta*dy));\
        else\
            sc::execute(sc::assign(dy, alpha*dot(dA, dx) + beta*dy));\
    }\
    cublasStatus_t cublas ## TYPE_CHAR ## gemv_v2 (cublasHandle_t, cublasOperation_t trans, int m,  int n, const TYPE_CU *alpha,\
                                   const TYPE_CU *A, int lda, const TYPE_CU *x, int incx, const TYPE_CU *beta, TYPE_CU *y, int incy)\
    {\
        if(trans==CUBLAS_OP_C)\
            return CUBLAS_STATUS_NOT_SUPPORTED;\
        cublas ## TYPE_CHAR ## gemv((trans==CUBLAS_OP_N)?'N':'T', m, n, *alpha, A, lda, x, incx, *beta, y, incy);\
        return CUBLAS_STATUS_SUCCESS;\
    }

    MAKE_GEMV(S, sc::FLOAT_TYPE, float)
    MAKE_GEMV(D, sc::DOUBLE_TYPE, double)


    #define MAKE_GER(TYPE_CHAR, TYPE_ISAAC, TYPE_CU) \
    void cublas ## TYPE_CHAR ## ger (int m, int n, TYPE_CU alpha, const TYPE_CU *x, int incx,\
                                     const TYPE_CU *y, int incy, TYPE_CU *A, int lda)\
    {\
        sc::array dx((sc::int_t)n, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)x,false), 0, incx); \
        sc::array dy((sc::int_t)n, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)y,false), 0, incy); \
        sc::array dA((sc::int_t)m, (sc::int_t)n, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)A, false), 0, (sc::int_t)lda);\
        sc::execute(sc::assign(dA, alpha*sc::outer(dx, dy) + dA));\
    }\
    cublasStatus_t cublas ## TYPE_CHAR ## ger_v2 (cublasHandle_t, int m, int n, const TYPE_CU * alpha, const TYPE_CU *x, int incx,\
                                                 const TYPE_CU *y, int incy, TYPE_CU *A, int lda)\
    {\
        cublas ## TYPE_CHAR ## ger(m, n, *alpha, x, incx, y, incy, A, lda);\
        return CUBLAS_STATUS_SUCCESS;\
    }

    MAKE_GER(S, sc::FLOAT_TYPE, float)
    MAKE_GER(D, sc::DOUBLE_TYPE, double)


    //*****************
    //BLAS3
    //*****************

    #define MAKE_GEMM(TYPE_CHAR, TYPE_ISAAC, TYPE_CU) \
    void cublas ## TYPE_CHAR ## gemm (char transa, char transb, int m, int n, int k,\
                                   TYPE_CU alpha, const TYPE_CU *A, int lda,\
                                   const TYPE_CU *B, int ldb, TYPE_CU beta, TYPE_CU *C,\
                                   int ldc)\
    {\
        std::cout << transa << " " << transb << " " << m << " " << n << " " << k << std::endl;\
        if(k==1 && m>1 && n>1){\
            sc::array dA((sc::int_t)m, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)A, false), 0, transa=='N'?1:lda);\
            sc::array dB((sc::int_t)n, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)B, false), 0, transb=='T'?1:ldb);\
            sc::array dC((sc::int_t)m, (sc::int_t)n, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)C, false), 0, (sc::int_t)ldc);\
            sc::execute(sc::assign(dC, alpha*sc::outer(dA, dB) + beta*dC));\
            return;\
        }\
        sc::int_t As1 = (sc::int_t)m, As2 = (sc::int_t)k;\
        sc::int_t Bs1 = (sc::int_t)k, Bs2 = (sc::int_t)n;\
        if(transa=='T') std::swap(As1, As2);\
        if(transb=='T') std::swap(Bs1, Bs2);\
        /*Struct*/\
        sc::array dA(As1, As2, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)A, false), 0, (sc::int_t)lda);\
        sc::array dB(Bs1, Bs2, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)B, false), 0, (sc::int_t)ldb);\
        sc::array dC((sc::int_t)m, (sc::int_t)n, TYPE_ISAAC, sc::driver::Buffer((CUdeviceptr)C, false), 0, (sc::int_t)ldc);\
        /*Operation*/\
        if((transa=='T') && (transb=='T'))\
            sc::execute(sc::assign(dC, alpha*dot(dA.T, dB.T) + beta*dC));\
        else if((transa=='T') && (transb=='N'))\
            sc::execute(sc::assign(dC, alpha*dot(dA.T, dB) + beta*dC));\
        else if((transa=='N') && (transb=='T'))\
            sc::execute(sc::assign(dC, alpha*dot(dA, dB.T) + beta*dC));\
        else\
            sc::execute(sc::assign(dC, alpha*dot(dA, dB) + beta*dC));\
    }\
    cublasStatus_t cublas ## TYPE_CHAR ## gemm_v2(cublasHandle_t, cublasOperation_t transa, cublasOperation_t transb,\
                                                    int m, int n, int k, const TYPE_CU *alpha, const TYPE_CU *A,\
                                                    int lda, const TYPE_CU *B, int ldb,const TYPE_CU *beta, TYPE_CU *C, int ldc)\
    {\
        if(transa==CUBLAS_OP_C || transb==CUBLAS_OP_C)\
            return CUBLAS_STATUS_NOT_SUPPORTED;\
        cublas ## TYPE_CHAR ## gemm((transa==CUBLAS_OP_N)?'N':'T', (transb==CUBLAS_OP_N)?'N':'T', m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);\
        return CUBLAS_STATUS_SUCCESS;\
    }

    MAKE_GEMM(S, sc::FLOAT_TYPE, cl_float)
    MAKE_GEMM(D, sc::DOUBLE_TYPE, cl_double)
}
