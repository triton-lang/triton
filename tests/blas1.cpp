
#include "common.hpp"
#include "viennacl/vector.hpp"

#include "atidlas/templates/vector_axpy_template.hpp"
#include "atidlas/execute.hpp"

template<typename NumericT, class XType, class YType, class ZType>
int test_vectors(NumericT epsilon, atidlas::vector_axpy_parameters const & vector_axpy_parameters,
                 XType & cx, YType & cy, ZType & cz)
{
  int failure_count = 0;
  ZType buffer = cz;

  NumericT a = 3.12, b = 3.5;
  viennacl::scalar<NumericT> da(a), db(b);

  viennacl::vector<NumericT> xtmp(cx.internal_size());
  viennacl::vector<NumericT> ytmp(cy.internal_size());
  viennacl::vector<NumericT> ztmp(cz.internal_size());

  typename vector_maker<XType>::result_type x = vector_maker<XType>::make(xtmp, cx);
  typename vector_maker<YType>::result_type y = vector_maker<YType>::make(ytmp, cy);
  typename vector_maker<ZType>::result_type z = vector_maker<ZType>::make(ztmp, cz);


#define RUN_TEST_VECTOR_AXPY(NAME, CPU_LOOP, GPU_STATEMENT) \
  std::cout << NAME "..." << std::flush;\
  for(int_t i = 0 ; i < cz.size() ; ++i)\
    CPU_LOOP;\
  atidlas::execute(atidlas::vector_axpy_template(vector_axpy_parameters),\
                   GPU_STATEMENT,\
                   viennacl::ocl::current_context(), true);\
  viennacl::copy(z, buffer);\
  if(failure_vector(cz, buffer, epsilon))\
  {\
    failure_count++;\
    std::cout << " [Failure!]" << std::endl;\
  }\
  else\
    std::cout << std::endl;

  RUN_TEST_VECTOR_AXPY("z = x", cz[i] = cx[i], viennacl::scheduler::statement(z, viennacl::op_assign(), x))
  RUN_TEST_VECTOR_AXPY("z = x + y", cz[i] = cx[i] + cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), x + y))
  RUN_TEST_VECTOR_AXPY("z = x - y", cz[i] = cx[i] - cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), x - y))
  RUN_TEST_VECTOR_AXPY("z = x + y + z", cz[i] = cx[i] + cy[i] + cz[i], viennacl::scheduler::statement(z, viennacl::op_assign(), x + y + z))

  RUN_TEST_VECTOR_AXPY("z = a*x", cz[i] = a*cx[i], viennacl::scheduler::statement(z, viennacl::op_assign(), a*x))
  RUN_TEST_VECTOR_AXPY("z = da*x", cz[i] = a*cx[i], viennacl::scheduler::statement(z, viennacl::op_assign(), da*x))
  RUN_TEST_VECTOR_AXPY("z = a*x + b*y", cz[i] = a*cx[i] + b*cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), a*x + b*y))
  RUN_TEST_VECTOR_AXPY("z = da*x + b*y", cz[i] = a*cx[i] + b*cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), da*x + b*y))
  RUN_TEST_VECTOR_AXPY("z = a*x + db*y", cz[i] = a*cx[i] + b*cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), a*x + db*y))
  RUN_TEST_VECTOR_AXPY("z = da*x + db*y", cz[i] = a*cx[i] + b*cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), da*x + db*y))
#undef RUN_TEST_VECTOR_AXPY

  return failure_count;
}

template<typename NumericT>
int test_impl(NumericT epsilon)
{
  int_t N = 24378;
  int x_start = 4, y_start = 7, z_start = 15;
  int x_stride = 5, y_stride = 8, z_stride = 12;
  viennacl::range xr(x_start, N + x_start), yr(y_start, N + y_start), zr(z_start, N + z_start);
  viennacl::slice xs(x_start, x_stride, N), ys(y_start, y_stride, N), zs(z_start, z_stride, N);

  simple_vector<NumericT> x_vector(N), y_vector(N), z_vector(N);
  init_rand(x_vector);
  init_rand(y_vector);
  init_rand(z_vector);

  simple_vector<NumericT> x_range_holder(N + x_start);
  simple_vector<NumericT> x_slice_holder(x_start + N*x_stride);
  init_rand(x_range_holder);
  init_rand(x_slice_holder);
  simple_vector_range< simple_vector<NumericT> > x_range(x_range_holder, xr);
  simple_vector_slice< simple_vector<NumericT> > x_slice(x_slice_holder, xs);

  simple_vector<NumericT> y_range_holder(N + y_start);
  simple_vector<NumericT> y_slice_holder(y_start + N*y_stride);
  init_rand(y_range_holder);
  init_rand(y_slice_holder);
  simple_vector_range< simple_vector<NumericT> > y_range(y_range_holder, yr);
  simple_vector_slice< simple_vector<NumericT> > y_slice(y_slice_holder, ys);

  simple_vector<NumericT> z_range_holder(N + z_start);
  simple_vector<NumericT> z_slice_holder(z_start + N*z_stride);
  init_rand(z_range_holder);
  init_rand(z_slice_holder);
  simple_vector_range< simple_vector<NumericT> > z_range(z_range_holder, zr);
  simple_vector_slice< simple_vector<NumericT> > z_slice(z_slice_holder, zs);

  int_t failure_count = 0;

  atidlas::vector_axpy_parameters vector_axpy_parameters(4, 32, 128, atidlas::FETCH_FROM_GLOBAL_CONTIGUOUS);


#define TEST_OPERATIONS(XTYPE, YTYPE, ZTYPE)\
  std::cout << "> x : " #XTYPE " | y : " #YTYPE " | z : " #ZTYPE << std::endl;\
  failure_count += test_vectors(epsilon, vector_axpy_parameters, x_ ## XTYPE, y_ ## YTYPE, z_ ## ZTYPE);\

  TEST_OPERATIONS(vector, vector, vector)
  TEST_OPERATIONS(vector, vector, range)
  TEST_OPERATIONS(vector, vector, slice)
  TEST_OPERATIONS(vector, range, vector)
  TEST_OPERATIONS(vector, range, range)
  TEST_OPERATIONS(vector, range, slice)
  TEST_OPERATIONS(vector, slice, vector)
  TEST_OPERATIONS(vector, slice, range)
  TEST_OPERATIONS(vector, slice, slice)

  TEST_OPERATIONS(range, vector, vector)
  TEST_OPERATIONS(range, vector, range)
  TEST_OPERATIONS(range, vector, slice)
  TEST_OPERATIONS(range, range, vector)
  TEST_OPERATIONS(range, range, range)
  TEST_OPERATIONS(range, range, slice)
  TEST_OPERATIONS(range, slice, vector)
  TEST_OPERATIONS(range, slice, range)
  TEST_OPERATIONS(range, slice, slice)

  TEST_OPERATIONS(slice, vector, vector)
  TEST_OPERATIONS(slice, vector, range)
  TEST_OPERATIONS(slice, vector, slice)
  TEST_OPERATIONS(slice, range, vector)
  TEST_OPERATIONS(slice, range, range)
  TEST_OPERATIONS(slice, range, slice)
  TEST_OPERATIONS(slice, slice, vector)
  TEST_OPERATIONS(slice, slice, range)
  TEST_OPERATIONS(slice, slice, slice)

  return failure_count;
}

int main()
{
  int n_failures = 0;
  std::cout << ">> float" << std::endl;
  n_failures += test_impl<float>(1e-5);
  std::cout << ">> double" << std::endl;
  n_failures += test_impl<double>(1e-9);

  if(n_failures>0)
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
