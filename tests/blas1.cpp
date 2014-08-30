
#include "common.hpp"
#include <cmath>
#include "viennacl/vector.hpp"

#include "atidlas/templates/vector_axpy_template.hpp"
#include "atidlas/execute.hpp"

template<typename NumericT, class XType, class YType, class ZType>
void test_vectors(NumericT epsilon, atidlas::vector_axpy_parameters const & vector_axpy_parameters,
                 XType & cx, YType & cy, ZType & cz)
{
  using namespace viennacl::linalg;
  using namespace std;

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
  cout << NAME "..." << flush;\
  for(int_t i = 0 ; i < cz.size() ; ++i)\
    CPU_LOOP;\
  atidlas::execute(atidlas::vector_axpy_template(vector_axpy_parameters),\
                   GPU_STATEMENT,\
                   viennacl::ocl::current_context(), true);\
  viennacl::copy(z, buffer);\
  if(failure_vector(cz, buffer, epsilon))\
  {\
    failure_count++;\
    cout << " [Failure!]" << endl;\
  }\
  else\
    cout << endl;

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

  RUN_TEST_VECTOR_AXPY("z = exp(x)", cz[i] = exp(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_exp(x)))
//  RUN_TEST_VECTOR_AXPY("z = abs(x)", cz[i] = abs(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_abs(x)))
  RUN_TEST_VECTOR_AXPY("z = acos(x)", cz[i] = acos(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_acos(x)))
  RUN_TEST_VECTOR_AXPY("z = asin(x)", cz[i] = asin(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_asin(x)))
  RUN_TEST_VECTOR_AXPY("z = atan(x)", cz[i] = atan(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_atan(x)))
  RUN_TEST_VECTOR_AXPY("z = ceil(x)", cz[i] = ceil(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_ceil(x)))
  RUN_TEST_VECTOR_AXPY("z = cos(x)", cz[i] = cos(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_cos(x)))
  RUN_TEST_VECTOR_AXPY("z = cosh(x)", cz[i] = cosh(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_cosh(x)))
  RUN_TEST_VECTOR_AXPY("z = exp(x)", cz[i] = exp(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_exp(x)))
  RUN_TEST_VECTOR_AXPY("z = fabs(x)", cz[i] = fabs(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_fabs(x)))
  RUN_TEST_VECTOR_AXPY("z = floor(x)", cz[i] = floor(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_floor(x)))
  RUN_TEST_VECTOR_AXPY("z = log(x)", cz[i] = log(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_log(x)))
  RUN_TEST_VECTOR_AXPY("z = log10(x)", cz[i] = log10(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_log10(x)))
  RUN_TEST_VECTOR_AXPY("z = sin(x)", cz[i] = sin(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_sin(x)))
  RUN_TEST_VECTOR_AXPY("z = sinh(x)", cz[i] = sinh(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_sinh(x)))
  RUN_TEST_VECTOR_AXPY("z = sqrt(x)", cz[i] = sqrt(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_sqrt(x)))
  RUN_TEST_VECTOR_AXPY("z = tan(x)", cz[i] = tan(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_tan(x)))
  RUN_TEST_VECTOR_AXPY("z = tanh(x)", cz[i] = tanh(cx[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_tanh(x)))

  RUN_TEST_VECTOR_AXPY("z = x./y", cz[i] = cx[i]/cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), element_div(x,y)))
  RUN_TEST_VECTOR_AXPY("z = x==y", cz[i] = cx[i]==cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), element_eq(x,y)))
  RUN_TEST_VECTOR_AXPY("z = x>=y", cz[i] = cx[i]>=cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), element_geq(x,y)))
  RUN_TEST_VECTOR_AXPY("z = x>y", cz[i] = cx[i]>cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), element_greater(x,y)))
  RUN_TEST_VECTOR_AXPY("z = x<=y", cz[i] = cx[i]<=cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), element_leq(x,y)))
  RUN_TEST_VECTOR_AXPY("z = x<y", cz[i] = cx[i]<cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), element_less(x,y)))
  RUN_TEST_VECTOR_AXPY("z = x!=y", cz[i] = cx[i]!=cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), element_neq(x,y)))
  RUN_TEST_VECTOR_AXPY("z = pow(x,y)", cz[i] = pow(cx[i], cy[i]), viennacl::scheduler::statement(z, viennacl::op_assign(), element_pow(x,y)))
  RUN_TEST_VECTOR_AXPY("z = x.*y", cz[i] = cx[i]*cy[i], viennacl::scheduler::statement(z, viennacl::op_assign(), element_prod(x,y)))

#undef RUN_TEST_VECTOR_AXPY

  if(failure_count > 0)
      exit(EXIT_FAILURE);
}

template<typename NumericT>
void test_impl(NumericT epsilon)
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

  atidlas::vector_axpy_parameters vector_axpy_parameters(4, 32, 128, atidlas::FETCH_FROM_GLOBAL_CONTIGUOUS);


#define TEST_OPERATIONS(XTYPE, YTYPE, ZTYPE)\
  std::cout << "> x : " #XTYPE " | y : " #YTYPE " | z : " #ZTYPE << std::endl;\
  test_vectors(epsilon, vector_axpy_parameters, x_ ## XTYPE, y_ ## YTYPE, z_ ## ZTYPE);\

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
}

int main()
{
  std::cout << ">> float" << std::endl;
  test_impl<float>(1e-5);
  std::cout << ">> double" << std::endl;
  test_impl<double>(1e-9);
  std::cout << "---" << std::endl;
  std::cout << "Passed" << std::endl;

  return EXIT_SUCCESS;
}
