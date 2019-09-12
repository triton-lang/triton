#include <iomanip>
#include <cstring>
#include <sstream>
#include <cstdio>
#include <functional>
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/tools/bench.hpp"
#include "triton/external/half.hpp"
#include "triton/runtime/function.h"
#include "src/reduce.h"
#include "cuda/cublas.h"
#include "util.h"

namespace drv = triton::driver;
namespace rt = triton::runtime;

void _loop_nest(std::vector<int> const & ranges,
                std::function<void(std::vector<int> const &)> const & f){
  int D = ranges.size();
  std::vector<int> values(D, 0);
  // Start with innermost loop
  int i = D - 1;
  while(true){
    //  Execute function
    f(values);
    while(values[i]++ == ranges[i] - 1){
      if(i == 0)
        return;
      values[i--] = 0;
    }
    i = D - 1;
  }
}

int offset(const std::vector<int>& idx, const std::vector<int>& shapes) {
  int result = idx[0];
  for(int i = 1; i < idx.size(); i++)
    result += idx[i]*shapes[i-1];
  return result;
}

template<class T>
void reduce_nd(std::vector<T> &y, const std::vector<T> &x, reduce_op_t op, size_t axis, const std::vector<int>& shapes) {
  assert(axis <= shapes.size() - 1);
  // remove shape at index axis to get outer dimensions
  std::vector<int> outer = shapes;
  outer.erase(outer.begin() + axis);
  // retrieve shape at index axis to get inner dimension
  int inner = shapes[axis];
  // accumualtion function
  auto acc = get_accumulator<T>(op);
  // iterate over outer dimensions
  _loop_nest(outer, [&](const std::vector<int>& y_idx) {
    T ret = 0;
    auto x_idx = y_idx;
    x_idx.insert(x_idx.begin() + axis, 0);
    // accumulate over inner dimensions
    for(int z = 0; z < inner; z++){
      x_idx[axis] = z;
      ret = acc(ret, x[offset(x_idx, shapes)]);
    }
    y[offset(y_idx, outer)] = ret;
  });
}


bool do_test(drv::stream* stream, std::vector<int> shape, int axis, reduce_op_t op, int nwarp){
  typedef float NumericT;
  std::string ty = "float";
  size_t dt_nbytes = sizeof(NumericT);
  drv::context* context = stream->context();
  size_t axy = (axis == 0) ? 1 : 0;
  std::string RY = (axis == 0) ? "rn" : "rm";
  std::vector<NumericT> hy(shape[axy]);
  std::vector<NumericT> ry(shape[axy]);
  std::vector<NumericT> hx(shape[0]*shape[1]);
  srand(0);
  init_zeros(hy);
  init_rand(hx);
  auto dy = std::shared_ptr<drv::buffer>(drv::buffer::create(context, hy.size()*dt_nbytes));
  auto dx = std::shared_ptr<drv::buffer>(drv::buffer::create(context, hx.size()*dt_nbytes));
  stream->write(&*dy, true, 0, hy);
  stream->write(&*dx, true, 0, hx);
  rt::function::options_space_t opt;
  opt.defines.push_back({"TYPE", {ty}});
  opt.defines.push_back({"TM", {std::to_string(shape[0])}});
  opt.defines.push_back({"TN", {std::to_string(shape[1])}});
  opt.defines.push_back({"TY", {std::to_string(shape[axy])}});
  opt.defines.push_back({"RY", {RY}});
  std::string RED = "";
  for(int n = 0; n < 2; n++){
    if(n > 0)
      RED += ", ";
    RED += (n==axis) ? to_str(op) : ":";
  }
  opt.defines.push_back({"RED", {RED}});
  opt.num_warps = {nwarp};
  rt::function function(src::reduce2d, opt);
  function({&*dx, &*dy, shape[0], shape[1], shape[0]}, grid2d(shape[0], shape[1]), stream);
  stream->synchronize();
  stream->read(&*dy, true, 0, hy);
  reduce_nd(ry, hx, op, axis, shape);
  return testing::diff(hy, ry);
}

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream* stream = triton::driver::stream::create(context);
  // shapes to benchmark
  typedef std::tuple<std::vector<int>, int, reduce_op_t> config_t;
  std::vector<config_t> configs = {
    config_t{{32, 32}, 0, MAX},
    config_t{{32, 32}, 1, ADD},
    config_t{{32, 64}, 0, ADD},
    config_t{{64, 32}, 1, ADD}
  };
  // does the work
  int axis;
  std::vector<int> shape;
  reduce_op_t op;
  for(const auto& c: configs){
    std::tie(shape, axis, op) = c;
    std::cout << "Testing " << c << " ... " << std::flush;
    if(do_test(stream, shape, axis, op, 1))
      std::cout << " Pass! " << std::endl;
    else
      std::cout << " Fail! " << std::endl;
  }
}
