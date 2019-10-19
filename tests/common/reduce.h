#include <iomanip>
#include <cstring>
#include <sstream>
#include <cstdio>
#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/tools/bench.hpp"
#include "triton/external/half.hpp"
#include "triton/runtime/function.h"
#include "src/reduce.h"
#include "util.h"

namespace drv = triton::driver;
namespace rt = triton::runtime;

template<class T>
void cc_reduce_nd(std::vector<T> &y, const std::vector<T> &x, reduce_op_t op, size_t axis, const std::vector<int>& shapes) {
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

enum run_mode_t {
  BENCH,
  TEST
};

void triton_reduce_nd(drv::stream* stream, const std::vector<int32_t>& shape,
                      int axis, reduce_op_t op,
                      const std::vector<int32_t>& x_order, const std::vector<int32_t>& y_order,
                      std::vector<std::vector<std::string>> TS,
                      run_mode_t mode, std::vector<double>& bench, bool &test) {
  typedef float NumericT;
  std::string ty = "float";
  size_t dtsize = sizeof(NumericT);
  drv::context* context = stream->context();
  size_t axy = (axis == 0) ? 1 : 0;

  // rank
  size_t rank = shape.size();
  // size
  size_t size = 1;
  for(int32_t d: shape)
    size *= d;
  std::vector<std::string> shapename = {"S0", "S1", "S2"};
  // strides for x
  std::vector<std::string> x_strides = {"1"};
  for(size_t d = 0; d < rank - 1; d++)
    x_strides.push_back(x_strides[d] + " * " + shapename[x_order[d]]);
  // strides for y
  std::vector<std::string> y_strides = {"1"};
  for(size_t d = 0; d < rank - 1; d++)
    y_strides.push_back(y_strides[d] + " * " + shapename[y_order[d]]);

  // create inputs
  auto dx = std::unique_ptr<drv::buffer>(drv::buffer::create(context, size*dtsize));
  auto dy = std::unique_ptr<drv::buffer>(drv::buffer::create(context, size*dtsize));
  // create options
  rt::function::options_space_t opt;

  // type
  opt.defines.push_back({"TYPE", {ty}});
  // x strides
  for(size_t d = 0; d < rank; d++)
    opt.defines.push_back({"STRIDE_XS" + std::to_string(x_order[d]), {x_strides[d]}});
  // y strides
  for(size_t d = 0; d < rank; d++)
    opt.defines.push_back({"STRIDE_YS" + std::to_string(y_order[d]), {y_strides[d]}});
  if(TS.empty())
    TS = tile_nd(rank);
  // tile size
  for(size_t d = 0; d < rank; d++)
    opt.defines.push_back({"TS" + std::to_string(d), TS[d]});
  // non-reduced axis
  std::string RY = (axis == 0) ? "rn" : "rm";
  opt.defines.push_back({"TY", {std::to_string(shape[axy])}});
  opt.defines.push_back({"RY", {RY}});
  // reduction broadcasting
  std::string RED = "";
  for(int n = 0; n < 2; n++){
    if(n > 0)
      RED += ", ";
    RED += (n==axis) ? to_str(op) : ":";
  }
  opt.defines.push_back({"RED", {RED}});

  opt.num_warps = {4};

  // kernel
  rt::function function(src::reduce2d, opt);

  // grid
  std::vector<rt::arg> args = {&*dx, &*dy};
  for(int32_t d: shape)
    args.push_back(d);
  args.push_back(shape[0]);
  std::vector<std::string> ts = {"TS0", "TS1", "TS2"};
  auto grid = grid_nd(shape, ts);

  // metrics
  if(mode == BENCH){
    auto gbps = [&](double ns) { return 2 * size * dtsize / (ns * 1e-9) * 1e-9; };
    double triton_ns = triton::tools::bench([&]() { function(args, grid, stream);}, stream);
    bench.push_back(gbps(triton_ns));
  }

  // test triton
  if(mode == TEST){
    std::vector<NumericT> hy(shape[axy]);
    std::vector<NumericT> ry(shape[axy]);
    std::vector<NumericT> hx(shape[0]*shape[1]);
    init_zeros(hy);
    init_rand(hx);
    stream->write(&*dx, true, 0, hx);
    function(args, grid, stream);
    stream->synchronize();
    stream->read(&*dy, true, 0, hy);
    cc_reduce_nd(ry, hx, op, axis, shape);
    test = testing::diff(hy, ry);
  }
}

bool do_test(drv::stream* stream, std::vector<int> shape, int axis, reduce_op_t op, int nwarp){
  std::vector<double> bench;
  bool test;
  std::vector<std::vector<std::string>> TSS;
  for(int32_t d: shape)
    TSS.push_back({std::to_string(d)});
  triton_reduce_nd(stream, shape, axis, op, {0, 1}, {0, 1}, TSS, TEST, bench, test);
  return test;
}
