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

struct reduce_arg_t{
  CUdeviceptr X;
  CUdeviceptr Y;
  int S0;
  int S1;
  int S2;
};


template<class T>
void cc_reduce_nd(std::vector<T> &y, const std::vector<T> &x, reduce_op_t op, size_t axis, const std::vector<int>& shapes) {
  assert(axis <= shapes.size() - 1);
  // remove shape at index axis to get outer dimensions
  std::vector<int> outer = shapes;
  outer.erase(outer.begin() + axis);
  if(outer.empty())
    outer.push_back(1);
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

void triton_reduce_nd(drv::context* context, drv::stream* stream, const std::vector<int32_t>& shape_x,
                      int axis, reduce_op_t op,
                      const std::vector<int32_t>& x_order, const std::vector<int32_t>& y_order,
                      std::vector<std::vector<std::string>> TS,
                      run_mode_t mode, std::vector<double>& bench, bool &test) {
  typedef float NumericT;
  std::string ty = "float";
  size_t dtsize = sizeof(NumericT);
  drv::device* device = context->device();



  // shape
  std::vector<int> shape_y = shape_x;
  shape_y.erase(shape_y.begin() + axis);

  // rank
  int rank_x = shape_x.size();
  int rank_y = shape_y.size();

  // size
  size_t size_x = 1;
  for(int32_t d: shape_x)
    size_x *= d;
  size_t size_y = 1;
  for(int32_t d: shape_y)
    size_y *= d;

  // strides for x
  std::vector<std::string> x_shapename = {"S0", "S1", "S2"};
  std::vector<std::string> x_strides = {"1"};
  for(int d = 0; d < rank_x - 1; d++)
    x_strides.push_back(x_strides[d] + " * " + x_shapename[x_order[d]]);

  // strides for y
  std::vector<std::string> y_shapename = x_shapename;
  y_shapename.erase(y_shapename.begin() + axis);
  std::vector<std::string> y_strides = {"1"};
  for(int d = 0; d < rank_y - 1; d++)
    y_strides.push_back(y_strides[d] + " * " + y_shapename[y_order[d]]);

  // options
  rt::function::options_space_t opt;
  opt.defines.push_back({"TYPE", {ty}});
  for(int d = 0; d < rank_x; d++)
    opt.defines.push_back({"STRIDE_XS" + std::to_string(x_order[d]), {x_strides[d]}});
  for(int d = 0; d < rank_y; d++)
    opt.defines.push_back({"STRIDE_YS" + std::to_string(y_order[d]), {y_strides[d]}});
  if(TS.empty())
    TS = tile_nd(rank_x);
  for(int d = 0; d < rank_x; d++)
    opt.defines.push_back({"TS" + std::to_string(d), TS[d]});

  std::vector<size_t> axy;
  for(int d = 0; d < rank_x; d++)
    if(d != axis)
      axy.push_back(d);
  for(int d = 0; d < rank_y; d++)
    opt.defines.push_back({"TY" + std::to_string(d), {std::to_string(shape_x[axy[d]])}});
  for(int d = 0; d < rank_y; d++)
    opt.defines.push_back({"RY" + std::to_string(d), {"rs" + std::to_string(axy[d])}});

  std::string RED = "";
  for(int n = 0; n < rank_x; n++){
    if(n > 0)
      RED += ", ";
    RED += (n==axis) ? to_str(op) : ":";
  }
  opt.defines.push_back({"RED", {RED}});
  opt.num_warps = {1};

  // kernel
  rt::function function(src::reduce_nd[rank_x - 1], opt);

  // input buffers
  auto dx = std::unique_ptr<drv::buffer>(drv::buffer::create(context, size_x*dtsize));
  auto dy = std::unique_ptr<drv::buffer>(drv::buffer::create(context, size_y*dtsize));

  // grid
  reduce_arg_t args = {*dx->cu(), *dy->cu(), shape_x[0]};
  if(shape_x.size() > 1) args.S1 = shape_x[1];
  if(shape_x.size() > 2) args.S2 = shape_x[2];
  std::vector<std::string> ts = {"TS0", "TS1", "TS2"};
  auto grid = grid_nd(shape_x, ts);

  // metrics
  if(mode == BENCH){
    auto gbps = [&](double ns) { return 2 * size_x * dtsize / (ns * 1e-9) * 1e-9; };
    double triton_ns = triton::tools::bench([&]() { function((void**)&args, sizeof(args), grid, stream, device);}, stream);
    bench.push_back(gbps(triton_ns));
  }

  // test triton
  if(mode == TEST){
    std::vector<NumericT> hy(size_y);
    std::vector<NumericT> ry(size_y);
    std::vector<NumericT> hx(size_x);
    init_zeros(hy);
    init_rand(hx);
    stream->write(&*dx, true, 0, hx);
    function((void**)&args, sizeof(args), grid, stream, device);
    stream->synchronize();
    stream->read(&*dy, true, 0, hy);
    cc_reduce_nd(ry, hx, op, axis, shape_x);
    test = testing::diff(hy, ry);
  }
}

bool do_test(drv::context* context, drv::stream* stream, std::vector<int> shape, int axis, reduce_op_t op, int nwarp){
  std::vector<double> bench;
  bool test;
  std::vector<std::vector<std::string>> TSS;
  for(int32_t d: shape)
    TSS.push_back({std::to_string(d)});
  triton_reduce_nd(context, stream, shape, axis, op, {0, 1, 2}, {0, 1, 2}, TSS, TEST, bench, test);
  return test;
}
