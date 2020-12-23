#include "src/copy.h"
#include "triton/driver/stream.h"
#include "triton/runtime/function.h"
#include "triton/tools/bench.hpp"
#include "triton/external/half.hpp"
#include "util.h"

int32_t off(const std::vector<int32_t>& idx, const std::vector<int32_t>& strides) {
  int32_t res = 0;
  for(size_t d = 0; d < idx.size(); d++)
    res += idx[d] * strides[d];
  return res;
}

struct copy_arg_t{
  CUdeviceptr X;
  CUdeviceptr Y;
  int S0;
  int S1;
  int S2;
};


enum run_mode_t {
  BENCH,
  TEST
};

enum dtype_t {
  FLOAT,
  HALF,
  DOUBLE
};


template<class T>
void cc_copy_nd(const std::vector<T>& x, std::vector<T>& y,
                const std::vector<int32_t>& shape,
                const std::vector<int32_t>& x_order, const std::vector<int32_t>& y_order) {
  size_t rank = shape.size();
  // strides for x
  std::vector<int32_t> x_strides(shape.size());
  for(size_t d = 0; d < rank; d++)
    x_strides[x_order[d]] = (d == 0) ? 1 : (x_strides[x_order[d-1]] * shape[x_order[d-1]]);
  // strides for y
  std::vector<int32_t> y_strides(shape.size());
  for(size_t d = 0; d < rank; d++)
    y_strides[y_order[d]] = (d == 0) ? 1 : (y_strides[y_order[d-1]] * shape[y_order[d-1]]);
  // copy 1d
  if(rank == 1)
    for(int32_t i = 0; i < shape[0]; i++)
      y[off({i}, y_strides)] = x[off({i}, x_strides)];
  // copy 2d
  if(rank == 2)
    for(int32_t i = 0; i < shape[0]; i++)
    for(int32_t j = 0; j < shape[1]; j++)
      y[off({i, j}, y_strides)] = x[off({i, j}, x_strides)];
  // copy 3d
  if(rank == 3)
    for(int32_t i = 0; i < shape[0]; i++)
    for(int32_t j = 0; j < shape[1]; j++)
    for(int32_t k = 0; k < shape[2]; k++)
      y[off({i, j, k}, y_strides)] = x[off({i, j, k}, x_strides)];
}

template<class T>
struct to_string;

template<> struct to_string<half_float::half>{
  static constexpr const char* value = "half";
};

template<> struct to_string<float>{
  static constexpr const char* value = "float";
};

template<> struct to_string<double>{
  static constexpr const char* value = "double";
};

template<typename T>
void triton_copy_nd(drv::context* context, drv::stream* stream, const std::vector<int32_t>& shape,
                    const std::vector<int32_t>& x_order, const std::vector<int32_t>& y_order,
                    std::vector<std::vector<std::string>> TS,
                    run_mode_t mode, std::vector<double>& bench, bool &test) {
  std::string ty = to_string<T>::value;
  size_t dtsize = sizeof(T);
  drv::device* device = context->device();

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


  // macros
  opt.defines.push_back({"TYPE", {ty}});
  for(size_t d = 0; d < rank; d++)
    opt.defines.push_back({"STRIDE_XS" + std::to_string(x_order[d]), {x_strides[d]}});
  for(size_t d = 0; d < rank; d++)
    opt.defines.push_back({"STRIDE_YS" + std::to_string(y_order[d]), {y_strides[d]}});
  if(TS.empty())
    TS = tile_nd(rank);
  for(size_t d = 0; d < rank; d++)
    opt.defines.push_back({"TS" + std::to_string(d), TS[d]});
  opt.num_warps = {4};

  // kernel
  rt::function function(src::copy_nd[rank - 1], opt);
  copy_arg_t args = {*dx->cu(), *dy->cu(), shape[0]};
  if(shape.size() > 1) args.S1 = shape[1];
  if(shape.size() > 2) args.S2 = shape[2];
  std::vector<std::string> ts = {"TS0", "TS1", "TS2"};
  auto grid = grid_nd(shape, ts);

  // metrics
  if(mode == BENCH){
    auto gbps = [&](double ns) { return 2 * size * dtsize / (ns * 1e-9) * 1e-9; };
    double triton_ns = triton::tools::bench([&]() { function((void**)&args, sizeof(args), grid, stream, device);}, stream);
    bench.push_back(gbps(triton_ns));
  }

  // test triton
  if(mode == TEST){
    std::vector<T> hx(size);
    std::vector<T> hy(size);
    std::vector<T> ry(size);
    for(size_t i = 0; i < hx.size(); i++)
      hx[i] = static_cast<T>((float)rand()/RAND_MAX);
    stream->write(&*dx, true, 0, hx);
    function((void**)&args, sizeof(args), grid, stream, device);
    stream->synchronize();
    stream->read(&*dy, true, 0, hy);
    cc_copy_nd(hx, ry, shape, x_order, y_order);
    test = testing::diff(hy, ry);
  }
}

std::vector<double> bench_copy_nd(drv::context* context, drv::stream* stream, dtype_t dtype, const std::vector<int32_t>& shape,
                                  const std::vector<int32_t>& x_order, const std::vector<int32_t>& y_order) {
  std::vector<double> bench;
  bool test;
  switch(dtype){
    case HALF:
      triton_copy_nd<half_float::half>(context, stream, shape, x_order, y_order, {}, BENCH, bench, test);
      break;
    case FLOAT:
      triton_copy_nd<float>(context, stream, shape, x_order, y_order, {}, BENCH, bench, test);
      break;
    default: break;
  }
  return bench;
}

bool test_copy_nd(drv::context* context, drv::stream* stream, dtype_t dtype, const std::vector<int32_t>& shape,
                  const std::vector<int32_t>& TS,
                  const std::vector<int32_t>& x_order, const std::vector<int32_t>& y_order) {
  std::vector<double> bench;
  bool test;
  std::vector<std::vector<std::string>> TSS;
  for(int32_t d: TS)
    TSS.push_back({std::to_string(d)});
  switch(dtype){
    case HALF:
      triton_copy_nd<half_float::half>(context, stream, shape, x_order, y_order, TSS, TEST, bench, test);
      break;
    case FLOAT:
      triton_copy_nd<float>(context, stream, shape, x_order, y_order, TSS, TEST, bench, test);
      break;
    default: break;
  }
  return test;
}
