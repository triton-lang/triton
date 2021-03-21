#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <tuple>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// row-major 3d tensor
class tensor_3d {
public:
  tensor_3d(int size_0, int size_1, int size_2, int *data = nullptr) : data_(size_0 * size_1 * size_2, 0) {
    if (data)
      std::copy(data, data + data_.size(), data_.begin());
    stride_0_ = size_1 * size_2;
    stride_1_ = size_2;
    stride_2_ = 1;
  }

  int &operator()(int i, int j, int k) {
    return data_[i * stride_0_ + j * stride_1_ + k];
  }

private:
  std::vector<int> data_;
  int stride_0_;
  int stride_1_;
  int stride_2_;
};

std::vector<int> segment_blocks(tensor_3d &layout, tensor_3d &idx, int max_width, int H, int M, int N) {
  tensor_3d tmp(H, M, N);
  std::vector<int> current(H, 0);
  int num = 0;
  std::vector<int> lut(H * M * N * 4);
  for (size_t h = 0; h < H; h++) {
    // surrounding indices
    std::vector<int> ii_left(max_width, -1);
    std::vector<std::vector<int>> ii_top(max_width, std::vector<int>(N, -1));
    // start the dynamic programming algorithm
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        int v = layout(h, m, n);
        if (v == 0)
          continue;
        int n_left = ii_left[max_width - 1];
        int m_top = ii_top[max_width - 1][n];
        int top = (m_top >= 0) ? tmp(h, m_top, n) : 0;
        int left = (n_left >= 0) ? tmp(h, m, n_left) : 0;
        int topleft = (m_top >= 0 && n_left >= 0) ? tmp(h, m_top, n_left) : 0;
        int width = std::min(left, std::min(top, topleft)) + 1;
        // reset width if blocks cannot be
        // packed together (i.e., there's a 1 "in the middle")
        for (int nn = n_left + 1; nn < n; nn++)
          if (ii_top[max_width - 1][nn] > ii_top[max_width - 1][n])
            width = 1;
        tmp(h, m, n) = width;
        // update n_left ring buffer
        for (int k = 0; k < max_width - 1; k++)
          ii_left[k] = ii_left[k + 1];
        ii_left[max_width - 1] = n;
        // update ii_top ring buffer
        for (int k = 0; k < max_width - 1; k++)
          ii_top[k][n] = ii_top[k + 1][n];
        ii_top[max_width - 1][n] = m;
        // block is too small -- skip
        if (width != max_width)
          continue;
        // retained blocks are set to zeros
        for (size_t km = 0; km < max_width; km++)
          for (size_t kn = 0; kn < max_width; kn++) {
            int mm = ii_top[km][n];
            int nn = ii_left[kn];
            if (mm < 0 || nn < 0)
              continue;
            layout(h, mm, nn) = 0;
            tmp(h, mm, nn) = 0;
            lut[num++] = (int)h;
            lut[num++] = (int)mm;
            lut[num++] = (int)nn;
            lut[num++] = idx(h, mm, nn);
          }
      }
    }
  }
  lut.resize(num);
  return lut;
}

typedef std::pair<int, pybind11::array_t<int>> lut_t;

std::vector<lut_t> superblock(uintptr_t LAYOUT, int H, int M, int N, int start_width) {
  std::vector<lut_t> ret;
  int current = 0;
  tensor_3d layout(H, M, N, (int *)LAYOUT);
  tensor_3d idx(H, M, N);
  for (int64_t h = 0; h < H; h++)
    for (int64_t m = 0; m < M; m++)
      for (int64_t n = 0; n < N; n++) {
        if (layout(h, m, n) == 0)
          continue;
        idx(h, m, n) = current++;
      }
  // create lut
  for (int max_width = start_width; max_width > 0; max_width /= 2) {
    auto lut = segment_blocks(layout, idx, max_width, H, M, N);
    if (lut.size() == 0)
      continue;
    ret.push_back(std::make_pair(max_width, pybind11::array_t<int>(lut.size(), lut.data())));
  }
  return ret;
}

void init_superblocking(pybind11::module &m) {
  m.def("superblock", &superblock, "super-blocking for block-sparse matrix multiplication");
}