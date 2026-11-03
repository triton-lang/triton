#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/STLExtras.h"
#include <iostream>
#include <optional>
#include <stdexcept>

namespace py = nanobind;
using LinearLayout = mlir::triton::LinearLayout;

namespace {

mlir::MLIRContext *getLinearLayoutContext() {
  // Process-lifetime singleton: LinearLayout objects hold attributes uniqued in
  // this context, so the context must outlive every Python LinearLayout object.
  // We also deliberately avoid running its destructor during interpreter
  // shutdown (to avoid segfaults).
  //
  // Do not create this via triton._C.libtriton.ir.context and release the
  // Python object. That still intentionally keeps the context alive, but it
  // also leaves a nanobind-owned instance registered at interpreter shutdown,
  // which reports leaked ir.context/types/functions. Keeping only the raw C++
  // context avoids leaking a Python/nanobind object.
  static auto *ctx =
      new mlir::MLIRContext(mlir::MLIRContext::Threading::DISABLED);
  return ctx;
}

} // namespace

void init_linear_layout(py::module_ &m) {
  py::class_<LinearLayout>(m, "LinearLayout")
      .def(py::init<>())
      .def_static(
          "identity_1d",
          [](int32_t size, std::string inDim, std::string outDim) {
            auto *ctx = getLinearLayoutContext();
            return LinearLayout::identity1D(size,
                                            mlir::StringAttr::get(ctx, inDim),
                                            mlir::StringAttr::get(ctx, outDim));
          },
          py::arg("size"), py::arg("inDim"), py::arg("outDim"))
      .def_static(
          "strided_1d",
          [](int32_t size, int32_t stride, std::string inDim,
             std::string outDim) {
            auto *ctx = getLinearLayoutContext();
            return LinearLayout::strided1D(size, stride,
                                           mlir::StringAttr::get(ctx, inDim),
                                           mlir::StringAttr::get(ctx, outDim));
          },
          py::arg("size"), py::arg("stride"), py::arg("inDim"),
          py::arg("outDim"))
      .def_static(
          "zeros_1d",
          [](int32_t size, std::string inDim, std::string outDim,
             int32_t outDimSize) {
            auto *ctx = getLinearLayoutContext();
            return LinearLayout::zeros1D(
                size, mlir::StringAttr::get(ctx, inDim),
                mlir::StringAttr::get(ctx, outDim), outDimSize);
          },
          py::arg("size"), py::arg("inDim"), py::arg("outDim"),
          py::arg("outDimSize") = 1)
      .def_static(
          "from_bases",
          [](const std::vector<std::pair<
                 std::string, std::vector<std::vector<int32_t>>>> &bases,
             const std::vector<std::string> &outDimNames,
             std::optional<std::vector<int32_t>> outDimSizes,
             bool requireSurjective) {
            auto *ctx = getLinearLayoutContext();

            std::vector<
                std::pair<mlir::StringAttr, std::vector<std::vector<int32_t>>>>
                convertedBases;
            convertedBases.reserve(bases.size());
            for (const auto &entry : bases) {
              std::vector<std::vector<int32_t>> converted;
              converted.reserve(entry.second.size());
              for (const auto &vec : entry.second)
                converted.emplace_back(vec.begin(), vec.end());
              convertedBases.emplace_back(
                  mlir::StringAttr::get(ctx, entry.first),
                  std::move(converted));
            }

            if (outDimSizes) {
              if (outDimSizes->size() != outDimNames.size())
                throw std::invalid_argument("out_dim_names and out_dim_sizes "
                                            "must have the same length");
              std::vector<std::pair<mlir::StringAttr, int32_t>> outDims;
              outDims.reserve(outDimNames.size());
              for (auto it : llvm::enumerate(outDimNames))
                outDims.emplace_back(mlir::StringAttr::get(ctx, it.value()),
                                     (*outDimSizes)[it.index()]);
              return LinearLayout(convertedBases, outDims, requireSurjective);
            }

            if (!requireSurjective)
              throw std::invalid_argument("out_dim_sizes must be provided when "
                                          "require_surjective is false");

            std::vector<mlir::StringAttr> convertedNames;
            convertedNames.reserve(outDimNames.size());
            for (const auto &name : outDimNames)
              convertedNames.push_back(mlir::StringAttr::get(ctx, name));
            return LinearLayout(convertedBases, convertedNames);
          },
          py::arg("bases"), py::arg("out_dim_names"),
          (py::arg("out_dim_sizes").none() = py::none()),
          py::arg("require_surjective") = true)
      .def("compose", &LinearLayout::compose)
      .def("invert_and_compose", &LinearLayout::invertAndCompose)
      .def("invert", &LinearLayout::invert)
      .def("pseudoinvert", &LinearLayout::pseudoinvert)
      .def("is_surjective", &LinearLayout::isSurjective)
      .def("is_injective", &LinearLayout::isInjective)
      .def("is_invertible", &LinearLayout::isInvertible)
      .def("get_in_dim_names",
           [](const LinearLayout &self) {
             std::vector<std::string> dims;
             dims.reserve(self.getNumInDims());
             for (mlir::StringAttr dim : self.getInDimNames())
               dims.push_back(dim.str());
             return dims;
           })
      .def("get_out_dim_names",
           [](const LinearLayout &self) {
             std::vector<std::string> dims;
             dims.reserve(self.getNumOutDims());
             for (mlir::StringAttr dim : self.getOutDimNames())
               dims.push_back(dim.str());
             return dims;
           })
      .def_prop_ro("bases",
                   [](const LinearLayout &self) {
                     auto bases = self.getBases();
                     py::list result;
                     for (const auto &it : bases) {
                       py::list dimBases;
                       for (const auto &vec : it.second)
                         dimBases.append(py::cast(
                             std::vector<int32_t>(vec.begin(), vec.end())));
                       result.append(py::make_tuple(it.first.str(), dimBases));
                     }
                     return result;
                   })
      .def_prop_ro("out_dims",
                   [](const LinearLayout &self) {
                     py::list result;
                     for (const auto &it : self.getOutDims()) {
                       result.append(py::make_tuple(it.first.str(), it.second));
                     }
                     return result;
                   })
      .def_prop_ro("num_in_dims", &LinearLayout::getNumInDims)
      .def_prop_ro("num_out_dims", &LinearLayout::getNumOutDims)
      .def("__mul__", [](const LinearLayout &lhs,
                         const LinearLayout &rhs) { return lhs * rhs; })
      .def(
          "__imul__",
          [](LinearLayout &lhs, const LinearLayout &rhs) -> LinearLayout & {
            lhs *= rhs;
            return lhs;
          },
          py::rv_policy::reference_internal)
      .def("__eq__", [](const LinearLayout &lhs,
                        const LinearLayout &rhs) { return lhs == rhs; })
      .def("__ne__", [](const LinearLayout &lhs,
                        const LinearLayout &rhs) { return lhs != rhs; })
      .def("__repr__", [](const LinearLayout &self) { return self.toString(); })
      .def("__str__", [](const LinearLayout &self) { return self.toString(); })
      .def("get_shared_view",
           [](const LinearLayout &self, bool useHWPointOfView) {
             return mlir::triton::gpu::getSharedLayoutStr(
                 const_cast<LinearLayout &>(self), useHWPointOfView);
           })
      .def("get_distributed_view",
           [](const LinearLayout &self, bool useHWPointOfView) {
             return mlir::triton::gpu::getDistributedLayoutStr(
                 const_cast<LinearLayout &>(self), useHWPointOfView);
           })
      .def(
          "apply",
          [](const LinearLayout &self, py::dict inputsDict) {
            std::vector<std::pair<std::string, int32_t>> inputs;
            inputs.reserve(inputsDict.size());
            for (auto item : inputsDict) {
              inputs.emplace_back(py::cast<std::string>(item.first),
                                  py::cast<int32_t>(item.second));
            }
            auto *ctx = getLinearLayoutContext();
            std::vector<std::pair<mlir::StringAttr, int32_t>> converted;
            converted.reserve(inputs.size());
            for (const auto &it : inputs) {
              converted.emplace_back(mlir::StringAttr::get(ctx, it.first),
                                     it.second);
            }
            auto outputs = self.apply(converted);
            py::dict result;
            for (const auto &out : outputs) {
              auto s = out.first.str();
              result[py::str(s.c_str(), s.size())] = out.second;
            }
            return result;
          },
          py::arg("inputs"))
      .def("get_matrix_view", [](const LinearLayout &self) {
        std::unique_ptr<uint64_t[]> matrix = mlir::triton::getMatrix(self);
        auto nRows = self.getTotalOutDimSizeLog2();
        auto nCols = self.getTotalInDimSizeLog2();
        std::vector<std::vector<int>> result(nRows, std::vector<int>(nCols));
        for (size_t i = 0; i < nRows; ++i) {
          for (size_t j = 0; j < nCols; ++j) {
            result[i][j] = (matrix[i] >> j) & 1;
          }
        }
        return result;
      });
}
