#include <iostream>
#include <string>
#include <variant>
#include <vector>

// #include <c10/cuda/CUDAFunctions.h>
// #include <c10/cuda/CUDAStream.h>
#include <c10/util/Logging.h>
#include <torch/extension.h>

const auto &torch_py_module = py::module_::import("torch");

using ValueType = std::variant<std::monostate, int, float, torch::Tensor>;

class SignatureParam {
public:
  const std::string name_;
  const int num_;
  const int kind_;
  const bool do_not_specialize_;
  const bool is_constexpr_;
  const std::string annotation_;
  const ValueType value_;
  const int divisibility_;
  const int divisibility_8_;

  static SignatureParam from_py_object(const py::object &param,
                                       const int divisibility,
                                       const int divisibility_8) {
    ValueType value = std::monostate{};
    if (param.attr("has_default").cast<bool>()) {
      if (py::isinstance<py::int_>(param.attr("default"))) {
        value = param.attr("default").cast<int>();
      } else if (py::isinstance<py::float_>(param.attr("default"))) {
        value = param.attr("default").cast<float>();
      } else {
        throw std::runtime_error("Unsupported default value type");
      }
    }
    return SignatureParam(param.attr("name").cast<std::string>(),
                          param.attr("num").cast<int>(),
                          param.attr("_param").attr("kind").cast<int>(),
                          param.attr("do_not_specialize").cast<bool>(),
                          param.attr("is_constexpr").cast<bool>(),
                          param.attr("annotation").cast<std::string>(),
                          divisibility, divisibility_8, value);
  }

  SignatureParam(const std::string &name, const int num, const int kind,
                 const bool do_not_specialize, const bool is_constexpr,
                 const std::string &annotation, const int divisibility,
                 const int divisibility_8,
                 const ValueType value = std::monostate{})
      : name_(name), num_(num), kind_(kind),
        do_not_specialize_(do_not_specialize), is_constexpr_(is_constexpr),
        annotation_(annotation), divisibility_(divisibility),
        divisibility_8_(divisibility_8), value_(value) {
    // LOG(INFO) << "Initialize SignatureParam " << name_;
  }

  std::string
  get_signature_key(const ValueType &value = std::monostate{}) const {
    if (annotation_.find("Tensor") != std::string::npos) {
      return "Tensor";
    } else if (annotation_ == "bool") {
      return "i1";
    } else if (annotation_ == "float") {
      return "fp32";
    } else if (std::holds_alternative<int>(value)) {
      const auto val = std::get<int>(value);
      if ((val >= -(2 << 31)) && (val <= (2 << 31) - 1)) {
        return "i32";
      } else if (val > 0) {
        return "u64";
      } else {
        return "i64";
      }
    } else if (std::holds_alternative<float>(value)) {
      return "fp32";
    } else if (std::holds_alternative<torch::Tensor>(value)) {
      // LOG(INFO) << "get_signature_key: Tensor";
      const auto &dtype = std::get<torch::Tensor>(value).dtype();
      if (dtype == torch::kFloat32) {
        return "torch.float32";
      } else if (dtype == torch::kFloat16) {
        return "torch.float16";
      } else {
        return "torch" + std::string(dtype.name());
      }
    } else if (std::holds_alternative<std::monostate>(value)) {
      return "None";
    } else {
      throw std::runtime_error("Unsupported dtype");
    }
  }

  std::vector<py::object>
  get_specialization_key(const ValueType &value = std::monostate{}) const {
    std::vector<py::object> spec_key;
    if (std::holds_alternative<int>(value)) {
      const auto &val = std::get<int>(value);
      spec_key.emplace_back(py::bool_(val % divisibility_ == 0));
      spec_key.emplace_back(py::bool_(val % divisibility_8_ == 0));
      spec_key.emplace_back(py::bool_(val == 1));
    } else if (std::holds_alternative<float>(value)) {
      spec_key.emplace_back(py::bool_(false));
    } else if (std::holds_alternative<torch::Tensor>(value)) {
      // LOG(INFO) << "get_signature_key: Tensor";
      const auto &val = std::get<torch::Tensor>(value);
      const auto data_ptr = reinterpret_cast<const uint64_t>(val.data_ptr());
      spec_key.emplace_back(py::bool_(data_ptr % divisibility_ == 0));
    } else {
      spec_key.emplace_back(py::bool_(false));
    }
    return spec_key;
  }

  void print() const {
    LOG(WARNING) << "parameter " << name_
                 << ", do_not_specialize = " << do_not_specialize_
                 << ", kind = " << kind_ << ", num = " << num_
                 << ", is_constexpr = " << is_constexpr_
                 << ", annotation = " << annotation_;
    if (std::holds_alternative<int>(value_)) {
      LOG(WARNING) << "default = " << std::get<int>(value_);
    } else if (std::holds_alternative<float>(value_)) {
      LOG(WARNING) << "default = " << std::get<float>(value_);
    }
  }
};

class LowLatencyJITFunction {
  std::vector<int> do_not_specialize_;
  std::vector<SignatureParam> signature_params_;
  std::string cuda_version_key_;
  const int divisibility_;
  const int divisibility_8_;

public:
  LowLatencyJITFunction(const int divisibility, const int divisibility_8,
                   const std::vector<py::object> &params,
                   const std::string &cuda_version_key)
      : divisibility_(divisibility), divisibility_8_(divisibility_8),
        cuda_version_key_(cuda_version_key) {
    LOG(INFO) << "LowLatencyJITFunction::init";
    for (const auto &x : params) {
      signature_params_.emplace_back(
          SignatureParam::from_py_object(x, divisibility_, divisibility_8_));
      LOG(INFO) << "idx = " << x.attr("num").cast<int>()
                << ", name = " << x.attr("name").cast<std::string>()
                << ", annotation = "
                << x.attr("annotation").cast<std::string>();
    }
  }

  py::tuple get_call_params_tuple(const std::vector<py::object> &args,
                                  const py::dict &kwargs) const {
    // LOG(INFO) << "args = " << args << ", kwargs = " << kwargs;

    std::vector<std::string> sig_key;
    std::vector<py::object> constexpr_key, spec_key, non_constexpr_arg_values;

    size_t idx = 0;
    const size_t args_size = args.size();
    for (; idx < args_size; ++idx) {
      const auto &s_param = signature_params_[idx];
      const auto &arg = args[idx];

      if (s_param.is_constexpr_) {
        constexpr_key.push_back(arg);
        if (!s_param.do_not_specialize_) {
          spec_key.emplace_back(py::tuple(
              py::cast(s_param.get_specialization_key(arg.cast<int>()))));
        }
      } else {
        non_constexpr_arg_values.push_back(arg);
        if (py::isinstance<py::int_>(arg)) {
          sig_key.emplace_back(s_param.get_signature_key(arg.cast<int>()));
          if (!s_param.do_not_specialize_) {
            spec_key.emplace_back(py::tuple(
                py::cast(s_param.get_specialization_key(arg.cast<int>()))));
          }
        } else if (py::isinstance<py::float_>(arg)) {
          sig_key.emplace_back(s_param.get_signature_key(arg.cast<float>()));
          if (!s_param.do_not_specialize_) {
            spec_key.emplace_back(py::tuple(
                py::cast(s_param.get_specialization_key(arg.cast<float>()))));
          }
        } else {
          if (torch_py_module.attr("is_tensor")(arg).cast<bool>()) {
            // LOG(INFO) << "is_tensor";
            sig_key.emplace_back(
                s_param.get_signature_key(arg.cast<torch::Tensor>()));
            if (!s_param.do_not_specialize_) {
              spec_key.emplace_back(py::tuple(py::cast(
                  s_param.get_specialization_key(arg.cast<torch::Tensor>()))));
            }
          } else {
            throw std::runtime_error("Unsupported type");
          }
        }
      }
    }
    if (idx != args_size) {
      throw std::runtime_error("Wrong number of params");
    }

    return py::make_tuple(py::make_tuple(cuda_version_key_,
                                         py::tuple(py::cast(sig_key)),
                                         py::tuple(py::cast(constexpr_key)),
                                         py::tuple(py::cast(spec_key))),
                          py::tuple(py::cast(non_constexpr_arg_values)));
  }

  void print() const { LOG(WARNING) << "LowLatencyJITFunction::print"; }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // py::class_<SignatureParam>(m, "SignatureParam")
  // .def(py::init<const std::string&, const int, const int, std::variant<int,
  // float, std::string>>()) .def("print", &SignatureParam::print);

  py::class_<LowLatencyJITFunction>(m, "LowLatencyJITFunction")
      .def(py::init<const int, const int, const std::vector<py::object> &,
                    const std::string &>(),
           py::arg("divisibility"), py::arg("divisibility_8"),
           py::arg("params") = std::vector<py::object>(),
           py::arg("cuda_version_key") = std::string())
      .def("print", &LowLatencyJITFunction::print)
      .def("get_call_params_tuple", &LowLatencyJITFunction::get_call_params_tuple);
}
