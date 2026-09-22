#ifndef PROTON_UTILITY_ERRORS_H_
#define PROTON_UTILITY_ERRORS_H_

#include <stdexcept>
#include <string>
#include <utility>

namespace proton {

inline constexpr const char *kProtonErrorPrefix = "[PROTON] ";

inline std::string prefixErrorMessage(std::string message) {
  return std::string(kProtonErrorPrefix) + message;
}

inline std::runtime_error makeRuntimeError(std::string message) {
  return std::runtime_error(prefixErrorMessage(std::move(message)));
}

inline std::invalid_argument makeInvalidArgument(std::string message) {
  return std::invalid_argument(prefixErrorMessage(std::move(message)));
}

inline std::out_of_range makeOutOfRange(std::string message) {
  return std::out_of_range(prefixErrorMessage(std::move(message)));
}

inline std::length_error makeLengthError(std::string message) {
  return std::length_error(prefixErrorMessage(std::move(message)));
}

inline std::logic_error makeLogicError(std::string message) {
  return std::logic_error(prefixErrorMessage(std::move(message)));
}

class NotImplemented : public std::logic_error {
public:
  NotImplemented()
      : std::logic_error(prefixErrorMessage("Not yet implemented")) {}
};

} // namespace proton

#endif // PROTON_UTILITY_ERRORS_H_
