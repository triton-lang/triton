#ifndef PROTON_COMMON_PARSER_H_
#define PROTON_COMMON_PARSER_H_

#include "ByteSpan.h"
#include "Device.h"
#include "EntryDecoder.h"
#include <cstdint>
#include <stdexcept>

namespace proton {

struct ParserConfig {
  enum class PrintMode {
    SILENT, // Don't print anything
    ALL     // Print all messages
  };

  // Configure exception message visibility
  PrintMode printLevel = PrintMode::SILENT;

  // Device type that generated the trace
  Device device;

  virtual ~ParserConfig() = default;
};

// Define exception severity levels
enum class ExceptionSeverity {
  WARNING, // Continue parsing
  ERROR    // Stop parsing
};

struct ParserException : public std::runtime_error {
  ExceptionSeverity severity;

  ParserException(const std::string &msg, ExceptionSeverity sev);
};

class ParserBase {
public:
  explicit ParserBase(ByteSpan &buffer, const ParserConfig &config);

  virtual ~ParserBase() = default;

  virtual void parse() = 0;

  virtual const ParserConfig &getConfig() const;

protected:
  void reportException(const ParserException &e, size_t pos);

  const ParserConfig &config;
  ByteSpan &buffer;
};

} // namespace proton

#endif // PROTON_COMMON_PARSER_H_
