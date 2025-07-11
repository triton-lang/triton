#include "TraceDataIO/Parser.h"
#include <iomanip>

using namespace proton;

ParserException::ParserException(const std::string &msg, ExceptionSeverity sev)
    : std::runtime_error(msg), severity(sev) {}

ParserBase::ParserBase(ByteSpan &buffer, const ParserConfig &config)
    : buffer(buffer), config(config) {}

void ParserBase::reportException(const ParserException &e, size_t pos) {

  if (e.severity == ExceptionSeverity::ERROR ||
      config.printLevel == ParserConfig::PrintMode::ALL) {
    std::cerr << "ParserException [offset=" << pos << "]: " << e.what()
              << std::endl;
    std::cerr << "Buffer: " << std::endl;
    buffer.seek(0);
    for (size_t i = 0; i < buffer.size() / 4; i++) {
      std::cerr << std::setw(8) << std::setfill('0') << std::hex
                << buffer.readInt32() << " ";
      if (i % 4 == 3) {
        std::cerr << std::endl;
      }
    }
    std::cerr << std::dec << std::endl;
  }

  if (e.severity == ExceptionSeverity::WARNING)
    return;

  throw e;
}

const ParserConfig &ParserBase::getConfig() const { return config; }
