#include "TraceDataIO/Parser.h"

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
  }

  if (e.severity == ExceptionSeverity::WARNING)
    return;

  throw e;
}

const ParserConfig &ParserBase::getConfig() const { return config; }
