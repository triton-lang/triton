#include "TraceDataIO/CircularLayoutParser.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_map>

using namespace proton;

CircularLayoutParser::CircularLayoutParser(
    ByteSpan &buffer, const CircularLayoutParserConfig &config)
    : ParserBase(buffer, config), decoder(buffer) {
  result = std::make_shared<CircularLayoutParserResult>();
}

std::shared_ptr<CircularLayoutParserResult> CircularLayoutParser::getResult() {
  return result;
}

void CircularLayoutParser::parse() {
  auto &uidVec = getConfig().uidVec;
  assert(uidVec.size());
  assert(std::is_sorted(uidVec.begin(), uidVec.end()));

  int numBlocks = getConfig().numBlocks;
  const int scratchMemSize = getConfig().scratchMemSize;
  uint32_t pos = buffer.position();
  for (int i = 0; i < numBlocks; i++) {
    buffer.seek(pos);
    parseBlock();
    pos += scratchMemSize;
  }
}

const CircularLayoutParserConfig &CircularLayoutParser::getConfig() const {
  return static_cast<const CircularLayoutParserConfig &>(config);
}

void CircularLayoutParser::parseMetadata() {
  uint32_t preamble = decoder.decode<I32Entry>()->value;
  if (preamble != kPreamble)
    throw PreambleException("Invalid preamble");
  auto &bt = result->blockTraces.emplace_back();
  bt.blockId = decoder.decode<I32Entry>()->value;
  bt.procId = decoder.decode<I32Entry>()->value;
  bt.bufSize = decoder.decode<I32Entry>()->value;

  std::vector<uint32_t> countVec;
  for (int i = 0; i < getConfig().totalUnits; i++) {
    countVec.push_back(decoder.decode<I32Entry>()->value);
  }

  // Each event is 8 bytes
  int maxCountPerUnit = bt.bufSize / getConfig().uidVec.size() / 8;

  for (auto uid : getConfig().uidVec) {
    auto count = countVec[uid];

    if (count > maxCountPerUnit) {
      std::cerr << "Warning (cta" << bt.blockId << ", warp" << uid
                << "): first " << count - maxCountPerUnit
                << " events are dropped due to insufficient buffer size ("
                << maxCountPerUnit << "/" << count << ")" << std::endl;
    }

    auto &trace = bt.traces.emplace_back();
    trace.uid = uid;
    trace.count = count;
  }
}

void CircularLayoutParser::parseProfileEvents() {
  auto &bt = result->blockTraces.back();
  const int bufferSize = bt.bufSize;
  const int numSegments = getConfig().uidVec.size();
  const int segmentByteSize = bufferSize / numSegments;
  auto position = buffer.position();
  for (int i = 0; i < numSegments; i++) {
    buffer.seek(position);
    auto &trace = bt.traces[i];
    parseSegment(segmentByteSize, trace);
    position += segmentByteSize;
  }
}

void CircularLayoutParser::parseSegment(
    int segmentByteSize, CircularLayoutParserResult::Trace &trace) {

  auto state = ParseState::INIT;
  int idealSize = trace.count * kWordSize;
  int byteSize = std::min(idealSize, segmentByteSize);
  const int maxNumEntries = byteSize / (kWordSize * kWordsPerEntry);

  std::unordered_map<int, CircularLayoutParserResult::ProfileEvent> activeEvent;
  std::unordered_map<int, ParseState> scopeState;

  for (int i = 0; i < maxNumEntries; i++) {
    try {
      auto entry = decoder.decode<CycleEntry>();
      if (!activeEvent.count(entry->scopeId)) {
        activeEvent[entry->scopeId] =
            CircularLayoutParserResult::ProfileEvent();
      }
      auto &activeProfileEvent = activeEvent[entry->scopeId];

      auto prevState = ParseState::INIT;
      if (scopeState.count(entry->scopeId))
        prevState = scopeState[entry->scopeId];

      if (entry->isStart) {
        if (prevState == ParseState::INIT || prevState == ParseState::END) {
          activeProfileEvent.first = entry;
          scopeState[entry->scopeId] = ParseState::START;
        } else {
          throw ScopeMisMatchException("Scope mismatch: start after start");
        }
      } else {
        if (prevState == ParseState::START) {
          activeProfileEvent.second = entry;
          scopeState[entry->scopeId] = ParseState::END;

          if (activeProfileEvent.first->cycle >
              activeProfileEvent.second->cycle) {
            throw ClockOverflowException("Clock overflow");
          }
          trace.profileEvents.push_back(activeProfileEvent);
        } else {
          throw ScopeMisMatchException("Scope mismatch: end after end");
        }
      }
    } catch (const ScopeMisMatchException &e) {
      reportException(e, buffer.position());
    } catch (const ClockOverflowException &e) {
      reportException(e, buffer.position());
    }
  }
}

void CircularLayoutParser::parseBlock() {
  try {
    parseMetadata();
    parseProfileEvents();
  } catch (const PreambleException &e) {
    reportException(e, buffer.position());
  }
}

PreambleException::PreambleException(const std::string &msg)
    : ParserException(msg, ExceptionSeverity::ERROR) {}

ScopeMisMatchException::ScopeMisMatchException(const std::string &msg)
    : ParserException(msg, ExceptionSeverity::WARNING) {}

ClockOverflowException::ClockOverflowException(const std::string &msg)
    : ParserException(msg, ExceptionSeverity::ERROR) {}

namespace {
Device decodeDevice(const uint32_t dev) {
  Device device;
  switch (dev) {
  case 1:
    device.type = DeviceType::CUDA;
    device.arch = "";
    break;
  case 2:
    device.type = DeviceType::HIP;
    device.arch = "";
    break;
  default:
    break;
  }
  return device;
}

void shift(CircularLayoutParserResult::Trace &trace, const uint64_t cost,
           const uint64_t timeBase) {
  for (auto &event : trace.profileEvents) {
    if (event.first->cycle >= timeBase)
      event.first->cycle -= cost;
    if (event.second->cycle >= timeBase)
      event.second->cycle -= cost;
  }
}
} // namespace

std::shared_ptr<CircularLayoutParserResult>
proton::readCircularLayoutTrace(ByteSpan &buffer, bool applyTimeShift) {
  CircularLayoutParserConfig config;
  auto decoder = EntryDecoder(buffer);
  uint32_t version = decoder.decode<I32Entry>()->value;
  assert(version == 1 && "Version mismatch");
  buffer.skip(8);
  uint32_t payloadOffset = decoder.decode<I32Entry>()->value;
  uint32_t payloadSize = decoder.decode<I32Entry>()->value;
  uint32_t device = decoder.decode<I32Entry>()->value;
  config.device = decodeDevice(device);
  config.numBlocks = decoder.decode<I32Entry>()->value;
  config.totalUnits = decoder.decode<I32Entry>()->value;
  config.scratchMemSize = decoder.decode<I32Entry>()->value;
  uint32_t uidNum = decoder.decode<I32Entry>()->value;

  config.uidVec.clear();
  for (int i = 0; i < uidNum; i++) {
    uint32_t uid = decoder.decode<I32Entry>()->value;
    config.uidVec.push_back(uid);
  }

  buffer.seek(payloadOffset);
  auto parser = std::make_unique<CircularLayoutParser>(buffer, config);
  parser->parse();
  auto result = parser->getResult();

  // Shift the clocks to reduce the constant profiling overhead
  if (applyTimeShift) {
    const uint64_t cost = getTimeShiftCost(config);
    timeShift(cost, result);
  }

  return result;
}

void proton::timeShift(const uint64_t cost,
                       std::shared_ptr<CircularLayoutParserResult> result) {
  for (auto &bt : result->blockTraces) {
    for (auto &trace : bt.traces) {
      for (auto &event : trace.profileEvents) {
        const uint64_t startTimeBase = event.first->cycle;
        shift(trace, cost, startTimeBase);

        const uint64_t endTimeBase = event.second->cycle;
        shift(trace, cost, endTimeBase);

        // Adjust the cycle for tiny events below the profiling precision
        if (event.second->cycle < event.first->cycle) {
          event.second->cycle = event.first->cycle + cost / 2;
        }
      }
    }
  }
}

uint64_t proton::getTimeShiftCost(const CircularLayoutParserConfig &config) {
  if (config.device.type == DeviceType::CUDA)
    return 7;
  else if (config.device.type == DeviceType::HIP)
    return 36;

  return 0;
}
