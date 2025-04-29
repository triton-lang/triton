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

  int numBlock = getConfig().numBlocks;
  const int scratchMemSize = getConfig().scratchMemSize;
  uint32_t pos = buffer.position();
  for (int i = 0; i < numBlock; i++) {
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

  for (auto uid : getConfig().uidVec) {
    auto count = countVec[uid];
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
    : ParserException(msg, ExceptionSeverity::WARNING) {}
