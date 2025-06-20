#ifndef PROTON_COMMON_CIRCULAR_LAYOUT_PARSER_H_
#define PROTON_COMMON_CIRCULAR_LAYOUT_PARSER_H_

#include "Parser.h"
#include <cstdint>

namespace proton {

constexpr uint32_t kPreamble = 0xdeadbeef;
constexpr uint32_t kHeaderSize = 16;
constexpr uint32_t kWordSize = 4;
constexpr uint32_t kWordsPerEntry = 2;

enum class ParseState { START, END, INIT };

struct CircularLayoutParserConfig : public ParserConfig {
  // The total number of unit (e.g., num of warps) in CTA
  size_t totalUnits = 0;
  // Scratch memory size in bytes per CTA (scratchMemSize = metadata_size +
  // bufSize)
  size_t scratchMemSize = 0;
  // The number of blocks in the grid
  size_t numBlocks = 0;
  // A vector of trace's uids
  std::vector<uint32_t> uidVec = {};
};

struct CircularLayoutParserResult {
  // start cycle entry and end cycle entry
  using ProfileEvent =
      std::pair<std::shared_ptr<CycleEntry>, std::shared_ptr<CycleEntry>>;

  struct Trace {
    uint32_t uid = 0;

    // Total count of words (i32) if we don't drop events.
    uint32_t count = 0;

    std::vector<ProfileEvent> profileEvents;
  };

  struct BlockTrace {
    uint32_t blockId = 0;
    uint32_t procId = 0;
    uint32_t bufSize = 0;
    std::vector<Trace> traces;
  };

  std::vector<BlockTrace> blockTraces;
};

class CircularLayoutParser : public ParserBase {
public:
  explicit CircularLayoutParser(ByteSpan &buffer,
                                const CircularLayoutParserConfig &config);

  void parse() final;

  const CircularLayoutParserConfig &getConfig() const override;

  std::shared_ptr<CircularLayoutParserResult> getResult();

private:
  void parseMetadata();
  void parseProfileEvents();
  void parseSegment(int byteSize, CircularLayoutParserResult::Trace &trace);
  void parseBlock();

  std::shared_ptr<CircularLayoutParserResult> result = nullptr;
  EntryDecoder decoder;
};

struct PreambleException : public ParserException {
  PreambleException(const std::string &msg);
};

struct ScopeMisMatchException : public ParserException {
  ScopeMisMatchException(const std::string &msg);
};

struct ClockOverflowException : public ParserException {
  ClockOverflowException(const std::string &msg);
};

std::shared_ptr<CircularLayoutParserResult>
readCircularLayoutTrace(ByteSpan &buffer, bool applyTimeShift = false);

uint64_t getTimeShiftCost(const CircularLayoutParserConfig &config);

void timeShift(const uint64_t cost,
               std::shared_ptr<CircularLayoutParserResult> result);

} // namespace proton

#endif // PROTON_COMMON_CIRCULAR_LAYOUT_PARSER_H_
