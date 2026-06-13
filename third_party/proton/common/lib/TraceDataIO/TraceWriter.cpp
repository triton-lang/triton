#include "TraceDataIO/TraceWriter.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace proton;
using json = nlohmann::json;

namespace {

uint64_t getMinInitTime(const std::vector<KernelTrace> &streamTrace) {
  uint64_t minInitTime = std::numeric_limits<uint64_t>::max();
  for (const auto &kernelTrace : streamTrace)
    for (const auto &bt : kernelTrace.first->blockTraces) {
      if (bt.initTime < minInitTime) {
        minInitTime = bt.initTime;
      }
    }
  return minInitTime;
}

} // namespace

StreamTraceWriter::StreamTraceWriter(
    const std::vector<KernelTrace> &streamTrace, const std::string &path)
    : streamTrace(streamTrace), path(path) {}

void StreamTraceWriter::dump() {
  std::ofstream outfile;

  if (path.empty()) {
    std::cerr << "Trace file path can't be empty!";
    return;
  }

  outfile.open(path);
  if (!outfile.is_open()) {
    std::cerr << "Failed to open trace file: " << path << std::endl;
    return;
  }

  write(outfile);

  outfile.close();
}

StreamChromeTraceWriter::StreamChromeTraceWriter(
    const std::vector<KernelTrace> &streamTrace, const std::string &path)
    : StreamTraceWriter(streamTrace, path) {}

void StreamChromeTraceWriter::write(std::ostream &outfile) {
  if (streamTrace.empty()) {
    std::cerr << "Failed to write the trace file: empty trace!" << std::endl;
    return;
  }

  json object = {{"displayTimeUnit", "ns"}, {"traceEvents", json::array()}};

  const auto minInitTime = getMinInitTime(streamTrace);

  for (const auto &kernelTrace : streamTrace) {
    writeKernel(object, kernelTrace, minInitTime);
  }
  outfile << object.dump() << "\n";
}

namespace {
using BlockTraceVec =
    std::vector<const CircularLayoutParserResult::BlockTrace *>;

void populateTraceInfo(std::shared_ptr<CircularLayoutParserResult> result,
                       std::map<int, BlockTraceVec> &procToBlockTraces) {
  for (auto &bt : result->blockTraces) {
    // Group block traces by proc id
    int procId = bt.procId;
    if (!procToBlockTraces.count(procId)) {
      procToBlockTraces[procId] = {};
    }
    procToBlockTraces[procId].push_back(&bt);
  }
}

std::vector<int> assignLineIds(
    const std::vector<CircularLayoutParserResult::ProfileEvent> &trace) {

  std::vector<int> result(trace.size());

  if (trace.empty()) {
    return result;
  }

  // Create indexed events and sort by start time
  std::vector<std::pair<size_t, CircularLayoutParserResult::ProfileEvent>>
      indexedEvents;
  indexedEvents.reserve(trace.size());

  for (size_t i = 0; i < trace.size(); ++i) {
    indexedEvents.push_back({i, trace[i]});
  }

  std::sort(indexedEvents.begin(), indexedEvents.end(),
            [](const auto &a, const auto &b) {
              return a.second.first->cycle < b.second.first->cycle;
            });

  // For each line, store all the intervals
  std::vector<std::vector<std::pair<uint64_t, uint64_t>>> lines;

  for (const auto &[originalIdx, event] : indexedEvents) {
    uint64_t startTime = event.first->cycle;
    uint64_t endTime = event.second->cycle;

    // Find the first line where this event can be placed
    int lineIdx = 0;
    bool foundLine = false;

    for (; lineIdx < lines.size(); ++lineIdx) {
      const auto &lineIntervals = lines[lineIdx];
      bool canPlace = true;

      // Check for overlap with any interval on this line
      for (const auto &[intervalStart, intervalEnd] : lineIntervals) {
        // Check if there's any overlap
        if (startTime < intervalEnd && endTime > intervalStart) {
          canPlace = false;
          break;
        }
      }

      if (canPlace) {
        foundLine = true;
        break;
      }
    }

    // If no suitable line found, create a new one
    if (!foundLine) {
      lineIdx = lines.size();
      lines.push_back({});
    }

    // Add the event to the line
    lines[lineIdx].push_back({startTime, endTime});
    result[originalIdx] = lineIdx;
  }

  return result;
}

} // namespace

void StreamChromeTraceWriter::writeKernel(json &object,
                                          const KernelTrace &kernelTrace,
                                          const uint64_t minInitTime) {
  auto result = kernelTrace.first;
  auto metadata = kernelTrace.second;

  json callStack = json::array();
  for (auto const &frame : metadata->callStack) {
    callStack.push_back(frame);
  }

  int curColorIndex = 0;
  // scope id -> color index in chrome color
  std::map<int, int> scopeColor;
  // proc id -> block traces
  std::map<int, BlockTraceVec> procToBlockTraces;

  populateTraceInfo(result, procToBlockTraces);

  std::string name;
  std::string pid;
  std::string category;
  std::string tid;
  for (auto &[procId, blockVec] : procToBlockTraces) {
    for (auto *bt : blockVec) {
      int ctaId = bt->blockId;
      for (auto &trace : bt->traces) {
        int warpId = trace.uid;
        auto lineInfo = assignLineIds(trace.profileEvents);
        int eventIdx = 0;
        for (auto &event : trace.profileEvents) {
          int lineId = lineInfo[eventIdx];
          int scopeId = event.first->scopeId;
          if (!scopeColor.count(scopeId)) {
            scopeColor[scopeId] = curColorIndex;
            curColorIndex = (curColorIndex + 1) % kChromeColor.size();
          }
          const std::string &color = kChromeColor[scopeColor[scopeId]];
          pid = metadata->kernelName + " Core" + std::to_string(procId) +
                " CTA" + std::to_string(ctaId);
          tid = "warp " + std::to_string(warpId) + " (line " +
                std::to_string(lineId) + ")";
          category = metadata->kernelName;
          if (!metadata->scopeName.count(scopeId))
            name = "scope_" + std::to_string(scopeId);
          else
            name = metadata->scopeName.at(scopeId);

          // All cycle values are in nanoseconds (from globalTime).
          const double USEC_PER_CYCLE = 1000.0;

          int64_t ts = static_cast<int64_t>(event.first->cycle) +
                       static_cast<int64_t>(minInitTime);
          int64_t dur = static_cast<int64_t>(event.second->cycle) -
                        static_cast<int64_t>(event.first->cycle);

          json element;
          element["cname"] = color;
          element["name"] = name;
          element["cat"] = category;
          element["ph"] = "X";
          element["pid"] = pid;
          element["tid"] = tid;
          element["ts"] = static_cast<double>(ts) / USEC_PER_CYCLE;
          element["dur"] = static_cast<double>(dur) / USEC_PER_CYCLE;
          json args;
          args["Init Time (ns)"] = bt->initTime;
          args["Post Final Time (ns)"] = bt->postFinalTime;
          args["Finalization Time (ns)"] = bt->postFinalTime - bt->preFinalTime;
          element["args"] = args;
          element["args"]["call_stack"] = callStack;

          object["traceEvents"].push_back(element);

          eventIdx++;
        }
      }
    }
  }
}
