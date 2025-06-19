#include "TraceDataIO/TraceWriter.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace proton;
using json = nlohmann::json;

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

  int totalKernelNum = streamTrace.size();
  for (int i = 0; i < totalKernelNum; i++) {
    writeKernel(object, streamTrace[i], kKernelTimeGap * i);
  }
  outfile << object.dump() << "\n";
}

namespace {
using BlockTraceVec =
    std::vector<const CircularLayoutParserResult::BlockTrace *>;

void populateTraceInfo(std::shared_ptr<CircularLayoutParserResult> result,
                       uint64_t kernelTimeStart,
                       std::map<int, int64_t> &cycleAdjust,
                       std::map<int, BlockTraceVec> &procToBlockTraces) {
  uint64_t minStartTime;
  for (auto &bt : result->blockTraces) {
    minStartTime = std::numeric_limits<uint64_t>::max();
    for (auto &trace : bt.traces)
      for (auto &event : trace.profileEvents)
        if (event.first->cycle < minStartTime)
          minStartTime = event.first->cycle;

    cycleAdjust[bt.blockId] = static_cast<int64_t>(kernelTimeStart) -
                              static_cast<int64_t>(minStartTime);
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
                                          uint64_t kernelTimeStart) {
  auto result = kernelTrace.first;
  auto metadata = kernelTrace.second;

  int curColorIndex = 0;
  // scope id -> color index in chrome color
  std::map<int, int> scopeColor;
  // block id -> cycle adjust
  std::map<int, int64_t> cycleAdjust;
  // proc id -> block traces
  std::map<int, BlockTraceVec> procToBlockTraces;

  populateTraceInfo(result, kernelTimeStart, cycleAdjust, procToBlockTraces);

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
                " CTA" + std::to_string(ctaId) +
                " [measure in clock cycle (assume 1GHz)]";
          tid = "warp " + std::to_string(warpId) + " (line " +
                std::to_string(lineId) + ")";
          category = metadata->kernelName;
          if (!metadata->scopeName.count(scopeId))
            name = "scope_" + std::to_string(scopeId);
          else
            name = metadata->scopeName.at(scopeId);

          int64_t ts =
              static_cast<int64_t>(event.first->cycle) + cycleAdjust[ctaId];
          int64_t dur =
              static_cast<int64_t>(event.second->cycle) - event.first->cycle;

          json element;
          element["cname"] = color;
          element["name"] = name;
          element["cat"] = category;
          element["ph"] = "X";
          element["pid"] = pid;
          element["tid"] = tid;
          element["ts"] = static_cast<float>(ts) / 1000.0;
          element["dur"] = static_cast<float>(dur) / 1000.0;
          json args;
          args["Unit"] = "GPU cycle";
          args["Kernel Gap"] = std::to_string(kKernelTimeGap) + "cycle(ns)";
          element["args"] = args;
          object["traceEvents"].push_back(element);

          eventIdx++;
        }
      }
    }
  }
}
