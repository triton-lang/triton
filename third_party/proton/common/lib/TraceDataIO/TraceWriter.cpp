#include "TraceDataIO/TraceWriter.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace proton;
using json = nlohmann::json;

namespace {
using PathKey = std::vector<std::string>;
using BlockTraceVec =
    std::vector<const CircularLayoutParserResult::BlockTrace *>;

struct TimeSpan {
  double start = std::numeric_limits<double>::max();
  double end = std::numeric_limits<double>::lowest();

  bool valid() const { return start <= end; }

  void include(double ts, double dur) {
    start = std::min(start, ts);
    end = std::max(end, ts + dur);
  }
};

struct CallPathBar {
  std::string name;
  json args = json::object();
  TimeSpan span;
};

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

void populateTraceInfo(std::shared_ptr<CircularLayoutParserResult> result,
                       std::map<int, uint64_t> &blockToMinCycle,
                       std::map<int, BlockTraceVec> &procToBlockTraces) {
  for (auto &bt : result->blockTraces) {
    // Find the minimum cycle for each block
    uint64_t minCycle = std::numeric_limits<uint64_t>::max();
    for (auto &trace : bt.traces)
      for (auto &event : trace.profileEvents)
        if (event.first->cycle < minCycle)
          minCycle = event.first->cycle;
    blockToMinCycle[bt.blockId] = minCycle;

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

json buildCallStackJson(const std::vector<std::string> &callStack) {
  json result = json::array();
  for (const auto &frame : callStack) {
    result.push_back(frame);
  }
  return result;
}

TimeSpan getKernelSpan(const KernelTrace &kernelTrace,
                       const uint64_t minInitTime) {
  auto result = kernelTrace.first;
  std::map<int, uint64_t> blockToMinCycle;
  std::map<int, BlockTraceVec> procToBlockTraces;
  populateTraceInfo(result, blockToMinCycle, procToBlockTraces);

  constexpr double freq = 1000.0;
  TimeSpan span;
  for (auto &[procId, blockVec] : procToBlockTraces) {
    (void)procId;
    for (auto *bt : blockVec) {
      const int ctaId = bt->blockId;
      const int64_t cycleAdjust =
          static_cast<int64_t>(bt->initTime - minInitTime) -
          static_cast<int64_t>(blockToMinCycle[ctaId]);
      for (auto &trace : bt->traces) {
        for (auto &event : trace.profileEvents) {
          const int64_t ts =
              static_cast<int64_t>(event.first->cycle) + cycleAdjust;
          const int64_t dur =
              static_cast<int64_t>(event.second->cycle) - event.first->cycle;
          span.include(static_cast<double>(ts) / freq,
                       static_cast<double>(dur) / freq);
        }
      }
    }
  }
  return span;
}

std::map<PathKey, CallPathBar>
collectCallPathBars(const std::vector<KernelTrace> &streamTrace,
                    const uint64_t minInitTime) {
  std::map<PathKey, CallPathBar> bars;
  for (const auto &kernelTrace : streamTrace) {
    const auto span = getKernelSpan(kernelTrace, minInitTime);
    if (!span.valid()) {
      continue;
    }

    PathKey path;
    for (const auto &frame : kernelTrace.second->callPathFrames) {
      path.push_back(frame.name);
      auto &bar = bars[path];
      bar.name = frame.name;
      if (!frame.args.empty()) {
        bar.args = frame.args;
      }
      bar.span.include(span.start, span.end - span.start);
    }
  }
  return bars;
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
  const auto callPathBars = collectCallPathBars(streamTrace, minInitTime);

  for (const auto &[path, bar] : callPathBars) {
    json element;
    element["cname"] = kChromeColor[(path.size() - 1) % kChromeColor.size()];
    element["name"] = bar.name;
    element["cat"] = "call_path";
    element["ph"] = "X";
    element["pid"] = "Kernel Call Path";
    element["tid"] = "path " + std::to_string(path.size() - 1);
    element["ts"] = bar.span.start;
    element["dur"] = bar.span.end - bar.span.start;
    element["args"] = bar.args;
    element["args"]["call_stack"] = buildCallStackJson(path);
    object["traceEvents"].push_back(element);
  }

  for (const auto &kernelTrace : streamTrace) {
    writeKernel(object, kernelTrace, minInitTime);
  }
  outfile << object.dump() << "\n";
}

void StreamChromeTraceWriter::writeKernel(json &object,
                                          const KernelTrace &kernelTrace,
                                          const uint64_t minInitTime) {
  auto result = kernelTrace.first;
  auto metadata = kernelTrace.second;

  const auto callStack = buildCallStackJson(metadata->callStack);

  int curColorIndex = 0;
  // scope id -> color index in chrome color
  std::map<int, int> scopeColor;
  // block id -> min cycle observed
  std::map<int, uint64_t> blockToMinCycle;
  // proc id -> block traces
  std::map<int, BlockTraceVec> procToBlockTraces;

  populateTraceInfo(result, blockToMinCycle, procToBlockTraces);

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

          // Unit: MHz, we assume freq is 1000MHz (1GHz)
          double freq = 1000.0;

          // Global time is in `ns` unit. With 1GHz assumption, we
          // could subtract with blockToMInCycle: (ns - ns) / 1GHz - cycle
          int64_t cycleAdjust =
              static_cast<int64_t>(bt->initTime - minInitTime) -
              static_cast<int64_t>(blockToMinCycle[ctaId]);
          int64_t ts = static_cast<int64_t>(event.first->cycle) + cycleAdjust;
          int64_t dur =
              static_cast<int64_t>(event.second->cycle) - event.first->cycle;

          json element;
          element["cname"] = color;
          element["name"] = name;
          element["cat"] = category;
          element["ph"] = "X";
          element["pid"] = pid;
          element["tid"] = tid;
          element["ts"] = static_cast<double>(ts) / freq;
          element["dur"] = static_cast<double>(dur) / freq;
          json args;
          args["Init Time (ns)"] = bt->initTime;
          args["Post Final Time (ns)"] = bt->postFinalTime;
          args["Finalization Time (ns)"] = bt->postFinalTime - bt->preFinalTime;
          args["Frequency (MHz)"] = freq;
          element["args"] = args;
          element["args"]["call_stack"] = callStack;

          object["traceEvents"].push_back(element);

          eventIdx++;
        }
      }
    }
  }
}
