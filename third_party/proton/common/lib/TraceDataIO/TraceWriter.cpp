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
  uint64_t nextFlowId = 1;

  for (const auto &kernelTrace : streamTrace) {
    writeKernel(object, kernelTrace, minInitTime, nextFlowId);
  }
  outfile << object.dump() << "\n";
}

namespace {
using BlockTrace = CircularLayoutParserResult::BlockTrace;
using BlockTraceVec = std::vector<const BlockTrace *>;
using EventLineIds = std::map<const CycleEntry *, int>;
using ScopeColors = std::map<int, int>;

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
    for (auto &trace : bt.traces)
      for (auto &record : trace.asyncEvents)
        if (record->cycle < minCycle)
          minCycle = record->cycle;
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

EventLineIds assignBlockLineIds(const BlockTrace &blockTrace) {
  std::map<uint32_t, std::vector<CircularLayoutParserResult::ProfileEvent>>
      eventsByWarp;
  // Assign synchronous scopes and same-warp async events from one combined
  // interval set so overlapping activity gets separate lines within the same
  // warp group.
  for (const auto &trace : blockTrace.traces)
    eventsByWarp[trace.uid].insert(eventsByWarp[trace.uid].end(),
                                   trace.profileEvents.begin(),
                                   trace.profileEvents.end());
  for (const auto &link : blockTrace.asyncLinks) {
    if (link.first.uid == link.second.uid)
      eventsByWarp[link.first.uid].emplace_back(link.first.entry,
                                                link.second.entry);
  }

  EventLineIds eventLineIds;
  for (const auto &warpEvents : eventsByWarp) {
    const auto &events = warpEvents.second;
    auto lineIds = assignLineIds(events);
    for (size_t i = 0; i < events.size(); ++i)
      eventLineIds[events[i].first.get()] = lineIds[i];
  }
  return eventLineIds;
}

void writeProfileEvents(json &object, const BlockTrace &blockTrace,
                        const KernelMetadata &metadata, const json &callStack,
                        const std::string &pid, int64_t cycleAdjust,
                        const EventLineIds &eventLineIds,
                        ScopeColors &scopeColor, int &curColorIndex,
                        const std::vector<std::string> &colors) {
  constexpr double freq = 1000.0;
  for (const auto &trace : blockTrace.traces) {
    int warpId = trace.uid;
    for (const auto &event : trace.profileEvents) {
      int lineId = eventLineIds.at(event.first.get());
      int scopeId = event.first->scopeId;
      if (!scopeColor.count(scopeId)) {
        scopeColor[scopeId] = curColorIndex;
        curColorIndex = (curColorIndex + 1) % colors.size();
      }
      const std::string &color = colors[scopeColor[scopeId]];
      std::string name = !metadata.scopeName.count(scopeId)
                             ? "scope_" + std::to_string(scopeId)
                             : metadata.scopeName.at(scopeId);
      std::string tid = "warp " + std::to_string(warpId) + " (line " +
                        std::to_string(lineId) + ")";
      int64_t ts = static_cast<int64_t>(event.first->cycle) + cycleAdjust;
      int64_t dur =
          static_cast<int64_t>(event.second->cycle) - event.first->cycle;

      json element;
      element["cname"] = color;
      element["name"] = name;
      element["cat"] = metadata.kernelName;
      element["ph"] = "X";
      element["pid"] = pid;
      element["tid"] = tid;
      element["ts"] = static_cast<double>(ts) / freq;
      element["dur"] = static_cast<double>(dur) / freq;
      json args;
      args["Init Time (ns)"] = blockTrace.initTime;
      args["Post Final Time (ns)"] = blockTrace.postFinalTime;
      args["Finalization Time (ns)"] =
          blockTrace.postFinalTime - blockTrace.preFinalTime;
      args["Frequency (MHz)"] = freq;
      element["args"] = args;
      element["args"]["call_stack"] = callStack;

      object["traceEvents"].push_back(std::move(element));
    }
  }
}

void writeAsyncEvents(json &object, const BlockTrace &blockTrace,
                      const KernelMetadata &metadata, const json &callStack,
                      const std::string &pid, int64_t cycleAdjust,
                      const EventLineIds &eventLineIds, ScopeColors &scopeColor,
                      int &curColorIndex,
                      const std::vector<std::string> &colors,
                      uint64_t &nextFlowId) {
  constexpr double freq = 1000.0;
  for (const auto &link : blockTrace.asyncLinks) {
    int scopeId = link.first.entry->scopeId;
    if (!scopeColor.count(scopeId)) {
      scopeColor[scopeId] = curColorIndex;
      curColorIndex = (curColorIndex + 1) % colors.size();
    }
    const std::string &color = colors[scopeColor[scopeId]];
    std::string name = !metadata.scopeName.count(scopeId)
                           ? "event_" + std::to_string(scopeId)
                           : metadata.scopeName.at(scopeId);

    // Both endpoints belong to one warp, so the event has a natural lane
    // and can be displayed as a duration slice.
    if (link.first.uid == link.second.uid) {
      int64_t ts = static_cast<int64_t>(link.first.entry->cycle) + cycleAdjust;
      int64_t dur = static_cast<int64_t>(link.second.entry->cycle) -
                    link.first.entry->cycle;
      json element;
      element["cname"] = color;
      element["name"] = name;
      element["cat"] = "async";
      element["ph"] = "X";
      element["pid"] = pid;
      element["tid"] = "warp " + std::to_string(link.first.uid) + " (line " +
                       std::to_string(eventLineIds.at(link.first.entry.get())) +
                       ")";
      element["ts"] = static_cast<double>(ts) / freq;
      element["dur"] = static_cast<double>(dur) / freq;
      element["args"]["call_stack"] = callStack;
      object["traceEvents"].push_back(std::move(element));
      continue;
    }

    // Cross-warp events have no single owning lane. Emit an instantaneous
    // endpoint on each warp and connect them with a dynamically bound flow.
    const uint64_t flowId = nextFlowId++;
    auto writeEndpoint = [&](const auto &endpoint, bool isStart) {
      int64_t ts = static_cast<int64_t>(endpoint.entry->cycle) + cycleAdjust;
      json element;
      element["cname"] = color;
      element["name"] = name;
      element["cat"] = "flow";
      element["ph"] = "X";
      element["dur"] = 0;
      element["bind_id"] = flowId;
      element[isStart ? "flow_out" : "flow_in"] = true;
      element["pid"] = pid;
      element["tid"] = "warp " + std::to_string(endpoint.uid) + " (line 0)";
      element["ts"] = static_cast<double>(ts) / freq;
      element["args"]["call_stack"] = callStack;
      object["traceEvents"].push_back(std::move(element));
    };

    writeEndpoint(link.first, true);
    writeEndpoint(link.second, false);
  }
}

} // namespace

void StreamChromeTraceWriter::writeKernel(json &object,
                                          const KernelTrace &kernelTrace,
                                          const uint64_t minInitTime,
                                          uint64_t &nextFlowId) {
  auto result = kernelTrace.first;
  auto metadata = kernelTrace.second;

  json callStack = json::array();
  for (auto const &frame : metadata->callStack) {
    callStack.push_back(frame);
  }

  int curColorIndex = 0;
  // scope id -> color index in chrome color
  std::map<int, int> scopeColor;
  // block id -> min cycle observed
  std::map<int, uint64_t> blockToMinCycle;
  // proc id -> block traces
  std::map<int, BlockTraceVec> procToBlockTraces;

  populateTraceInfo(result, blockToMinCycle, procToBlockTraces);

  for (auto &[procId, blockVec] : procToBlockTraces) {
    for (auto *bt : blockVec) {
      int ctaId = bt->blockId;
      auto eventLineIds = assignBlockLineIds(*bt);
      std::string pid = metadata->kernelName + " Core" +
                        std::to_string(procId) + " CTA" + std::to_string(ctaId);
      int64_t cycleAdjust = static_cast<int64_t>(bt->initTime - minInitTime) -
                            static_cast<int64_t>(blockToMinCycle.at(ctaId));
      writeProfileEvents(object, *bt, *metadata, callStack, pid, cycleAdjust,
                         eventLineIds, scopeColor, curColorIndex, kChromeColor);
      writeAsyncEvents(object, *bt, *metadata, callStack, pid, cycleAdjust,
                       eventLineIds, scopeColor, curColorIndex, kChromeColor,
                       nextFlowId);
    }
  }
}
