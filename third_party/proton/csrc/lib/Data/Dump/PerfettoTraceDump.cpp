#include "Dump/TraceDataDump.h"
#include "Utility/ProtoWriter.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

namespace proton::trace_data_dump {

namespace {

constexpr uint64_t kPerfettoProcessTrackUuid = 1;
constexpr uint64_t kPerfettoLaneTrackUuidBase = 1000;
constexpr uint64_t kPerfettoFlowIdBase = 1ULL << 32;
constexpr uint32_t kPerfettoTracePacketSequenceId = 1;
constexpr uint32_t kPerfettoSeqIncrementalStateCleared = 1;
constexpr uint32_t kPerfettoSeqNeedsIncrementalState = 2;
constexpr int32_t kPerfettoCpuTrackOrderBase = 0;
constexpr int32_t kPerfettoGraphTrackOrderBase = 100000;
constexpr int32_t kPerfettoGpuTrackOrderBase = 200000;
constexpr uint32_t kPerfettoChildTracksOrderingExplicit = 3;

enum class PerfettoEventCategory { Scope, Metric, Kernel, Count };

struct PerfettoAnnotation {
  enum class Kind { String, UInt64, Int64, Double, Bool };

  uint64_t nameIid{};
  uint64_t stringValueIid{};
  uint64_t uintValue{};
  int64_t intValue{};
  double doubleValue{};
  bool boolValue{};
  Kind kind{Kind::String};
};

struct PerfettoTrack {
  std::string name;
  int32_t siblingOrderRank{};
};

class PerfettoInternedStringTable {
public:
  std::pair<uint64_t, bool> intern(const std::string &name) {
    if (auto it = nameToIid.find(name); it != nameToIid.end()) {
      return {it->second, false};
    }

    const auto iid = nextIid++;
    nameToIid.emplace(name, iid);
    iidToName.emplace(iid, name);
    return {iid, true};
  }

  void internWithIid(const std::string &name, uint64_t iid) {
    if (auto it = nameToIid.find(name); it != nameToIid.end()) {
      if (it->second != iid) {
        throw std::logic_error("Perfetto name interned with mismatched iid: " +
                               name);
      }
      return;
    }
    if (auto it = iidToName.find(iid); it != iidToName.end()) {
      if (it->second != name) {
        throw std::logic_error("Perfetto iid interned with mismatched name");
      }
      return;
    }

    nameToIid.emplace(name, iid);
    iidToName.emplace(iid, name);
    nextIid = std::max(nextIid, iid + 1);
  }

  uint64_t get(const std::string &name) const {
    auto it = nameToIid.find(name);
    if (it == nameToIid.end()) {
      throw std::logic_error("Perfetto name was not interned: " + name);
    }
    return it->second;
  }

  bool empty() const { return iidToName.empty(); }

  const std::map<uint64_t, std::string> &entries() const { return iidToName; }

private:
  uint64_t nextIid = 1;
  std::unordered_map<std::string, uint64_t> nameToIid;
  std::map<uint64_t, std::string> iidToName;
};

struct PerfettoInternedNames {
  PerfettoInternedStringTable eventCategories;
  PerfettoInternedStringTable eventNames;
  PerfettoInternedStringTable debugAnnotationNames;
  PerfettoInternedStringTable debugAnnotationStringValues;
  std::array<std::optional<uint64_t>,
             static_cast<size_t>(PerfettoEventCategory::Count)>
      eventCategoryIids;

  bool empty() const {
    return eventCategories.empty() && eventNames.empty() &&
           debugAnnotationNames.empty() && debugAnnotationStringValues.empty();
  }
};

struct PerfettoSliceEventInternedIds {
  uint64_t nameIid{};
  uint64_t categoryIid{};
};

uint64_t getPerfettoLaneTrackUuid(uint64_t laneId) {
  return kPerfettoLaneTrackUuidBase + laneId;
}

uint64_t internPerfettoString(PerfettoInternedStringTable &internedStrings,
                              PerfettoInternedStringTable &newInternedStrings,
                              const std::string &name) {
  const auto [iid, inserted] = internedStrings.intern(name);
  if (inserted) {
    newInternedStrings.internWithIid(name, iid);
  }
  return iid;
}

const char *getPerfettoEventCategoryName(PerfettoEventCategory category) {
  switch (category) {
  case PerfettoEventCategory::Scope:
    return "scope";
  case PerfettoEventCategory::Metric:
    return "metric";
  case PerfettoEventCategory::Kernel:
    return "kernel";
  case PerfettoEventCategory::Count:
    break;
  }
  throw std::logic_error("Invalid Perfetto event category");
}

uint64_t internPerfettoEventCategory(PerfettoInternedNames &internedNames,
                                     PerfettoInternedNames &newInternedNames,
                                     PerfettoEventCategory category) {
  const auto *categoryName = getPerfettoEventCategoryName(category);
  auto &categoryIid =
      internedNames.eventCategoryIids[static_cast<size_t>(category)];
  if (categoryIid.has_value()) {
    return *categoryIid;
  }

  const auto [iid, inserted] =
      internedNames.eventCategories.intern(categoryName);
  categoryIid = iid;
  if (inserted) {
    newInternedNames.eventCategories.internWithIid(categoryName, iid);
  }
  return iid;
}

void appendTracePacket(ProtoWriter &trace, const ProtoWriter &packet) {
  trace.writeMessage(/*Trace.packet=*/1, packet);
}

void appendTrackDescriptorPacket(ProtoWriter &trace, const ProtoWriter &track) {
  ProtoWriter packet;
  packet.writeMessage(/*TracePacket.track_descriptor=*/60, track);
  appendTracePacket(trace, packet);
}

// Perfetto incremental state is scoped to a packet sequence, not to a track.
// All TracePackets that share the same trusted_packet_sequence_id belong to the
// same writer-local sequence and can therefore share:
// - interned strings in TracePacket.interned_data
// - any other sequence-local defaults / cached state
//
// This file emits CPU, graph, and GPU events onto different tracks, but all of
// those packets are written by this one synthetic writer, so they intentionally
// use the same trusted_packet_sequence_id.
//
// The sequence_flags parameter controls how Perfetto should interpret the
// sequence-local cached state for this packet:
// - SEQ_NEEDS_INCREMENTAL_STATE means this packet either defines or depends on
//   sequence-scoped cached state such as interned strings.
// - SEQ_INCREMENTAL_STATE_CLEARED means the decoder should discard any older
//   cached state for this sequence before applying this packet.
//
// In practice:
// - the first interned-data packet on this sequence sets both flags
// - later packets on the same sequence set SEQ_NEEDS_INCREMENTAL_STATE only
//
// If we changed the sequence id between packets, later TrackEvent name_iid /
// category_iids / debug annotation name_iid references would no longer resolve
// against the interned_data we emitted earlier.
void setTracePacketSequence(ProtoWriter &packet, uint32_t sequenceFlags) {
  packet.writeUInt32(/*TracePacket.trusted_packet_sequence_id=*/10,
                     kPerfettoTracePacketSequenceId);
  packet.writeUInt32(/*TracePacket.sequence_flags=*/13, sequenceFlags);
}

void appendCallStackAnnotations(std::vector<PerfettoAnnotation> &annotations,
                                const std::vector<Context> &contexts,
                                PerfettoInternedNames &internedNames,
                                PerfettoInternedNames &newInternedNames) {
  for (size_t i = 0; i < contexts.size(); ++i) {
    PerfettoAnnotation annotation;
    annotation.nameIid = internPerfettoString(
        internedNames.debugAnnotationNames,
        newInternedNames.debugAnnotationNames,
        "call_stack_" + std::to_string(i));
    annotation.stringValueIid = internPerfettoString(
        internedNames.debugAnnotationStringValues,
        newInternedNames.debugAnnotationStringValues, contexts[i].name);
    annotation.kind = PerfettoAnnotation::Kind::String;
    annotations.push_back(std::move(annotation));
  }
}

void appendFlexibleMetricAnnotations(
    std::vector<PerfettoAnnotation> &annotations,
    const DataEntry::FlexibleMetricMap &flexibleMetrics,
    PerfettoInternedNames &internedNames,
    PerfettoInternedNames &newInternedNames) {
  for (const auto &[metricName, metricValue] : flexibleMetrics) {
    PerfettoAnnotation annotation;
    annotation.nameIid = internPerfettoString(
        internedNames.debugAnnotationNames,
        newInternedNames.debugAnnotationNames, "metric." + metricName);
    const auto &value = metricValue.getValues()[0];
    std::visit(
        [&annotation, &value, &internedNames,
         &newInternedNames](const auto &metricValue) {
          using T = std::decay_t<decltype(metricValue)>;
          if constexpr (std::is_same_v<T, uint64_t>) {
            annotation.uintValue = metricValue;
            annotation.kind = PerfettoAnnotation::Kind::UInt64;
          } else if constexpr (std::is_same_v<T, int64_t>) {
            annotation.intValue = metricValue;
            annotation.kind = PerfettoAnnotation::Kind::Int64;
          } else if constexpr (std::is_same_v<T, double>) {
            annotation.doubleValue = metricValue;
            annotation.kind = PerfettoAnnotation::Kind::Double;
          } else if constexpr (std::is_same_v<T, std::string>) {
            annotation.stringValueIid = internPerfettoString(
                internedNames.debugAnnotationStringValues,
                newInternedNames.debugAnnotationStringValues, metricValue);
            annotation.kind = PerfettoAnnotation::Kind::String;
          } else if constexpr (std::is_same_v<T, std::vector<uint64_t>> ||
                               std::is_same_v<T, std::vector<int64_t>> ||
                               std::is_same_v<T, std::vector<double>>) {
            annotation.stringValueIid = internPerfettoString(
                internedNames.debugAnnotationStringValues,
                newInternedNames.debugAnnotationStringValues,
                details::formatFlexibleMetricValue(value));
            annotation.kind = PerfettoAnnotation::Kind::String;
          } else {
            static_assert(sizeof(T) == 0, "Unsupported MetricValueType");
          }
        },
        value);
    annotations.push_back(std::move(annotation));
  }
}

PerfettoSliceEventInternedIds
internSliceEvent(PerfettoInternedNames &internedNames,
                 PerfettoInternedNames &newInternedNames,
                 const std::string &name, PerfettoEventCategory category) {
  PerfettoSliceEventInternedIds ids;
  ids.nameIid = internPerfettoString(internedNames.eventNames,
                                     newInternedNames.eventNames, name);
  ids.categoryIid =
      internPerfettoEventCategory(internedNames, newInternedNames, category);
  return ids;
}

void appendInternedDataPacket(ProtoWriter &trace,
                              const PerfettoInternedNames &internedNames,
                              bool incrementalStateCleared) {
  if (internedNames.empty()) {
    return;
  }

  ProtoWriter internedData;
  for (const auto &[iid, name] : internedNames.eventCategories.entries()) {
    ProtoWriter entry;
    entry.writeUInt64(/*InternedString.iid=*/1, iid);
    entry.writeString(/*InternedString.name=*/2, name);
    internedData.writeMessage(/*InternedData.event_categories=*/1, entry);
  }
  for (const auto &[iid, name] : internedNames.eventNames.entries()) {
    ProtoWriter entry;
    entry.writeUInt64(/*InternedString.iid=*/1, iid);
    entry.writeString(/*InternedString.name=*/2, name);
    internedData.writeMessage(/*InternedData.event_names=*/2, entry);
  }
  for (const auto &[iid, name] : internedNames.debugAnnotationNames.entries()) {
    ProtoWriter entry;
    entry.writeUInt64(/*DebugAnnotationName.iid=*/1, iid);
    entry.writeString(/*DebugAnnotationName.name=*/2, name);
    internedData.writeMessage(/*InternedData.debug_annotation_names=*/3, entry);
  }
  for (const auto &[iid, value] :
       internedNames.debugAnnotationStringValues.entries()) {
    ProtoWriter entry;
    entry.writeUInt64(/*InternedString.iid=*/1, iid);
    entry.writeString(/*InternedString.name=*/2, value);
    internedData.writeMessage(
        /*InternedData.debug_annotation_string_values=*/29, entry);
  }

  ProtoWriter packet;
  packet.writeMessage(/*TracePacket.interned_data=*/12, internedData);
  setTracePacketSequence(
      packet,
      kPerfettoSeqNeedsIncrementalState |
          (incrementalStateCleared ? kPerfettoSeqIncrementalStateCleared : 0));
  appendTracePacket(trace, packet);
}

void appendTrackEventPacket(ProtoWriter &trace, uint64_t timestampNs,
                            uint32_t type, uint64_t trackUuid,
                            const PerfettoSliceEventInternedIds &eventIds,
                            const std::vector<PerfettoAnnotation> &annotations,
                            std::optional<uint64_t> flowId,
                            std::optional<uint64_t> terminatingFlowId) {
  ProtoWriter trackEvent;
  trackEvent.writeUInt32(/*TrackEvent.type=*/9, type);
  trackEvent.writeUInt64(/*TrackEvent.track_uuid=*/11, trackUuid);
  trackEvent.writeUInt64(/*TrackEvent.name_iid=*/10, eventIds.nameIid);
  if (type == 1) {
    trackEvent.writeUInt64(/*TrackEvent.category_iids=*/3,
                           eventIds.categoryIid);
    for (const auto &annotation : annotations) {
      ProtoWriter message;
      message.writeUInt64(/*DebugAnnotation.name_iid=*/1,
                          annotation.nameIid);
      switch (annotation.kind) {
      case PerfettoAnnotation::Kind::String:
        message.writeUInt64(/*DebugAnnotation.string_value_iid=*/17,
                            annotation.stringValueIid);
        break;
      case PerfettoAnnotation::Kind::UInt64:
        message.writeUInt64(/*DebugAnnotation.uint_value=*/3,
                            annotation.uintValue);
        break;
      case PerfettoAnnotation::Kind::Int64:
        message.writeInt64(/*DebugAnnotation.int_value=*/4,
                           annotation.intValue);
        break;
      case PerfettoAnnotation::Kind::Double:
        message.writeDouble(/*DebugAnnotation.double_value=*/5,
                            annotation.doubleValue);
        break;
      case PerfettoAnnotation::Kind::Bool:
        message.writeBool(/*DebugAnnotation.bool_value=*/2,
                          annotation.boolValue);
        break;
      }
      trackEvent.writeMessage(/*TrackEvent.debug_annotations=*/4, message);
    }
    if (flowId.has_value()) {
      trackEvent.writeUInt64(/*TrackEvent.flow_ids=*/36, *flowId);
    }
    if (terminatingFlowId.has_value()) {
      trackEvent.writeUInt64(/*TrackEvent.terminating_flow_ids=*/42,
                             *terminatingFlowId);
    }
  }

  ProtoWriter packet;
  packet.writeUInt64(/*TracePacket.timestamp=*/8, timestampNs);
  packet.writeMessage(/*TracePacket.track_event=*/11, trackEvent);
  setTracePacketSequence(packet, kPerfettoSeqNeedsIncrementalState);
  appendTracePacket(trace, packet);
}

uint64_t getRelativeTimestamp(uint64_t minTimeStamp, uint64_t timestamp) {
  return minTimeStamp == std::numeric_limits<uint64_t>::max()
             ? uint64_t{0}
             : timestamp - minTimeStamp;
}

void appendSlicePackets(ProtoWriter &trace, uint64_t minTimeStamp,
                        uint64_t trackUuid, uint64_t startTimeNs,
                        uint64_t endTimeNs, const std::string &name,
                        PerfettoEventCategory category,
                        const std::vector<PerfettoAnnotation> &annotations,
                        std::optional<uint64_t> flowId,
                        std::optional<uint64_t> terminatingFlowId,
                        PerfettoInternedNames &newInternedNames,
                        bool incrementalStateCleared,
                        PerfettoInternedNames &internedNames) {
  const auto eventIds =
      internSliceEvent(internedNames, newInternedNames, name, category);
  appendInternedDataPacket(trace, newInternedNames, incrementalStateCleared);

  appendTrackEventPacket(trace, getRelativeTimestamp(minTimeStamp, startTimeNs),
                         1, trackUuid, eventIds, annotations, flowId,
                         terminatingFlowId);
  appendTrackEventPacket(trace, getRelativeTimestamp(minTimeStamp, endTimeNs),
                         2, trackUuid, eventIds, annotations, std::nullopt,
                         std::nullopt);
}

void appendCpuTrackPackets(ProtoWriter &trace, const TraceDump &traceDump,
                           PerfettoInternedNames &internedNames) {
  for (const auto &[threadId, cpuEvents] : traceDump.cpuScopeEvents) {
    const auto trackUuid =
        getPerfettoLaneTrackUuid(details::getCpuLaneId(threadId));
    for (const auto &event : cpuEvents) {
      PerfettoInternedNames newInternedNames;
      const bool incrementalStateCleared = internedNames.empty();
      std::vector<PerfettoAnnotation> annotations;
      appendCallStackAnnotations(annotations, event.contexts, internedNames,
                                 newInternedNames);
      std::string name;
      PerfettoEventCategory category;
      if (event.flexibleMetrics != nullptr && !event.flexibleMetrics->empty()) {
        name = details::buildFlexibleMetricEventName(event.contexts,
                                                     *event.flexibleMetrics);
        category = PerfettoEventCategory::Metric;
        appendFlexibleMetricAnnotations(annotations, *event.flexibleMetrics,
                                        internedNames, newInternedNames);
      } else {
        name = event.contexts.empty() ? "" : event.contexts.back().name;
        category = PerfettoEventCategory::Scope;
      }

      std::optional<uint64_t> flowId;
      if (event.targetEventId != details::kNoLaunchEventId) {
        flowId = kPerfettoFlowIdBase + event.eventId;
      }

      appendSlicePackets(trace, traceDump.minTimeStamp, trackUuid,
                         event.startTimeNs, event.endTimeNs, name, category,
                         annotations, flowId, std::nullopt, newInternedNames,
                         incrementalStateCleared, internedNames);
    }
  }
}

void appendGraphTrackPackets(ProtoWriter &trace, const TraceDump &traceDump,
                             PerfettoInternedNames &internedNames) {
  for (const auto &[streamId, graphEvents] : traceDump.graphScopeEvents) {
    const auto trackUuid =
        getPerfettoLaneTrackUuid(details::getGraphLaneId(streamId));
    for (const auto &event : graphEvents) {
      PerfettoInternedNames newInternedNames;
      const bool incrementalStateCleared = internedNames.empty();
      std::vector<PerfettoAnnotation> annotations;
      std::string name;
      PerfettoEventCategory category;
      if (event.flexibleMetrics != nullptr && !event.flexibleMetrics->empty()) {
        name = details::buildFlexibleMetricEventName(event.context,
                                                     *event.flexibleMetrics);
        category = PerfettoEventCategory::Metric;
        appendFlexibleMetricAnnotations(annotations, *event.flexibleMetrics,
                                        internedNames, newInternedNames);
      } else {
        name = event.context.name;
        category = PerfettoEventCategory::Scope;
      }

      appendSlicePackets(trace, traceDump.minTimeStamp, trackUuid,
                         event.startTimeNs, event.endTimeNs, name, category,
                         annotations, std::nullopt, std::nullopt,
                         newInternedNames, incrementalStateCleared,
                         internedNames);
    }
  }
}

void appendKernelTrackPackets(ProtoWriter &trace, const TraceDump &traceDump,
                              PerfettoInternedNames &internedNames) {
  for (const auto &[streamId, streamKernelEvents] : traceDump.kernelEvents) {
    const auto trackUuid =
        getPerfettoLaneTrackUuid(details::getGpuLaneId(streamId));
    for (const auto &event : streamKernelEvents) {
      PerfettoInternedNames newInternedNames;
      const bool incrementalStateCleared = internedNames.empty();
      std::vector<PerfettoAnnotation> annotations;
      appendCallStackAnnotations(annotations, event.contexts, internedNames,
                                 newInternedNames);
      if (event.flexibleMetrics != nullptr) {
        appendFlexibleMetricAnnotations(annotations, *event.flexibleMetrics,
                                        internedNames, newInternedNames);
      }

      std::optional<uint64_t> terminatingFlowId;
      if (event.launchEventId != details::kNoLaunchEventId) {
        terminatingFlowId = kPerfettoFlowIdBase + event.launchEventId;
      }

      appendSlicePackets(trace, traceDump.minTimeStamp, trackUuid,
                         event.getStartTimeNs(), event.getEndTimeNs(),
                         event.getName(), PerfettoEventCategory::Kernel,
                         annotations, std::nullopt, terminatingFlowId,
                         newInternedNames, incrementalStateCleared,
                         internedNames);
    }
  }
}

} // namespace

void dumpPerfettoTraceData(const TraceDump &traceDump, std::ostream &os) {
  std::map<uint64_t, PerfettoTrack> tracks;
  for (const auto &[threadId, _] : traceDump.cpuScopeEvents) {
    const auto laneId = details::getCpuLaneId(threadId);
    tracks.emplace(getPerfettoLaneTrackUuid(laneId),
                   PerfettoTrack{"CPU Thread " + std::to_string(threadId),
                                 kPerfettoCpuTrackOrderBase +
                                     static_cast<int32_t>(threadId)});
  }
  for (const auto &[streamId, _] : traceDump.graphScopeEvents) {
    const auto laneId = details::getGraphLaneId(streamId);
    tracks.emplace(getPerfettoLaneTrackUuid(laneId),
                   PerfettoTrack{"Graph: Stream " + std::to_string(streamId),
                                 kPerfettoGraphTrackOrderBase +
                                     static_cast<int32_t>(streamId)});
  }
  for (const auto &[streamId, _] : traceDump.kernelEvents) {
    const auto laneId = details::getGpuLaneId(streamId);
    tracks.emplace(getPerfettoLaneTrackUuid(laneId),
                   PerfettoTrack{"GPU Stream " + std::to_string(streamId),
                                 kPerfettoGpuTrackOrderBase +
                                     static_cast<int32_t>(streamId)});
  }

  PerfettoInternedNames internedNames;

  ProtoWriter trace;
  {
    ProtoWriter process;
    process.writeInt32(/*ProcessDescriptor.pid=*/1,
                       static_cast<int32_t>(details::kTraceProcessId));
    process.writeString(/*ProcessDescriptor.process_name=*/6, "Trace");

    ProtoWriter track;
    track.writeUInt64(/*TrackDescriptor.uuid=*/1, kPerfettoProcessTrackUuid);
    track.writeString(/*TrackDescriptor.name=*/2, "Trace");
    track.writeMessage(/*TrackDescriptor.process=*/3, process);
    track.writeUInt32(/*TrackDescriptor.child_ordering=*/11,
                      kPerfettoChildTracksOrderingExplicit);
    appendTrackDescriptorPacket(trace, track);
  }
  for (const auto &[uuid, trackInfo] : tracks) {
    ProtoWriter track;
    track.writeUInt64(/*TrackDescriptor.uuid=*/1, uuid);
    track.writeString(/*TrackDescriptor.name=*/2, trackInfo.name);
    track.writeUInt64(/*TrackDescriptor.parent_uuid=*/5,
                      kPerfettoProcessTrackUuid);
    track.writeInt32(/*TrackDescriptor.sibling_order_rank=*/12,
                     trackInfo.siblingOrderRank);
    appendTrackDescriptorPacket(trace, track);
  }

  appendCpuTrackPackets(trace, traceDump, internedNames);
  appendGraphTrackPackets(trace, traceDump, internedNames);
  appendKernelTrackPackets(trace, traceDump, internedNames);

  const auto &data = trace.data();
  os.write(data.data(), static_cast<std::streamsize>(data.size()));
}

} // namespace proton::trace_data_dump
