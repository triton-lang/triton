#include "python/triton/experimental/gsan/src/GSan.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

struct PyScalarClock {
  uint32_t epoch = 0;
  uint32_t threadId = 0;
  gsan::AtomicScope scope = gsan::AtomicScope::NonAtomic;
};

struct PyShadowCell {
  std::array<PyScalarClock, gsan::ShadowCell::kReadClockSize> readClocks;
  PyScalarClock writeClock;
  uint32_t numReads = 0;
};

struct PyGlobalState {
  uintptr_t reserveBase = 0;
  uintptr_t globalsBase = 0;
  uint32_t rngSeed = 0;
  uint32_t numSms = 0;
  uint32_t numDevices = 0;
  uint32_t numThreads = 0;
  uint32_t clockBufferSize = 0;
};

struct PyGSanThreadState {
  uintptr_t globalsPtr = 0;
  uintptr_t reserveBase = 0;
  uint32_t numReads = 0;
  bool clockBufferDirty = false;
  uint32_t clockBufferHead = 0;
  uint32_t threadId = 0;
  uint32_t numThreads = 0;
  uint32_t clockBufferSize = 0;
  std::vector<uint32_t> vectorClock;
  std::vector<uint32_t> clockBuffer;
};

const char *atomicScopeName(gsan::AtomicScope scope) {
  switch (scope) {
  case gsan::AtomicScope::NonAtomic:
    return "non_atomic";
  case gsan::AtomicScope::CTA:
    return "cta";
  case gsan::AtomicScope::GPU:
    return "gpu";
  case gsan::AtomicScope::System:
    return "system";
  }
  return "unknown";
}

const char *atomicScopeEnumName(gsan::AtomicScope scope) {
  switch (scope) {
  case gsan::AtomicScope::NonAtomic:
    return "NON_ATOMIC";
  case gsan::AtomicScope::CTA:
    return "CTA";
  case gsan::AtomicScope::GPU:
    return "GPU";
  case gsan::AtomicScope::System:
    return "SYSTEM";
  }
  return "UNKNOWN";
}

std::string atomicScopeStr(gsan::AtomicScope scope) {
  return std::string("AtomicScope.") + atomicScopeEnumName(scope);
}

std::string scalarClockStr(const PyScalarClock &c) {
  std::ostringstream oss;
  oss << "ScalarClock(epoch=" << static_cast<uint64_t>(c.epoch)
      << ", thread_id=" << static_cast<uint64_t>(c.threadId)
      << ", scope=" << atomicScopeStr(c.scope) << ")";
  return oss.str();
}

template <typename T>
std::string vectorPreviewStr(const std::vector<T> &values, size_t limit = 8) {
  std::ostringstream oss;
  oss << "[";
  size_t n = values.size();
  size_t shown = n < limit ? n : limit;
  for (size_t i = 0; i < shown; ++i) {
    if (i != 0)
      oss << ", ";
    oss << values[i];
  }
  if (n > limit)
    oss << ", ...";
  oss << "]";
  return oss.str();
}

std::string shadowCellStr(const PyShadowCell &cell) {
  std::ostringstream oss;
  oss << "ShadowCell(read_clocks=[";
  for (size_t i = 0; i < cell.readClocks.size(); ++i) {
    if (i != 0)
      oss << ", ";
    oss << scalarClockStr(cell.readClocks[i]);
  }
  oss << "], write_clock=" << scalarClockStr(cell.writeClock)
      << ", num_reads=" << static_cast<uint64_t>(cell.numReads) << ")";
  return oss.str();
}

std::string globalStateStr(const PyGlobalState &s) {
  std::ostringstream oss;
  oss << "GlobalState(reserve_base=" << s.reserveBase
      << ", globals_base=" << s.globalsBase << ", rng_seed=" << s.rngSeed
      << ", num_sms=" << s.numSms << ", num_devices=" << s.numDevices
      << ", num_threads=" << s.numThreads
      << ", clock_buffer_size=" << s.clockBufferSize << ")";
  return oss.str();
}

std::string threadStateStr(const PyGSanThreadState &s) {
  std::ostringstream oss;
  oss << "ThreadState(globals_ptr=" << s.globalsPtr
      << ", reserve_base=" << s.reserveBase << ", num_reads=" << s.numReads
      << ", clock_buffer_dirty=" << (s.clockBufferDirty ? "True" : "False")
      << ", clock_buffer_head=" << s.clockBufferHead
      << ", thread_id=" << s.threadId << ", num_threads=" << s.numThreads
      << ", clock_buffer_size=" << s.clockBufferSize;
  if (static_cast<size_t>(s.threadId) < s.vectorClock.size())
    oss << ", active_epoch=" << s.vectorClock[s.threadId];
  oss << ", vector_clock=" << vectorPreviewStr(s.vectorClock)
      << " (len=" << s.vectorClock.size() << ")"
      << ", clock_buffer=" << vectorPreviewStr(s.clockBuffer)
      << " (len=" << s.clockBuffer.size() << "))";
  return oss.str();
}

PyScalarClock toPyScalarClock(const gsan::ScalarClock &clock) {
  PyScalarClock out;
  out.epoch = clock.epoch;
  out.threadId = static_cast<uint16_t>(clock.threadId);
  out.scope = clock.scope;
  return out;
}

PyShadowCell toPyShadowCell(const gsan::ShadowCell &cell) {
  PyShadowCell out;
  for (size_t i = 0; i < gsan::ShadowCell::kReadClockSize; ++i)
    out.readClocks[i] = toPyScalarClock(cell.readClocks[i]);
  out.writeClock = toPyScalarClock(cell.writeClock);
  out.numReads = cell.numReads;
  return out;
}

PyGlobalState toPyGlobalState(const gsan::GlobalState &state) {
  PyGlobalState out;
  out.reserveBase = static_cast<uintptr_t>(state.reserveBase);
  out.globalsBase = static_cast<uintptr_t>(state.globalsBase);
  out.rngSeed = state.rngSeed;
  out.numSms = state.numSms;
  out.numDevices = state.numDevices;
  out.numThreads = state.numThreads;
  out.clockBufferSize = state.clockBufferSize;
  return out;
}

constexpr size_t kThreadStateHeaderSize =
    offsetof(gsan::ThreadState, vectorClock);
constexpr size_t kThreadStateSize = sizeof(gsan::ThreadState);

inline uintptr_t roundUp(uintptr_t ptr, uintptr_t align) {
  return (ptr % align) == 0 ? ptr : ptr + align - (ptr % align);
}

size_t threadStateStrideBytes(uint16_t numThreads, uint16_t clockBufferSize) {
  auto clocksPerThread = static_cast<size_t>(1) + clockBufferSize;
  auto clockWords = static_cast<size_t>(numThreads) * clocksPerThread;
  return kThreadStateSize + sizeof(gsan::epoch_t) * clockWords;
}

PyGSanThreadState decodeThreadStateBytes(const char *bytes, size_t size,
                                         uint16_t numThreads,
                                         uint16_t clockBufferSize) {
  const size_t stride = threadStateStrideBytes(numThreads, clockBufferSize);
  if (size < stride) {
    throw py::value_error("decode_thread_state expected at least " +
                          std::to_string(stride) + " bytes");
  }

  PyGSanThreadState out;
  out.numThreads = numThreads;
  out.clockBufferSize = clockBufferSize;

  gsan::ThreadState state;
  std::memcpy(&state, bytes, sizeof(gsan::ThreadState));

  out.globalsPtr = reinterpret_cast<uintptr_t>(state.globals);
  out.reserveBase = state.reserveBase;
  out.numReads = state.numReads;
  out.clockBufferDirty = state.clockBufferDirty;
  out.clockBufferHead = state.clockBufferHead;
  out.threadId = state.threadId;

  const auto *epochs =
      reinterpret_cast<const gsan::epoch_t *>(bytes + kThreadStateHeaderSize);
  out.vectorClock.resize(numThreads);
  for (size_t i = 0; i < numThreads; ++i)
    out.vectorClock[i] = epochs[i];

  size_t clockBufferElems = static_cast<size_t>(numThreads) * clockBufferSize;
  out.clockBuffer.resize(clockBufferElems);
  for (size_t i = 0; i < clockBufferElems; ++i)
    out.clockBuffer[i] = epochs[numThreads + i];

  return out;
}

} // namespace

void init_gsan_testing(py::module &&m) {
  m.doc() = "GSan testing helpers";

  py::enum_<gsan::AtomicScope>(m, "AtomicScope")
      .value("NON_ATOMIC", gsan::AtomicScope::NonAtomic)
      .value("CTA", gsan::AtomicScope::CTA)
      .value("GPU", gsan::AtomicScope::GPU)
      .value("SYSTEM", gsan::AtomicScope::System)
      .def("__str__",
           [](gsan::AtomicScope scope) { return atomicScopeStr(scope); })
      .def("__repr__",
           [](gsan::AtomicScope scope) { return atomicScopeStr(scope); });

  py::class_<PyScalarClock>(m, "ScalarClock")
      .def(py::init(
               [](uint64_t epoch, uint64_t threadId, gsan::AtomicScope scope) {
                 PyScalarClock out;
                 out.epoch = static_cast<uint32_t>(epoch);
                 out.threadId = static_cast<uint32_t>(threadId);
                 out.scope = scope;
                 return out;
               }),
           py::arg("epoch"), py::arg("thread_id"), py::arg("scope"))
      .def_property_readonly(
          "epoch",
          [](const PyScalarClock &c) { return static_cast<uint64_t>(c.epoch); })
      .def_property_readonly("thread_id",
                             [](const PyScalarClock &c) {
                               return static_cast<uint64_t>(c.threadId);
                             })
      .def_property_readonly("scope",
                             [](const PyScalarClock &c) { return c.scope; })
      .def_property_readonly(
          "scope_name",
          [](const PyScalarClock &c) { return atomicScopeName(c.scope); })
      .def(
          "__eq__",
          [](const PyScalarClock &lhs, const PyScalarClock &rhs) {
            return lhs.epoch == rhs.epoch && lhs.threadId == rhs.threadId &&
                   lhs.scope == rhs.scope;
          },
          py::is_operator())
      .def("__str__", [](const PyScalarClock &c) { return scalarClockStr(c); })
      .def("__repr__",
           [](const PyScalarClock &c) { return scalarClockStr(c); });

  py::class_<PyShadowCell>(m, "ShadowCell")
      .def_property_readonly(
          "read_clocks",
          [](const PyShadowCell &cell) { return cell.readClocks; })
      .def_property_readonly(
          "write_clock",
          [](const PyShadowCell &cell) { return cell.writeClock; })
      .def_property_readonly("num_reads",
                             [](const PyShadowCell &cell) {
                               return static_cast<uint64_t>(cell.numReads);
                             })
      .def("__str__",
           [](const PyShadowCell &cell) { return shadowCellStr(cell); })
      .def("__repr__",
           [](const PyShadowCell &cell) { return shadowCellStr(cell); });

  py::class_<PyGlobalState>(m, "GlobalState")
      .def_property_readonly(
          "reserve_base", [](const PyGlobalState &s) { return s.reserveBase; })
      .def_property_readonly(
          "globals_base", [](const PyGlobalState &s) { return s.globalsBase; })
      .def_property_readonly("rng_seed",
                             [](const PyGlobalState &s) { return s.rngSeed; })
      .def_property_readonly("num_sms",
                             [](const PyGlobalState &s) { return s.numSms; })
      .def_property_readonly(
          "num_devices", [](const PyGlobalState &s) { return s.numDevices; })
      .def_property_readonly(
          "num_threads", [](const PyGlobalState &s) { return s.numThreads; })
      .def_property_readonly(
          "clock_buffer_size",
          [](const PyGlobalState &s) { return s.clockBufferSize; })
      .def("__str__", [](const PyGlobalState &s) { return globalStateStr(s); })
      .def("__repr__",
           [](const PyGlobalState &s) { return globalStateStr(s); });

  py::class_<PyGSanThreadState>(m, "ThreadState")
      .def_property_readonly(
          "globals_ptr",
          [](const PyGSanThreadState &s) { return s.globalsPtr; })
      .def_property_readonly(
          "reserve_base",
          [](const PyGSanThreadState &s) { return s.reserveBase; })
      .def_property_readonly(
          "num_reads", [](const PyGSanThreadState &s) { return s.numReads; })
      .def_property_readonly(
          "clock_buffer_dirty",
          [](const PyGSanThreadState &s) { return s.clockBufferDirty; })
      .def_property_readonly(
          "clock_buffer_head",
          [](const PyGSanThreadState &s) { return s.clockBufferHead; })
      .def_property_readonly(
          "thread_id", [](const PyGSanThreadState &s) { return s.threadId; })
      .def_property_readonly(
          "num_threads",
          [](const PyGSanThreadState &s) { return s.numThreads; })
      .def_property_readonly(
          "clock_buffer_size",
          [](const PyGSanThreadState &s) { return s.clockBufferSize; })
      .def_property_readonly(
          "vector_clock",
          [](const PyGSanThreadState &s) { return s.vectorClock; })
      .def_property_readonly(
          "clock_buffer",
          [](const PyGSanThreadState &s) { return s.clockBuffer; })
      .def("__str__",
           [](const PyGSanThreadState &s) { return threadStateStr(s); })
      .def("__repr__",
           [](const PyGSanThreadState &s) { return threadStateStr(s); });

  m.def(
      "shadow_cell_address", gsan::getShadowAddress, py::arg("real_address"),
      "Return the address of the ShadowCell corresponding to a real address.");

  m.attr("SHADOW_CELL_SIZE_BYTES") = sizeof(gsan::ShadowCell);
  m.attr("SHADOW_GRANULARITY_BYTES") = gsan::kShadowMemGranularityBytes;
  m.attr("GLOBAL_STATE_SIZE_BYTES") = sizeof(gsan::GlobalState);
  m.attr("THREAD_STATE_HEADER_SIZE_BYTES") = kThreadStateHeaderSize;
  m.attr("PER_DEVICE_STATE_STRIDE_BYTES") = gsan::kPerDeviceStateStride;

  m.def(
      "thread_state_stride_bytes",
      [](uint64_t numThreads, uint64_t clockBufferSize) -> uint64_t {
        return static_cast<uint64_t>(
            threadStateStrideBytes(static_cast<uint16_t>(numThreads),
                                   static_cast<uint16_t>(clockBufferSize)));
      },
      py::arg("num_threads"), py::arg("clock_buffer_size"),
      "Return the full byte size/stride of one ThreadState record.");

  m.def(
      "thread_state_address",
      [](uint64_t globalStateAddress, uint64_t numThreads,
         uint64_t clockBufferSize, uint64_t smid) -> uint64_t {
        uintptr_t base = static_cast<uintptr_t>(globalStateAddress);
        base = roundUp(base + sizeof(gsan::GlobalState),
                       alignof(gsan::ThreadState));
        uintptr_t stride = static_cast<uintptr_t>(
            threadStateStrideBytes(static_cast<uint16_t>(numThreads),
                                   static_cast<uint16_t>(clockBufferSize)));
        return static_cast<uint64_t>(base + stride * smid);
      },
      py::arg("global_state_address"), py::arg("num_threads"),
      py::arg("clock_buffer_size"), py::arg("smid"),
      "Return the address of the per-SM ThreadState record for the given SM "
      "index.");

  m.def(
      "decode_global_state",
      [](py::bytes data) -> PyGlobalState {
        std::string bytes = data;
        if (bytes.size() < static_cast<ssize_t>(sizeof(gsan::GlobalState))) {
          throw py::value_error("decode_global_state expected at least " +
                                std::to_string(sizeof(gsan::GlobalState)) +
                                " bytes");
        }
        gsan::GlobalState state{};
        std::memcpy(&state, bytes.data(), sizeof(state));
        return toPyGlobalState(state);
      },
      py::arg("data"),
      "Decode a GlobalState from a bytes object and return a typed object.");

  m.def(
      "decode_shadow_cell",
      [](py::bytes data) -> PyShadowCell {
        std::string bytes = data;
        if (bytes.size() < static_cast<ssize_t>(sizeof(gsan::ShadowCell))) {
          throw py::value_error("decode_shadow_cell expected at least " +
                                std::to_string(sizeof(gsan::ShadowCell)) +
                                " bytes");
        }

        gsan::ShadowCell cell{};
        std::memcpy(&cell, bytes.data(), sizeof(cell));
        return toPyShadowCell(cell);
      },
      py::arg("data"),
      "Decode a ShadowCell from a bytes object and return a typed object.");

  m.def(
      "decode_thread_state",
      [](py::bytes data, uint64_t numThreads,
         uint64_t clockBufferSize) -> PyGSanThreadState {
        std::string bytes = data;
        return decodeThreadStateBytes(bytes.data(), bytes.size(),
                                      static_cast<uint16_t>(numThreads),
                                      static_cast<uint16_t>(clockBufferSize));
      },
      py::arg("data"), py::arg("num_threads"), py::arg("clock_buffer_size"),
      "Decode a ThreadState from a bytes object and return a typed object.");
}
