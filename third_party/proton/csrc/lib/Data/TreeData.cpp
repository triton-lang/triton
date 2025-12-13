#include "Data/TreeData.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Device.h"

#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <chrono>
#include <functional>
#include <unordered_map>
#include <type_traits>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace proton {

namespace {

class MsgPackWriter {
public:
  void reserve(size_t bytes) { out.reserve(bytes); }

  std::vector<uint8_t> take() && { return std::move(out); }

  void packNil() { out.push_back(0xc0); }

  void packBool(bool value) { out.push_back(value ? 0xc3 : 0xc2); }

  void packUInt(uint64_t value) {
    if (value <= 0x7f) {
      out.push_back(static_cast<uint8_t>(value));
    } else if (value <= 0xff) {
      out.push_back(0xcc);
      out.push_back(static_cast<uint8_t>(value));
    } else if (value <= 0xffff) {
      out.push_back(0xcd);
      writeBE(static_cast<uint16_t>(value));
    } else if (value <= 0xffffffffull) {
      out.push_back(0xce);
      writeBE(static_cast<uint32_t>(value));
    } else {
      out.push_back(0xcf);
      writeBE(static_cast<uint64_t>(value));
    }
  }

  void packInt(int64_t value) {
    if (value >= 0) {
      packUInt(static_cast<uint64_t>(value));
      return;
    }
    if (value >= -32) {
      out.push_back(static_cast<uint8_t>(0xe0 | (value + 32)));
    } else if (value >= std::numeric_limits<int8_t>::min()) {
      out.push_back(0xd0);
      out.push_back(static_cast<uint8_t>(static_cast<int8_t>(value)));
    } else if (value >= std::numeric_limits<int16_t>::min()) {
      out.push_back(0xd1);
      writeBE(static_cast<int16_t>(value));
    } else if (value >= std::numeric_limits<int32_t>::min()) {
      out.push_back(0xd2);
      writeBE(static_cast<int32_t>(value));
    } else {
      out.push_back(0xd3);
      writeBE(static_cast<int64_t>(value));
    }
  }

  void packDouble(double value) {
    out.push_back(0xcb);
    uint64_t bits{};
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    writeBE(bits);
  }

  void packStr(std::string_view value) {
    const auto size = static_cast<uint32_t>(value.size());
    if (size <= 31) {
      out.push_back(static_cast<uint8_t>(0xa0 | size));
    } else if (size <= 0xff) {
      out.push_back(0xd9);
      out.push_back(static_cast<uint8_t>(size));
    } else if (size <= 0xffff) {
      out.push_back(0xda);
      writeBE(static_cast<uint16_t>(size));
    } else {
      out.push_back(0xdb);
      writeBE(static_cast<uint32_t>(size));
    }
    out.insert(out.end(), value.begin(), value.end());
  }

  void packArray(uint32_t size) {
    if (size <= 15) {
      out.push_back(static_cast<uint8_t>(0x90 | size));
    } else if (size <= 0xffff) {
      out.push_back(0xdc);
      writeBE(static_cast<uint16_t>(size));
    } else {
      out.push_back(0xdd);
      writeBE(static_cast<uint32_t>(size));
    }
  }

  void packMap(uint32_t size) {
    if (size <= 15) {
      out.push_back(static_cast<uint8_t>(0x80 | size));
    } else if (size <= 0xffff) {
      out.push_back(0xde);
      writeBE(static_cast<uint16_t>(size));
    } else {
      out.push_back(0xdf);
      writeBE(static_cast<uint32_t>(size));
    }
  }

private:
  template <typename T> void writeBE(T value) {
    using U = std::make_unsigned_t<T>;
    U u = static_cast<U>(value);
    for (int i = sizeof(U) - 1; i >= 0; --i) {
      out.push_back(static_cast<uint8_t>((u >> (i * 8)) & 0xff));
    }
  }

  std::vector<uint8_t> out;
};

void packMetricValue(MsgPackWriter &writer, const MetricValueType &value) {
  std::visit(
      [&](auto &&v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, uint64_t>) {
          writer.packUInt(v);
        } else if constexpr (std::is_same_v<T, int64_t>) {
          writer.packInt(v);
        } else if constexpr (std::is_same_v<T, double>) {
          writer.packDouble(v);
        } else if constexpr (std::is_same_v<T, std::string>) {
          writer.packStr(v);
        } else {
          static_assert(sizeof(T) == 0, "Unsupported MetricValueType");
        }
      },
      value);
}

} // namespace

class TreeData::Tree {
public:
  struct TreeNode : public Context {
    inline static const size_t RootId = 0;
    inline static const size_t DummyId = std::numeric_limits<size_t>::max();

    TreeNode() = default;
    explicit TreeNode(size_t id, const std::string &name)
        : id(id), Context(name) {}
    TreeNode(size_t id, size_t parentId, const std::string &name)
        : id(id), parentId(parentId), Context(name) {}
    virtual ~TreeNode() = default;

    void addChild(const Context &context, size_t id) {
      children.emplace(context.name, id);
      childIds.push_back(id);
    }

    bool hasChild(const Context &context) const {
      return children.find(context.name) != children.end();
    }

    size_t getChild(const Context &context) const {
      return children.at(context.name);
    }

    size_t parentId = DummyId;
    size_t id = DummyId;
    std::unordered_map<std::string, size_t> children = {};
    std::vector<size_t> childIds = {};
    std::map<MetricKind, std::shared_ptr<Metric>> metrics = {};
    std::map<std::string, FlexibleMetric> flexibleMetrics = {};
    friend class Tree;
  };

  Tree() {
    treeNodeMap.try_emplace(TreeNode::RootId, TreeNode::RootId,
                            TreeNode::RootId, "ROOT");
  }

  size_t addNode(const std::vector<Context> &contexts, size_t parentId) {
    for (const auto &context : contexts) {
      parentId = addNode(context, parentId);
    }
    return parentId;
  }

  size_t addNode(const Context &context, size_t parentId) {
    auto &parent = treeNodeMap.at(parentId);
    if (parent.hasChild(context)) {
      return parent.getChild(context);
    }
    auto id = nextContextId++;
    treeNodeMap.try_emplace(id, id, parentId, context.name);
    parent.addChild(context, id);
    return id;
  }

  size_t addNode(const std::vector<Context> &indices) {
    auto parentId = TreeNode::RootId;
    for (auto index : indices) {
      parentId = addNode(index, parentId);
    }
    return parentId;
  }

  TreeNode &getNode(size_t id) { return treeNodeMap.at(id); }

  enum class WalkPolicy { PreOrder, PostOrder };

  template <WalkPolicy walkPolicy, typename FnT> void walk(FnT &&fn) {
    if constexpr (walkPolicy == WalkPolicy::PreOrder) {
      walkPreOrder(TreeNode::RootId, fn);
    } else if constexpr (walkPolicy == WalkPolicy::PostOrder) {
      walkPostOrder(TreeNode::RootId, fn);
    }
  }

  template <typename FnT> void walkPreOrder(size_t contextId, FnT &&fn) {
    fn(getNode(contextId));
    for (auto childId : getNode(contextId).childIds) {
      walkPreOrder(childId, fn);
    }
  }

  template <typename FnT> void walkPostOrder(size_t contextId, FnT &&fn) {
    for (auto childId : getNode(contextId).childIds) {
      walkPostOrder(childId, fn);
    }
    fn(getNode(contextId));
  }

  size_t size() const { return nextContextId; }

private:
  size_t nextContextId = TreeNode::RootId + 1;
  // tree node id -> tree node
  std::unordered_map<size_t, TreeNode> treeNodeMap;
};

json TreeData::buildHatchetJson(TreeData::Tree *tree) const {
  std::vector<json *> jsonNodes(tree->size(), nullptr);
  json output = json::array();
  output.push_back(json::object());
  jsonNodes[TreeData::Tree::TreeNode::RootId] = &(output.back());
  tree->template walk<TreeData::Tree::WalkPolicy::PreOrder>(
      [&](TreeData::Tree::TreeNode &treeNode) {
        const auto contextName = treeNode.name;
        auto contextId = treeNode.id;
        json *jsonNode = jsonNodes[contextId];
        (*jsonNode)["frame"] = {{"name", contextName}, {"type", "function"}};
        (*jsonNode)["metrics"] = json::object();
        auto &metricsJson = (*jsonNode)["metrics"];
        for (auto &[metricKind, metric] : treeNode.metrics) {
          if (metricKind == MetricKind::Kernel) {
            std::shared_ptr<KernelMetric> kernelMetric =
                std::static_pointer_cast<KernelMetric>(metric);
            uint64_t duration = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::Duration));
            uint64_t invocations = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::Invocations));
            uint64_t deviceId = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::DeviceId));
            uint64_t deviceType = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::DeviceType));
            const std::string deviceTypeName =
                getDeviceTypeString(static_cast<DeviceType>(deviceType));
            const auto &durationName =
                kernelMetric->getValueName(KernelMetric::Duration);
            const auto &invocationsName =
                kernelMetric->getValueName(KernelMetric::Invocations);
            const auto &deviceIdName =
                kernelMetric->getValueName(KernelMetric::DeviceId);
            const auto &deviceTypeNameKey =
                kernelMetric->getValueName(KernelMetric::DeviceType);
            const auto deviceIdStr = std::to_string(deviceId);

            metricsJson[durationName] = duration;
            metricsJson[invocationsName] = invocations;
            metricsJson[deviceIdName] = deviceIdStr;
            metricsJson[deviceTypeNameKey] = deviceTypeName;
          } else if (metricKind == MetricKind::PCSampling) {
            auto pcSamplingMetric =
                std::static_pointer_cast<PCSamplingMetric>(metric);
            for (size_t i = 0; i < PCSamplingMetric::Count; i++) {
              const auto &valueName = pcSamplingMetric->getValueName(i);
              std::visit(
                  [&](auto &&value) {
                    metricsJson[valueName] = value;
                  },
                  pcSamplingMetric->getValuesRef()[i]);
            }
          } else if (metricKind == MetricKind::Cycle) {
            auto cycleMetric = std::static_pointer_cast<CycleMetric>(metric);
            uint64_t duration = std::get<uint64_t>(
                cycleMetric->getValue(CycleMetric::Duration));
            double normalizedDuration = std::get<double>(
                cycleMetric->getValue(CycleMetric::NormalizedDuration));
            uint64_t deviceId = std::get<uint64_t>(
                cycleMetric->getValue(CycleMetric::DeviceId));
            uint64_t deviceType = std::get<uint64_t>(
                cycleMetric->getValue(CycleMetric::DeviceType));
            const auto &durationName =
                cycleMetric->getValueName(CycleMetric::Duration);
            const auto &normalizedDurationName =
                cycleMetric->getValueName(CycleMetric::NormalizedDuration);
            const auto &deviceIdName =
                cycleMetric->getValueName(CycleMetric::DeviceId);
            const auto &deviceTypeName =
                cycleMetric->getValueName(CycleMetric::DeviceType);
            const auto deviceIdStr = std::to_string(deviceId);
            const auto deviceTypeStr = std::to_string(deviceType);

            metricsJson[durationName] = duration;
            metricsJson[normalizedDurationName] = normalizedDuration;
            metricsJson[deviceIdName] = deviceIdStr;
            metricsJson[deviceTypeName] = deviceTypeStr;
          } else if (metricKind == MetricKind::Flexible) {
            // Flexible metrics are handled in a different way
          } else {
            throw std::runtime_error("MetricKind not supported");
          }
	        }
	        for (auto &[_, flexibleMetric] : treeNode.flexibleMetrics) {
	          const auto &valueName = flexibleMetric.getValueName(0);
          std::visit(
              [&](auto &&value) { metricsJson[valueName] = value; },
              flexibleMetric.getValuesRef()[0]);
        }
        auto &childrenArray = (*jsonNode)["children"];
        childrenArray = json::array();
        childrenArray.get_ref<json::array_t &>().reserve(treeNode.childIds.size());
        for (auto childId : treeNode.childIds) {
          childrenArray.push_back(json::object());
          jsonNodes[childId] = &childrenArray.back();
        }
      });
  for (const auto &valueName : inclusiveValueNamesCache) {
    output[TreeData::Tree::TreeNode::RootId]["metrics"][valueName] = 0;
  }
  output.push_back(json::object());
  auto &deviceJson = output.back();
  for (auto &[deviceType, deviceIdSet] : deviceIdsCache) {
    auto deviceTypeName =
        getDeviceTypeString(static_cast<DeviceType>(deviceType));
    if (!deviceJson.contains(deviceTypeName))
      deviceJson[deviceTypeName] = json::object();
    for (auto deviceId : deviceIdSet) {
      Device device = getDevice(static_cast<DeviceType>(deviceType), deviceId);
      deviceJson[deviceTypeName][std::to_string(deviceId)] = {
          {"clock_rate", device.clockRate},
          {"memory_clock_rate", device.memoryClockRate},
          {"bus_width", device.busWidth},
          {"arch", device.arch},
          {"num_sms", device.numSms}};
    }
  }
  return output;
}

std::vector<uint8_t> TreeData::buildHatchetMsgPack(TreeData::Tree *tree) const {
  MsgPackWriter writer;
  writer.reserve(16 * 1024 * 1024); // 16 MB

  std::function<void(TreeData::Tree::TreeNode &)> packNode =
      [&](TreeData::Tree::TreeNode &treeNode) {
        writer.packMap(3);

        writer.packStr("frame");
        writer.packMap(2);
        writer.packStr("name");
        writer.packStr(treeNode.name);
        writer.packStr("type");
        writer.packStr("function");

        writer.packStr("metrics");
        uint32_t metricEntries = 0;
        for (auto &[metricKind, metric] : treeNode.metrics) {
          if (metricKind == MetricKind::Kernel) {
            metricEntries += 4;
          } else if (metricKind == MetricKind::PCSampling) {
            metricEntries += PCSamplingMetric::Count;
          } else if (metricKind == MetricKind::Cycle) {
            metricEntries += 4;
          }
        }
        metricEntries += static_cast<uint32_t>(treeNode.flexibleMetrics.size());
        if (treeNode.id == TreeData::Tree::TreeNode::RootId) {
          metricEntries +=
              static_cast<uint32_t>(inclusiveValueNamesCache.size());
        }
        writer.packMap(metricEntries);

        for (auto &[metricKind, metric] : treeNode.metrics) {
          if (metricKind == MetricKind::Kernel) {
            auto kernelMetric = std::static_pointer_cast<KernelMetric>(metric);
            uint64_t duration = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::Duration));
            uint64_t invocations = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::Invocations));
            uint64_t deviceId = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::DeviceId));
            uint64_t deviceType = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::DeviceType));
	            const std::string deviceTypeName =
	                getDeviceTypeString(static_cast<DeviceType>(deviceType));
	            const auto &durationName =
	                kernelMetric->getValueName(KernelMetric::Duration);
	            const auto &invocationsName =
	                kernelMetric->getValueName(KernelMetric::Invocations);
	            const auto &deviceIdName =
	                kernelMetric->getValueName(KernelMetric::DeviceId);
	            const auto &deviceTypeNameKey =
	                kernelMetric->getValueName(KernelMetric::DeviceType);

            writer.packStr(durationName);
            writer.packUInt(duration);
            writer.packStr(invocationsName);
            writer.packUInt(invocations);
            writer.packStr(deviceIdName);
            writer.packStr(std::to_string(deviceId));
            writer.packStr(deviceTypeNameKey);
            writer.packStr(deviceTypeName);
          } else if (metricKind == MetricKind::PCSampling) {
            auto pcSamplingMetric =
                std::static_pointer_cast<PCSamplingMetric>(metric);
	            for (size_t i = 0; i < PCSamplingMetric::Count; i++) {
	              const auto &valueName = pcSamplingMetric->getValueName(i);
	              writer.packStr(valueName);
	              packMetricValue(writer, pcSamplingMetric->getValuesRef()[i]);
	            }
	          } else if (metricKind == MetricKind::Cycle) {
            auto cycleMetric = std::static_pointer_cast<CycleMetric>(metric);
            uint64_t duration = std::get<uint64_t>(
                cycleMetric->getValue(CycleMetric::Duration));
            double normalizedDuration = std::get<double>(
                cycleMetric->getValue(CycleMetric::NormalizedDuration));
            uint64_t deviceId = std::get<uint64_t>(
                cycleMetric->getValue(CycleMetric::DeviceId));
	            uint64_t deviceType = std::get<uint64_t>(
	                cycleMetric->getValue(CycleMetric::DeviceType));
	            const auto &durationName =
	                cycleMetric->getValueName(CycleMetric::Duration);
	            const auto &normalizedDurationName =
	                cycleMetric->getValueName(CycleMetric::NormalizedDuration);
	            const auto &deviceIdName =
	                cycleMetric->getValueName(CycleMetric::DeviceId);
	            const auto &deviceTypeName =
	                cycleMetric->getValueName(CycleMetric::DeviceType);

            writer.packStr(durationName);
            writer.packUInt(duration);
            writer.packStr(normalizedDurationName);
            writer.packDouble(normalizedDuration);
            writer.packStr(deviceIdName);
            writer.packStr(std::to_string(deviceId));
            writer.packStr(deviceTypeName);
            writer.packStr(std::to_string(deviceType));
          }
	        }

	        for (auto &[_, flexibleMetric] : treeNode.flexibleMetrics) {
	          const auto &valueName = flexibleMetric.getValueName(0);
	          writer.packStr(valueName);
	          packMetricValue(writer, flexibleMetric.getValuesRef()[0]);
	        }

        if (treeNode.id == TreeData::Tree::TreeNode::RootId) {
          for (const auto &valueName : inclusiveValueNamesCache) {
            writer.packStr(valueName);
            writer.packUInt(0);
          }
        }

        writer.packStr("children");
        writer.packArray(static_cast<uint32_t>(treeNode.childIds.size()));
        for (auto childId : treeNode.childIds) {
          packNode(tree->getNode(childId));
        }
      };

  writer.packArray(2);
  packNode(tree->getNode(TreeData::Tree::TreeNode::RootId));

  writer.packMap(static_cast<uint32_t>(deviceIdsCache.size()));
  for (auto &[deviceType, deviceIdSet] : deviceIdsCache) {
    const auto deviceTypeName =
        getDeviceTypeString(static_cast<DeviceType>(deviceType));
    writer.packStr(deviceTypeName);
    writer.packMap(static_cast<uint32_t>(deviceIdSet.size()));
    for (auto deviceId : deviceIdSet) {
      Device device = getDevice(static_cast<DeviceType>(deviceType), deviceId);
      writer.packStr(std::to_string(deviceId));
      writer.packMap(5);
      writer.packStr("clock_rate");
      writer.packUInt(device.clockRate);
      writer.packStr("memory_clock_rate");
      writer.packUInt(device.memoryClockRate);
      writer.packStr("bus_width");
      writer.packUInt(device.busWidth);
      writer.packStr("arch");
      writer.packStr(device.arch);
      writer.packStr("num_sms");
      writer.packUInt(device.numSms);
    }
  }

  return std::move(writer).take();
}

void TreeData::enterScope(const Scope &scope) {
  // enterOp and addMetric maybe called from different threads
  std::unique_lock<std::shared_mutex> lock(mutex);
  std::vector<Context> contexts;
  if (contextSource != nullptr)
    contexts = contextSource->getContexts();
  else
    contexts.push_back(scope.name);
  auto contextId = tree->addNode(contexts);
  scopeIdToContextId[scope.scopeId] = contextId;
}

void TreeData::exitScope(const Scope &scope) {}

size_t TreeData::addOp(size_t scopeId, const std::string &name) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeIdIt = scopeIdToContextId.find(scopeId);
  if (scopeIdIt == scopeIdToContextId.end()) {
    // Obtain the current context
    std::vector<Context> contexts;
    if (contextSource != nullptr)
      contexts = contextSource->getContexts();
    // Add an op under the current context
    if (!name.empty())
      contexts.emplace_back(name);
    scopeIdToContextId[scopeId] = tree->addNode(contexts);
  } else {
    // Add a new context under it and update the context
    scopeId = Scope::getNewScopeId();
    scopeIdToContextId[scopeId] =
        tree->addNode(Context(name), scopeIdIt->second);
  }
  return scopeId;
}

size_t TreeData::addOp(size_t scopeId, const std::vector<Context> &contexts) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeIdIt = scopeIdToContextId.find(scopeId);
  if (scopeIdIt == scopeIdToContextId.end()) {
    // Obtain the current context
    std::vector<Context> currentContexts;
    if (contextSource != nullptr)
      currentContexts = contextSource->getContexts();
    // Add an op under the current context
    if (!currentContexts.empty())
      std::merge(currentContexts.begin(), currentContexts.end(),
                 contexts.begin(), contexts.end(), currentContexts.begin());
    scopeIdToContextId[scopeId] = tree->addNode(currentContexts);
  } else {
    // Add a new context under it and update the context
    scopeId = Scope::getNewScopeId();
    scopeIdToContextId[scopeId] = tree->addNode(contexts, scopeIdIt->second);
  }
  return scopeId;
}

void TreeData::addMetric(size_t scopeId, std::shared_ptr<Metric> metric) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeIdIt = scopeIdToContextId.find(scopeId);
  // The profile data is deactivated, ignore the metric
  if (scopeIdIt == scopeIdToContextId.end())
    return;
  auto contextId = scopeIdIt->second;
  auto &node = tree->getNode(contextId);
  if (node.metrics.find(metric->getKind()) == node.metrics.end())
    node.metrics.emplace(metric->getKind(), metric);
  else
    node.metrics[metric->getKind()]->updateMetric(*metric);

  const auto kind = metric->getKind();
  if (kind == MetricKind::Kernel) {
    auto kernelMetric = std::static_pointer_cast<KernelMetric>(metric);
    inclusiveValueNamesCache.insert(
        kernelMetric->getValueName(KernelMetric::Duration));
    inclusiveValueNamesCache.insert(
        kernelMetric->getValueName(KernelMetric::Invocations));
    const uint64_t deviceId = std::get<uint64_t>(
        kernelMetric->getValueRef(KernelMetric::DeviceId));
    const uint64_t deviceType = std::get<uint64_t>(
        kernelMetric->getValueRef(KernelMetric::DeviceType));
    deviceIdsCache[deviceType].insert(deviceId);
  } else if (kind == MetricKind::PCSampling) {
    auto pcSamplingMetric = std::static_pointer_cast<PCSamplingMetric>(metric);
    for (size_t i = 0; i < PCSamplingMetric::Count; i++) {
      inclusiveValueNamesCache.insert(pcSamplingMetric->getValueName(i));
    }
  } else if (kind == MetricKind::Cycle) {
    auto cycleMetric = std::static_pointer_cast<CycleMetric>(metric);
    const uint64_t deviceId =
        std::get<uint64_t>(cycleMetric->getValueRef(CycleMetric::DeviceId));
    const uint64_t deviceType =
        std::get<uint64_t>(cycleMetric->getValueRef(CycleMetric::DeviceType));
    deviceIdsCache[deviceType].insert(deviceId);
  }
}

void TreeData::addMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &metrics) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeIdIt = scopeIdToContextId.find(scopeId);
  // The profile data is deactivated, ignore the metric
  if (scopeIdIt == scopeIdToContextId.end())
    return;
  auto contextId = scopeIdIt->second;
  auto &node = tree->getNode(contextId);
  for (auto [metricName, metricValue] : metrics) {
    if (node.flexibleMetrics.find(metricName) == node.flexibleMetrics.end()) {
      auto [it, inserted] = node.flexibleMetrics.emplace(
          metricName, FlexibleMetric(metricName, metricValue));
      auto &flexibleMetric = it->second;
      if (!flexibleMetric.isExclusive(0)) {
        inclusiveValueNamesCache.insert(flexibleMetric.getValueName(0));
      }
    } else {
      node.flexibleMetrics.at(metricName).updateValue(metricValue);
      auto &flexibleMetric = node.flexibleMetrics.at(metricName);
      if (!flexibleMetric.isExclusive(0)) {
        inclusiveValueNamesCache.insert(flexibleMetric.getValueName(0));
      }
    }
  }
}

void TreeData::clear() {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto newTree = std::make_unique<Tree>();
  tree.swap(newTree);
  scopeIdToContextId.clear();
  inclusiveValueNamesCache.clear();
  deviceIdsCache.clear();
}

void TreeData::clearCache() {
  std::unique_lock<std::shared_mutex> lock(mutex);
  scopeIdToContextId.clear();
}

void TreeData::dumpHatchet(std::ostream &os) const {
  auto output = toJson();
  os << std::endl << output.dump(4) << std::endl;
}

json TreeData::toJson() const {
  std::shared_lock<std::shared_mutex> lock(mutex);
  auto startTime = std::chrono::steady_clock::now();
  auto ret = buildHatchetJson(tree.get());
  auto endTime = std::chrono::steady_clock::now();
  std::cout << "[PROTON] TreeData toJson took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                    endTime - startTime)
                    .count()
            << " ms" << std::endl;
  return ret;
}

std::vector<uint8_t> TreeData::toMsgPack() const {
  std::shared_lock<std::shared_mutex> lock(mutex);
  return buildHatchetMsgPack(tree.get());
}

std::string TreeData::toJsonString() const {
  auto output = toJson();
  return output.dump();
}

void TreeData::doDump(std::ostream &os, OutputFormat outputFormat) const {
  if (outputFormat == OutputFormat::Hatchet) {
    dumpHatchet(os);
  } else {
    std::logic_error("Output format not supported");
  }
}

TreeData::TreeData(const std::string &path, ContextSource *contextSource)
    : Data(path, contextSource) {
  tree = std::make_unique<Tree>();
}

TreeData::~TreeData() {}

} // namespace proton
