

#include "nn/layer_factory.hpp"
#include "check.hpp"
#include "log.hpp"
#include "runtime/rt_op.hpp"
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>


namespace ctl {

LayerRegister::CreateRegistry *LayerRegister::registry_ = nullptr;

void LayerRegister::RegisterCreator(const std::string &layer_type,
                                    const Creator &creator) {
  CHECK(!layer_type.empty()) << "Layer type cannot be empty";
  CHECK(creator != nullptr);
  CreateRegistry *registry = Registry();
  CHECK_EQ(registry->count(layer_type), 0)
      << "Layer type:" << layer_type << "has already registered";
  registry->insert({layer_type, creator});
}

LayerRegister::CreateRegistry *LayerRegister::Registry() {
  if (registry_ == nullptr) {
    registry_ = new CreateRegistry();
    static RegisterGarbageCollector c;
  }
  CHECK(registry_ != nullptr);
  return registry_;
}

std::shared_ptr<Layer<float>>
LayerRegister::CreateLayer(const std::shared_ptr<RuntimeOperator> &op) {
  CreateRegistry *registry = Registry();
  const std::string &layer_type = op->type;
  LOG_IF(FATAL, registry->count(layer_type) <= 0)
      << "Can't find layer type:" << layer_type;
  const auto &creator = registry->find(layer_type)->second;

  LOG_IF(FATAL, !creator) << "Layer creator is null";
  std::shared_ptr<Layer<float>> layer;
  const auto &status = creator(op, layer);
  LOG_IF(FATAL, status != StatusCode::Success)
      << "Create Layer:" << layer_type
      << " failed, error code:" << int32_t(status);
  return layer;
}

std::vector<std::string> LayerRegister::layer_types() {
  std::set<std::string> layer_type_unique;
  CreateRegistry *registry = Registry();
  for (const auto &[layer_type, _] : *registry) {
    layer_type_unique.insert(layer_type);
  }
  std::vector<std::string> layer_types(layer_type_unique.begin(),
                                       layer_type_unique.end());
  return layer_types;
}

} // namespace ctl