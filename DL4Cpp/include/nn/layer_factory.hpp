

#ifndef LAYER_FACTORY_HPP
#define LAYER_FACTORY_HPP

#include "runtime/rt_op.hpp"
#include "runtime/rt_type.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace ctl {

class LayerRegister {

private:
  typedef StatusCode (*Creator)(const std::shared_ptr<RuntimeOperator> &op,
                                std::shared_ptr<Layer<float>> &layer);

  typedef std::map<std::string, Creator> CreateRegistry;

public:
  friend class LayerRegisterWrapper;
  friend class RegisterGarbageCollector;

  static void RegisterCreator(const std::string &layer_type,
                              const Creator &creator);

  static std::shared_ptr<Layer<float>>
  CreateLayer(const std::shared_ptr<RuntimeOperator> &op);

  static CreateRegistry *Registry();

  static std::vector<std::string> layer_types();

private:
  static CreateRegistry *registry_;
};

class LayerRegisterWrapper {
public:
  explicit LayerRegisterWrapper(const LayerRegister::Creator &creator,
                                const std::string &layer_type) {
    LayerRegister::RegisterCreator(layer_type, creator);
  }

  template <typename... Ts>
  explicit LayerRegisterWrapper(const LayerRegister::Creator &creator,
                                const std::string &layer_type,
                                const Ts &...other_layer_types)
      : LayerRegisterWrapper(creator, other_layer_types...) {
    LayerRegister::RegisterCreator(layer_type, creator);
  }

  explicit LayerRegisterWrapper(const LayerRegister::Creator &creator) {}
};

class RegisterGarbageCollector {
public:
  ~RegisterGarbageCollector() {
    if (LayerRegister::registry_ != nullptr) {
      delete LayerRegister::registry_;
      LayerRegister::registry_ = nullptr;
    }
  }
  friend class LayerRegister;

private:
  RegisterGarbageCollector() = default;
  RegisterGarbageCollector(const RegisterGarbageCollector &) = default;
};

} // namespace ctl

#endif
