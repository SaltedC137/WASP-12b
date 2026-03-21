

/**
 * @file layer_factory.hpp
 * @author Aska Lyn
 * @brief A C++ implementation of layer registration and factory pattern for neural network layers
 * @details This header provides a registry mechanism for neural network layer types in the DL4Cpp
 * framework. It uses the factory pattern to create layer instances based on string identifiers,
 * enabling dynamic layer creation during model loading and network construction.
 * @date 2026-03-20
 */

#ifndef LAYER_FACTORY_HPP
#define LAYER_FACTORY_HPP

#include "runtime/rt_op.hpp"
#include "runtime/rt_type.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ctl {

/**
 * @brief Registry and factory class for creating neural network layer instances
 * @details LayerRegister maintains a global registry of layer types and their corresponding
 * creator functions. It provides a centralized mechanism for registering custom layer types
 * and creating layer instances by type name. This class is typically used during model
 * initialization and layer deserialization.
 *
 * The registry is implemented as a singleton pattern with thread-unsafe static methods.
 * All layer types must be registered before calling CreateLayer().
 */
class LayerRegister {

private:
  /**
   * @brief Creator function type for layer instantiation
   * @param op The runtime operator containing layer configuration
   * @param layer Output parameter to store the created layer instance
   * @return StatusCode indicating creation success or failure
   * @details Creator functions are responsible for constructing layer instances
   *          and configuring them based on the provided runtime operator.
   */
  typedef StatusCode (*Creator)(const std::shared_ptr<RuntimeOperator> &op,
                                std::shared_ptr<Layer<float>> &layer);

  /**
   * @brief Registry map type storing layer type names and their creators
   * @details Maps string identifiers (e.g., "Conv2D", "ReLU") to creator functions.
   */
  typedef std::map<std::string, Creator> CreateRegistry;

public:
  friend class LayerRegisterWrapper;
  friend class RegisterGarbageCollector;

  /**
   * @brief Register a creator function for a specific layer type
   * @param layer_type The string identifier for the layer type (e.g., "Conv2D")
   * @param creator The creator function responsible for instantiating the layer
   * @details Associates a layer type name with its creator function in the global registry.
   *          If the layer type is already registered, the creator will be overwritten.
   * @note This function is not thread-safe. All registrations should complete before
   *       calling CreateLayer() from multiple threads.
   */
  static void RegisterCreator(const std::string &layer_type,
                              const Creator &creator);

  /**
   * @brief Create a layer instance based on the operator's type
   * @param op The runtime operator containing layer type and configuration
   * @return std::shared_ptr<Layer<float>> The created layer instance, or nullptr on failure
   * @details Looks up the creator function for the layer type specified in the operator
   *          and invokes it to construct the layer. Returns nullptr if the layer type
   *          is not registered or if creation fails.
   */
  static std::shared_ptr<Layer<float>>
  CreateLayer(const std::shared_ptr<RuntimeOperator> &op);

  /**
   * @brief Get a pointer to the internal registry
   * @return CreateRegistry* Pointer to the registry map
   * @details Provides direct access to the registry for advanced use cases.
   *          Use with caution as modifications may affect layer creation.
   */
  static CreateRegistry *Registry();

  /**
   * @brief Get all registered layer type names
   * @return std::vector<std::string> A vector containing all registered layer type names
   * @details Returns a copy of all layer type identifiers currently in the registry.
   *          Useful for debugging and validation purposes.
   */
  static std::vector<std::string> layer_types();

private:
  static CreateRegistry *registry_; ///< Pointer to the singleton registry instance
};

/**
 * @brief Helper class for automatic layer registration
 * @details LayerRegisterWrapper provides a convenient RAII-style interface for
 *          registering layer types. It is typically used in static initialization
 *          contexts to ensure layer types are registered before main() execution.
 *          The variadic template constructor allows registering multiple layer types
 *          with a single creator function.
 */
class LayerRegisterWrapper {
public:
  /**
   * @brief Register a single layer type with its creator
   * @param creator The creator function for the layer type
   * @param layer_type The string identifier for the layer type
   */
  explicit LayerRegisterWrapper(const LayerRegister::Creator &creator,
                                const std::string &layer_type) {
    LayerRegister::RegisterCreator(layer_type, creator);
  }

  /**
   * @brief Register multiple layer types with the same creator
   * @tparam Ts Variadic template parameter for additional layer type strings
   * @param creator The creator function shared by all layer types
   * @param layer_type The first layer type to register
   * @param other_layer_types Additional layer types to register with the same creator
   * @details This variadic template allows registering aliases or multiple names
   *          for the same layer implementation (e.g., "Relu" and "ReLU").
   */
  template <typename... Ts>
  explicit LayerRegisterWrapper(const LayerRegister::Creator &creator,
                                const std::string &layer_type,
                                const Ts &...other_layer_types)
      : LayerRegisterWrapper(creator, other_layer_types...) {
    LayerRegister::RegisterCreator(layer_type, creator);
  }

  /**
   * @brief Empty constructor for creator-only registration
   * @param creator The creator function (unused in this overload)
   * @details This overload exists for template metaprogramming compatibility.
   *          It accepts a creator but does not register any layer type.
   */
  explicit LayerRegisterWrapper(const LayerRegister::Creator &creator) {}
};

/**
 * @brief Singleton garbage collector for registry cleanup
 * @details RegisterGarbageCollector manages the lifetime of the static registry.
 *          It ensures proper cleanup of the registry singleton during program
 *          termination. The class follows the singleton pattern with controlled
 *          destruction order.
 */
class RegisterGarbageCollector {
public:
  /**
   * @brief Destructor that cleans up the registry
   * @details Deletes the registry singleton and resets the pointer to nullptr.
   *          Called automatically during program termination to prevent memory leaks.
   */
  ~RegisterGarbageCollector() {
    if (LayerRegister::registry_ != nullptr) {
      delete LayerRegister::registry_;
      LayerRegister::registry_ = nullptr;
    }
  }
  friend class LayerRegister;

private:
  RegisterGarbageCollector() = default; ///< Private default constructor
  RegisterGarbageCollector(const RegisterGarbageCollector &) = default; ///< Private copy constructor
};

} // namespace ctl

#endif
