

/**
 * @file layer.hpp
 * @author Aska Lyn
 * @brief A C++ implementation of neural network layer base classes for deep
 * learning
 * @details This header defines a templated Layer class that serves as the base
 * class for all neural network layers in the DL4Cpp framework. It provides
 * common interfaces for forward propagation, weight management, and layer
 * configuration. The implementation supports float precision for deep learning
 * computations.
 * @date 2026-03-20
 */

#ifndef LAYER_HPP
#define LAYER_HPP

#include "runtime/rt_op.hpp"
#include "runtime/rt_type.hpp"
#include "tensor.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace ctl {

/**
 * @brief Primary template declaration for Layer class
 * @tparam T The data type used in the layer (default: float)
 * @details This is the primary template. Specializations are provided for
 *          int8_t and float types.
 */
template <typename T> class Layer;

/**
 * @brief Template specialization for int8_t layers
 * @details Placeholder specialization for 8-bit integer layers, typically
 *          used for quantized inference. Implementation pending.
 */
template <> class Layer<int8_t> {};

/**
 * @brief Template specialization for float layers - the main implementation
 * @details This is the base class for all neural network layers operating on
 *          32-bit floating-point data. It defines the common interface for
 *          layer operations including forward propagation, weight/bias
 *          management, and input/output validation. All concrete layer types
 *          (e.g., Convolution, FullyConnected, Activation) should inherit from
 *          this class.
 */
template <> class Layer<float> {

public:
  /**
   * @brief Construct a new Layer object
   * @param layer_name The name identifier for this layer
   * @details Initializes the layer with a unique name for identification
   *          during network construction and debugging.
   */
  explicit Layer(std::string layer_name) : layer_name_(std::move(layer_name)) {}

  /**
   * @brief Validate input and output tensors for the layer
   * @param inputs The input tensors to validate
   * @param outputs The output tensors to validate
   * @return StatusCode Status code indicating validation success or failure
   * @details Checks if the input and output tensor shapes, types, and counts
   *          are compatible with this layer's requirements. Should be
   *          overridden by derived classes to implement specific validation
   *          logic.
   */
  virtual StatusCode Check(const std::vector<sften> &inputs,
                           const std::vector<sften> &outputs);

  /**
   * @brief Perform forward propagation using internal inputs and outputs
   * @return StatusCode Status code indicating execution success or failure
   * @details Executes the layer's forward pass using internally stored
   *          input/output tensors. This is the primary interface for
   *          layer execution in a network graph.
   */
  virtual StatusCode Forward();

  /**
   * @brief Perform forward propagation with explicit input and output tensors
   * @param inputs The input tensors for forward propagation
   * @param outputs The output tensors to store results
   * @return StatusCode Status code indicating execution success or failure
   * @details Executes the layer's forward pass using the provided tensors.
   *          This overload allows explicit control over input/output buffers.
   */
  virtual StatusCode
  Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
          std::vector < std::shared_ptr<Tensor<float>>> &outputs);


  /**
   * @brief Get the layer's weight tensors
   * @return const std::vector<std::shared_ptr<Tensor<float>>>& Const reference
   * to weight tensors
   * @details Returns the learnable weight parameters of this layer.
   *          For convolutional layers, this typically contains the filter
   *          kernels. For fully connected layers, this contains the weight
   *          matrix.
   */
  virtual const std::vector<std::shared_ptr<Tensor<float>>>& weights() const;


  /**
   * @brief Get the layer's bias tensors
   * @return const std::vector<std::shared_ptr<Tensor<float>>>& Const reference
   * to bias tensors
   * @details Returns the bias parameters of this layer. Not all layers
   *          have biases (e.g., activation layers typically don't).
   */
  virtual const std::vector<std::shared_ptr<Tensor<float>>>& bias() const;


  /**
   * @brief Set the layer's weight tensors
   * @param weights The new weight tensors to set
   * @details Replaces the current weight parameters with the provided tensors.
   *          Used during model loading or weight initialization.
   */
  virtual void set_weight(const std::vector<std::shared_ptr<Tensor<float>>> &weights);


  /**
   * @brief Set the layer's bias tensors
   * @param bias The new bias tensors to set
   * @details Replaces the current bias parameters with the provided tensors.
   *          Used during model loading or bias initialization.
   */
  virtual void set_bias(const std::vector<std::shared_ptr<Tensor<float>>> &bias);


  /**
   * @brief Set the layer's weights from a flat float vector
   * @param weights The flat vector containing weight values
   * @details Convenience method to set weights from a flattened array.
   *          The values will be reshaped into the appropriate tensor
   *          format internally.
   * @note Typo in function name 'set_weught' is preserved for compatibility
   */
  virtual void set_weight(const std::vector<float>& weights);

  /**
   * @brief Set the layer's biases from a flat float vector
   * @param bias The flat vector containing bias values
   * @details Convenience method to set biases from a flattened array.
   *          The values will be reshaped into the appropriate tensor
   *          format internally.
   */
  virtual void set_bias(const std::vector<float>& bias);

  /**
   * @brief Get the layer's name
   * @return const std::string& The name of this layer
   * @details Returns the identifier name assigned during construction.
   *          Useful for debugging and network visualization.
   */
  virtual const std::string& layer_name() const{return this->layer_name_;}


  /**
   * @brief Set the runtime operator for this layer
   * @param runtime_operator The shared pointer to the runtime operator
   * @details Associates a runtime operator with this layer. The runtime
   *          operator provides execution context and resource management
   *          capabilities during inference or training.
   */
  void set_runtime_operator(const std::shared_ptr<RuntimeOperator>& runtime_operator);




protected:
  std::string layer_name_; ///< The name identifier of this layer
  std::weak_ptr<RuntimeOperator> runtime_operator_; ///< Weak reference to the
                                                    ///< associated runtime
                                                    ///< operator
};

} // namespace ctl

#endif