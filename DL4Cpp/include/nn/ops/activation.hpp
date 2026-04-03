

/**
 * @file activation.hpp
 * @author Aska Lyn
 * @brief Activation layer implementation for neural networks
 * @details This header defines the ActivationLayer class and related utilities
 * for applying various activation functions in neural networks. Activation
 * functions introduce non-linearity into the network, enabling it to learn
 * complex patterns. Supported activation types include ReLU, SiLU, Sigmoid,
 * Hard Swish, Hard Sigmoid, and ReLU6.
 * @date 2026-03-20
 */

#ifndef _NN_OPS_ACTIVATION_HPP_
#define _NN_OPS_ACTIVATION_HPP_

#include "core/tensor.hpp"
#include "nn/param_layer.hpp"
#include "runtime/rt_type.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ctl {
namespace nn {

/**
 * @brief Type alias for activation function signature
 * @details Represents a generic activation function that takes an input tensor
 * and an output tensor. The function applies the activation element-wise.
 */
using ActivationFunc = std::function<void(sften, sften)>;

/**
 * @brief Enumeration of supported activation function types
 * @details Each enum value corresponds to a specific activation function
 * implementation. ActivationType_None indicates no activation function.
 * 
 * Mathematical definitions:
 * - ReLU: @f$ f(x) = \max(0, x) @f$
 * - SiLU: @f$ f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} @f$
 * - Sigmoid: @f$ \sigma(x) = \frac{1}{1 + e^{-x}} @f$
 * - Hard Swish: @f$ f(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6} @f$
 * - Hard Sigmoid: @f$ f(x) = \frac{\text{ReLU6}(x + 3)}{6} @f$
 * - ReLU6: @f$ f(x) = \min(\max(0, x), 6) @f$
 */
enum ActivationType {
  ActivationType_None = -1, ///< No activation function applied
  ActivationRelu = 0,       ///< ReLU: @f$ f(x) = \max(0, x) @f$
  ActivationSilu = 1,       ///< SiLU/Swish: @f$ f(x) = x \cdot \sigma(x) @f$
  ActivationSigmoid = 2,    ///< Sigmoid: @f$ \sigma(x) = \frac{1}{1 + e^{-x}} @f$
  ActivationHardSwish = 3,  ///< Hard Swish: @f$ f(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6} @f$
  ActivationHardSigmoid = 4, ///< Hard Sigmoid: @f$ f(x) = \frac{\text{ReLU6}(x + 3)}{6} @f$
  ActivationRelu6 = 5,      ///< ReLU6: @f$ f(x) = \min(\max(0, x), 6) @f$
};

/**
 * @brief Convert an activation type enum to its string representation
 * @param type The activation type to convert
 * @return std::string String representation of the activation type
 * @details Returns a human-readable string name for the given activation type.
 * Useful for debugging, logging, and serialization purposes.
 */
std::string ActivationTypeToString(ActivationType type);

/**
 * @brief Activation layer for applying non-linear functions
 * @details Applies element-wise activation functions to input tensors.
 * The layer supports multiple activation types selected at construction time.
 * Activation functions are implemented using SSE instructions for efficient
 * vectorized computation on supported hardware.
 */
class ActivationLayer : public NoneParamLayer {
public:
  /**
   * @brief Construct a new Activation Layer object
   * @param type The type of activation function to apply
   * @param layer_name Name identifier for this layer
   * @details Initializes the layer with the specified activation function
   * type and assigns a unique name for identification in the computation
   * graph.
   */
  explicit ActivationLayer(nn::ActivationType type, std::string layer_name);

  /**
   * @brief Validate input and output tensors for activation layer
   * @param inputs The input tensors to validate
   * @param outputs The output tensors to validate
   * @return StatusCode Status code indicating validation success or failure
   * @details Checks that input and output tensors have compatible shapes
   * and data types for activation function application.
   */
  StatusCode Check(const std::vector<sften> &inputs,
                   const std::vector<sften> &outputs) override;

  /**
   * @brief Perform activation forward propagation
   * @param inputs Input tensors for activation computation
   * @param outputs Output tensors to store activation results
   * @return StatusCode Status code indicating execution success or failure
   * @details Applies the configured activation function element-wise to
   * each input tensor. The activation is computed using SSE-optimized
   * implementations when available.
   */
  StatusCode Forward(const std::vector<std::shared_ptr<ften>> &inputs,
                     std::vector<std::shared_ptr<ften>> &outputs) override;

private:
  ActivationType activation_type_ = ActivationType::ActivationType_None; ///< Type of activation function
};

} // namespace nn
} // namespace ctl
#endif