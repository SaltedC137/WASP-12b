

/**
 * @file param_layer.hpp
 * @author Aska Lyn
 * @brief Parameter layer base class for neural network layers with learnable weights
 * @details This header defines the ParamLayer class, which serves as the base class
 * for layers that contain learnable parameters (weights and biases). Layers such as
 * Convolution, FullyConnected (Linear), and BatchNormalization inherit from this class.
 * The class provides interfaces for initializing, accessing, and modifying weight and
 * bias tensors.
 * @date 2026-03-20
 */

#ifndef PARAM_LAYER_HPP
#define PARAM_LAYER_HPP

#include "nn/layer.hpp"
#include "tensor.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ctl {

/**
 * @brief Base class for neural network layers with learnable parameters
 * @details ParamLayer extends the base Layer class by adding weight and bias
 * tensor management. It provides methods for initializing parameter storage
 * with specific shapes, accessing weight/bias tensors, and setting parameters
 * from flat vectors or tensor collections.
 * 
 * Derived classes include:
 * - Convolutional layers (weights: [out_channels, in_channels, kH, kW])
 * - Fully connected layers (weights: [out_features, in_features])
 * - Normalization layers (weights: [channels], bias: [channels])
 */
class ParamLayer : public Layer<float> {
public:
  /**
   * @brief Construct a new Param Layer object
   * @param layer_name The name identifier for this layer
   * @details Initializes an empty parameter layer. Weights and biases
   *          must be initialized via InitWeightParam/InitBiasParam or
   *          set_weight/set_bias methods before use.
   */
  explicit ParamLayer(const std::string &layer_name);

  /**
   * @brief Initialize weight parameter storage
   * @param param_count Number of parameter tensors (e.g., output channels)
   * @param param_channel Number of channels per parameter tensor
   * @param param_height Height dimension of each parameter tensor
   * @param param_width Width dimension of each parameter tensor
   * @details Allocates weight tensors with shape (param_count, param_channel,
   *          param_height, param_width). For convolutional layers, this typically
   *          corresponds to (out_channels, in_channels, kernel_h, kernel_w).
   */
  void InitWeightParam(uint32_t param_count, uint32_t param_channel,
                       uint32_t param_height, uint32_t param_width);

  /**
   * @brief Initialize bias parameter storage
   * @param param_count Number of bias parameters (e.g., output channels)
   * @param param_channel Number of channels per bias tensor
   * @param param_height Height dimension (usually 1 for biases)
   * @param param_width Width dimension (usually 1 for biases)
   * @details Allocates bias tensors. For most layers, biases are 1D vectors
   *          with shape (param_count, 1, 1).
   */
  void InitBiasParam(uint32_t param_count, uint32_t param_channel,
                     uint32_t param_height, uint32_t param_width);

  /**
   * @brief Get the weight tensors
   * @return const std::vector<std::shared_ptr<Tensor<float>>>& Const reference to weight tensors
   * @details Returns the learnable weight parameters. The number and shape of
   *          tensors depend on the layer type and initialization.
   */
  const std::vector<std::shared_ptr<Tensor<float>>> &weights() const override;

  /**
   * @brief Get the bias tensors
   * @return const std::vector<std::shared_ptr<Tensor<float>>>& Const reference to bias tensors
   * @details Returns the bias parameters. Not all layers have biases;
   *          some layers (e.g., activation layers) may return an empty vector.
   */
  const std::vector<std::shared_ptr<Tensor<float>>> &bias() const override;

  /**
   * @brief Set weights from a flat float vector
   * @param weights Flat vector containing weight values
   * @details Reshapes the flat vector into the appropriate tensor structure.
   *          The total size must match the initialized weight capacity.
   */
  void set_weight(const std::vector<float> &weights) override;

  /**
   * @brief Set biases from a flat float vector
   * @param bias Flat vector containing bias values
   * @details Reshapes the flat vector into bias tensors.
   *          The total size must match the initialized bias capacity.
   */
  void set_bias(const std::vector<float> &bias) override;

  /**
   * @brief Set weight tensors directly
   * @param weights Vector of weight tensors
   * @details Replaces the current weight tensors with the provided ones.
   *          Used during model loading from file.
   */
  void set_weight(
      const std::vector<std::shared_ptr<Tensor<float>>> &weights) override;

  /**
   * @brief Set bias tensors directly
   * @param biax Vector of bias tensors
   * @details Replaces the current bias tensors with the provided ones.
   *          Used during model loading from file.
   */
  void
  set_bias(const std::vector<std::shared_ptr<Tensor<float>>> &biax) override;

protected:
  std::vector<std::shared_ptr<Tensor<float>>> weights_; ///< Weight parameter tensors
  std::vector<std::shared_ptr<Tensor<float>>> bias_;    ///< Bias parameter tensors
};

/**
 * @brief Type alias for layers without learnable parameters
 * @details NoneParamLayer is simply Layer<float>, used for layers that do
 * not have weights or biases, such as activation functions, pooling, and
 * element-wise operations.
 */
using NoneParamLayer = Layer<float>;

} // namespace ctl

#endif