/**
 * @file relu.hpp
 * @author Aska Lyn
 * @brief ReLU activation function functor and layer implementation
 * @details SSE/AVX-optimized ReLU activation: @f$ f(x) = \max(0, x) @f$
 *          The ReLU (Rectified Linear Unit) is the most commonly used
 *          activation function in deep learning, providing non-linearity
 *          while avoiding vanishing gradient problems.
 * 
 *          This header provides two interfaces:
 *          1. ReLU functor for standalone usage: ReLU()(input, output)
 *          2. ReLULayer for graph execution via RuntimeGraph
 * @date 2026-04-03
 */

#ifndef RELU_HPP
#define RELU_HPP

#include "nn/ops/activation.hpp"
#include "rt_type.hpp"

namespace ctl {
namespace nn {

/**
 * @brief ReLU activation functor with SSE/AVX optimization
 * @details Computes element-wise rectified linear unit:
 *          @f[
 *          f(x) = \max(0, x) = \begin{cases} 
 *          x & \text{if } x > 0 \\
 *          0 & \text{otherwise}
 *          \end{cases}
 *          @f]
 *          Usage: ReLU()(input, output)
 */
class ReLU {
public:
  /**
   * @brief Apply ReLU activation element-wise
   * @param input  Input tensor
   * @param output Output tensor (must have same size as input)
   * @details Computes @f$ \text{output}_i = \max(0, \text{input}_i) @f$
   *          using SIMD instructions for accelerated computation.
   */
  void operator()(const sften &input, const sften &output) const;
};

/**
 * @brief ReLU activation layer for neural network graph execution
 * @details Wraps the ReLU functor for use with RuntimeGraph.
 *          Registered as "nn.ReLU" and "F.relu" for PNNX model loading.
 */
class ReLULayer : public nn::ActivationLayer {

public:
  /**
   * @brief Construct a new ReLU Layer object
   * @details Initializes with ActivationType::ActivationRelu
   */
  explicit ReLULayer();

  /**
   * @brief Perform ReLU forward propagation
   * @param inputs Input tensors for ReLU computation
   * @param outputs Output tensors to store ReLU results
   * @return StatusCode Status code indicating execution success or failure
   * @details Applies @f$ f(x) = \max(0, x) @f$ element-wise
   */
  StatusCode
  Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
          std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  /**
   * @brief Create a ReLULayer instance from a runtime operator
   * @param op Shared pointer to the runtime operator
   * @param relu_layer Output parameter to receive the created layer
   * @return StatusCode Status code indicating creation success or failure
   */
  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                                   std::shared_ptr<Layer<float>> &relu_layer);
};

/**
 * @brief Legacy factory function for dynamic activation selection
 * @param act_type The type of activation function to retrieve
 * @return ActivationFunc Function object for the specified activation type
 * @deprecated Use ReLU functor directly for better performance and simplicity
 */
ActivationFunc ApplySSEActivation(ActivationType act_type);

} // namespace nn
} // namespace ctl

#endif