/**
 * @file relu.hpp
 * @author Aska Lyn
 * @brief ReLU activation function functor
 * @details SSE/AVX-optimized ReLU activation: f(x) = max(0, x)
 *          The ReLU (Rectified Linear Unit) is the most commonly used
 *          activation function in deep learning, providing non-linearity
 *          while avoiding vanishing gradient problems.
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
 * @details Computes element-wise: f(x) = max(0, x)
 *          Usage: ReLU()(input, output)
 */
class ReLU {
public:
  /**
   * @brief Apply ReLU activation element-wise
   * @param input  Input tensor
   * @param output Output tensor (must have same size as input)
   */
  void operator()(const sften &input, const sften &output) const;
};

class ReLULayer : public nn::ActivationLayer {

public:
  explicit ReLULayer();

  StatusCode
  Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
          std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

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