
/**
 * @file sigmoid.hpp
 * @author Aska Lyn
 * @brief Sigmoid activation function utilities
 * @details This header provides sigmoid activation function support using
 * SSE (Streaming SIMD Extensions) for accelerated computation. The sigmoid
 * function is a fundamental activation function that maps inputs to the
 * range (0, 1), commonly used in binary classification and gating mechanisms.
 * @date 2026-03-20
 */

#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "activation.hpp"
#include "rt_type.hpp"
#include "tensor.hpp"
#include <memory>
#include <vector>

namespace ctl {
namespace nn {

/**
 * @brief Sigmoid activation functor with SSE/AVX optimization
 * @details Computes element-wise sigmoid activation function:
 *          @f[
 *          \sigma(x) = \frac{1}{1 + e^{-x}}
 *          @f]
 *          The sigmoid function maps inputs to the range (0, 1),
 *          commonly used in binary classification and gating mechanisms.
 *          Usage: Sigmoid()(input, output)
 */
class Sigmoid {
public:
  /**
   * @brief Apply sigmoid activation element-wise
   * @param input  Input tensor
   * @param output Output tensor (must have same size as input)
   * @details Computes @f$ \text{output}_i = \frac{1}{1 + e^{-\text{input}_i}} @f$
   *          using SIMD instructions for accelerated computation.
   */
  void operator()(const sften &input, sften &output) const;
};

/**
 * @brief Sigmoid activation layer for neural network graph execution
 * @details Wraps the sigmoid functor for use with RuntimeGraph.
 *          The sigmoid function is defined as:
 *          @f[
 *          \sigma(x) = \frac{1}{1 + e^{-x}}
 *          @f]
 *          Registered as "nn.Sigmoid" for PNNX model loading.
 */
class SigmoidLayer : public ActivationLayer {

public:
  /**
   * @brief Construct a new Sigmoid Layer object
   * @details Initializes with ActivationType::ActivationSigmoid
   */
  explicit SigmoidLayer();

  /**
   * @brief Perform sigmoid forward propagation
   * @param inputs Input tensors for sigmoid computation
   * @param outputs Output tensors to store sigmoid results
   * @return StatusCode Status code indicating execution success or failure
   * @details Applies @f$ \sigma(x) = \frac{1}{1 + e^{-x}} @f$ element-wise
   */
  StatusCode Forward(const std::vector<std::shared_ptr<ften>> &inputs,
                     std::vector<std::shared_ptr<ften>> &outputs) override;

  /**
   * @brief Create a SigmoidLayer instance from a runtime operator
   * @param op Shared pointer to the runtime operator
   * @param sigmoid_layer Output parameter to receive the created layer
   * @return StatusCode Status code indicating creation success or failure
   */
  static StatusCode
  CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                 std::shared_ptr<Layer<float>> &sigmoid_layer);
};

} // namespace nn
} // namespace ctl

#endif