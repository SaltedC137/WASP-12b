
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

class Sigmoid {
public:
  void operator()(const sften &input, const sften &output) const;
};

class SigmoidLayer : public ActivationLayer {

public:
  explicit SigmoidLayer();

  StatusCode Forward(const std::vector<std::shared_ptr<ften>> &inputs,
                     std::vector<std::shared_ptr<ften>> &outputs) override;

  static StatusCode
  CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                 std::shared_ptr<Layer<float>> &sigmoid_layer);
};

/**
 * @brief Legacy factory function for dynamic activation selection
 * @param act_type The type of activation function to retrieve
 * @return ActivationFunc Function object for the specified activation type
 * @deprecated Use Sigmoid functor directly for better performance and
 * simplicity
 */
ActivationFunc ApplySSEActivation(ActivationType act_type);

} // namespace nn
} // namespace ctl

#endif