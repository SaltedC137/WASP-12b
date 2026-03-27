

/**
 * @file softmax.hpp
 * @author Aska Lyn
 * @brief Softmax activation layer implementation
 * @details This header defines the SoftmaxLayer class that applies the softmax
 * activation function along a specified dimension. The softmax function
 * normalizes the input values to a probability distribution where all outputs
 * are in the range [0, 1] and sum to 1. Commonly used in the final layer of
 * classification networks.
 * @date 2026-03-20
 */

#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "param_layer.hpp"
#include "rt_type.hpp"
#include "tensor.hpp"
#include <cstdint>

namespace ctl {
namespace nn {

/**
 * @brief Softmax activation layer for neural networks
 * @details Applies the softmax function to the input tensor along a specified
 * dimension. The softmax function is defined as:
 * @f[
 * \text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
 * @f]
 * For numerical stability, the implementation subtracts the maximum value
 * before computing the exponential.
 */
class SoftmaxLayer : public NoneParamLayer {
public:
  /**
   * @brief Construct a new Softmax Layer object
   * @param dim The dimension along which to apply softmax (default: -1)
   * @details If dim is negative, it is interpreted as dim + ndim.
   * For example, dim=-1 applies softmax along the last dimension.
   */
  explicit SoftmaxLayer(int32_t dim = -1);

  /**
   * @brief Perform softmax forward propagation
   * @param inputs Input tensors for softmax computation
   * @param outputs Output tensors to store softmax results
   * @return StatusCode Status code indicating execution success or failure
   * @details Applies the softmax function element-wise to each input tensor.
   * The input and output tensors must have matching shapes. If an output
   * tensor is null or empty, a new tensor with appropriate shape will be
   * allocated.
   */
  StatusCode Forward(const std::vector<std::shared_ptr<ften>> &inputs,
                     std::vector<std::shared_ptr<ften>> outputs) override;

  /**
   * @brief Create a SoftmaxLayer instance from a runtime operator
   * @param op Shared pointer to the runtime operator containing parameters
   * @param softmax_layer Output parameter to receive the created layer
   * @return StatusCode Status code indicating creation success or failure
   * @details Factory method that extracts the softmax dimension from the
   * operator's parameters and constructs a SoftmaxLayer instance. Used
   * during computation graph initialization.
   */
  static StatusCode
  CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                 std::shared_ptr<Layer<float>> &softmax_layer);

private:
  int32_t softmax_dim_ = -1; ///< Dimension along which softmax is applied
};

} // namespace nn
} // namespace ctl

#endif