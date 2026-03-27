
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

namespace ctl {
namespace nn {

/**
 * @brief Get the SSE-optimized activation function for a given type
 * @param act_type The type of activation function to retrieve
 * @return ActivationFunc Function object for the specified activation type
 * @details Returns a std::function that wraps the SSE-optimized implementation
 * of the specified activation function. Currently supports Sigmoid activation.
 * The returned function can be used to apply the activation element-wise to
 * input tensors with SIMD acceleration for improved performance.
 * @note For Sigmoid, the function computes: @f$ \sigma(x) = \frac{1}{1 + e^{-x}} @f$
 */
ActivationFunc ApplySSEActivation(ActivationType act_type);

} // namespace nn
} // namespace ctl

#endif