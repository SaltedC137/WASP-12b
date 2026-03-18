/**
 * @file tensor_linalg.hpp
 * @author Aska Lyn
 * @brief Linear algebra operations for tensor objects in deep learning
 * @details This header provides linear algebra operations including norms,
 * inner/outer products, matrix transformations (transpose, inverse), and
 * other matrix properties (determinant, trace). All operations support both
 * in-place modification (void functions with output parameter) and functional
 * style (inline functions returning shared_ptr).
 * @date 2026-03-12
 */

#ifndef TENSOR_LINALG_HPP
#define TENSOR_LINALG_HPP

#include <armadillo>
#pragma once

#include "core/tensor.hpp"

namespace ctl {

/**
 * @brief Compute the Euclidean norm (L2 norm) of a tensor
 * @param tensor The input tensor
 * @return float The Euclidean norm value
 * @details Computes ||tensor||_2 = sqrt(sum(x_i^2)), which is the standard
 *          Euclidean length of the tensor when viewed as a vector.
 */
float Euclidean_norm(const ften &tensor);

/**
 * @brief Compute the L1 norm (absolute value norm) of a tensor
 * @param tensor The input tensor
 * @return float The L1 norm value
 * @details Computes ||tensor||_1 = sum(|x_i|), which is the sum of absolute
 *          values of all elements in the tensor.
 */
float Absolute_value_norm(const ften &tensor);

/**
 * @brief Compute the inner product (dot product) of two tensors
 * @param tensor1 The first input tensor
 * @param tensor2 The second input tensor
 * @return float The inner product value
 * @details Computes the sum of element-wise products: sum(x_i * y_i).
 *          Both input tensors must have the same shape.
 */
float Transvection(const ften &tensor1, const ften &tensor2);

/**
 * @brief Compute the batch determinant of square matrices
 * @param tensor The input tensor (must be square matrices with multiple channels)
 * @param output Reference to the output tensor storing determinant values for each channel
 * @details Computes det(tensor) for each channel in parallel. The input tensor
 *          must have shape (rows, rows, channels) where rows == cols for each slice.
 *          Output tensor should have size equal to the number of channels.
 *          Uses OpenMP for parallel processing when channels > 1.
 * @warning The input tensor must represent square matrices (rows == cols).
 */
void Batch_Determinant(const ften &tensor, ften &output);

/**
 * @brief Compute the trace of square matrices in batch
 * @param tensor The input tensor (must be square matrices with multiple channels)
 * @param output Reference to the output tensor storing trace values for each channel
 * @details Computes trace(tensor) for each channel in parallel. The input tensor
 *          must have shape (rows, rows, channels) where rows == cols for each slice.
 *          Output tensor should have size equal to the number of channels.
 *          Uses OpenMP for parallel processing when channels > 1.
 * @warning The input tensor must represent square matrices (rows == cols).
 */
void Batch_Trace(const ften &tensor, ften &output);

/**
 * @brief Compute the outer product of two tensors
 * @param tensor1 The first input tensor
 * @param tensor2 The second input tensor
 * @param output Reference to the output tensor storing the result
 * @details Computes the outer product, producing a matrix where
 *          output[i,j] = tensor1[i] * tensor2[j].
 */
void Outer_product(const ften &tensor1, const ften &tensor2, ften &output);

/**
 * @brief Compute the inverse of square matrices in batch
 * @param tensor The input tensor (must be invertible square matrices with multiple channels)
 * @param output Reference to the output tensor storing inverse matrices for each channel
 * @details Computes tensor^(-1) for each channel in parallel such that tensor · tensor^(-1) = I.
 *          The input tensor must have shape (rows, rows, channels) where rows == cols for each slice.
 *          Output tensor should have the same shape as the input tensor.
 *          Uses OpenMP for parallel processing when channels > 1.
 * @warning The input tensor must be non-singular square matrices.
 */
void Batch_Inverse(const ften &tensor, ften &output);

/**
 * @brief Compute the transpose of matrices in batch
 * @param tensor The input tensor
 * @param output Reference to the output tensor storing the transposed result
 * @details Computes tensor^T for each channel in parallel, swapping rows and columns.
 *          For a matrix A, output[i,j] = A[j,i]. The output tensor must have
 *          shape (cols, rows, channels) if input is (rows, cols, channels).
 *          Uses OpenMP for parallel processing when channels > 1.
 */
void Transposition(const ften &tensor, ften &output);

// ==================== Functional Interface ====================
// Inline functions that return shared_ptr or direct values for convenient chaining

/**
 * @brief Compute the Euclidean norm of a tensor
 * @param tensor The input tensor
 * @return float The Euclidean norm value
 * @details Convenience wrapper for Euclidean_norm().
 */
inline float norm(const ften &tensor) { return Euclidean_norm(tensor); }

/**
 * @brief Compute the L1 norm of a tensor
 * @param tensor The input tensor
 * @return float The L1 norm value
 * @details Convenience wrapper for Absolute_value_norm().
 */
inline float norm1(const ften &tensor) { return Absolute_value_norm(tensor); }

/**
 * @brief Compute the inner product of two tensors
 * @param tensor1 The first input tensor
 * @param tensor2 The second input tensor
 * @return float The inner product value
 * @details Convenience wrapper for Transvection().
 */
inline float dot(const ften &tensor1, const ften &tensor2) {
  return Transvection(tensor1, tensor2);
}

/**
 * @brief Compute the batch determinant of square matrices
 * @param tensor The input tensor (must be square matrices with multiple channels)
 * @return sften Shared pointer to a tensor containing determinant values for each channel
 * @details Convenience wrapper for Batch_Determinant(). Creates an output tensor
 *          with shape (channels, 1, 1) and computes determinants for all channels.
 */
inline sften det(const ften &tensor) {
  sften output = std::make_shared<ften>(tensor.channels(), 1, 1);
  Batch_Determinant(tensor, *output);
  return output;
}

/**
 * @brief Compute the batch trace of square matrices
 * @param tensor The input tensor (must be square matrices with multiple channels)
 * @return sften Shared pointer to a tensor containing trace values for each channel
 * @details Convenience wrapper for Batch_Trace(). Creates an output tensor
 *          with shape (channels, 1, 1) and computes traces for all channels.
 */
inline sften trace(const ften &tensor) {
  sften output = std::make_shared<ften>(tensor.channels(), 1, 1);
  Batch_Trace(tensor, *output);
  return output;
}

/**
 * @brief Compute the outer product returning a new tensor
 * @param tensor1 The first input tensor
 * @param tensor2 The second input tensor
 * @return sften Shared pointer to the result tensor
 * @details Creates a new tensor and computes the outer product.
 *          Convenient for expression chaining without pre-allocating output.
 */
inline sften outer(const ften &tensor1, const ften &tensor2) {
  sften output = std::make_shared<ften>(1, tensor1.size(), tensor2.size());
  Outer_product(tensor1, tensor2, *output);
  return output;
}

/**
 * @brief Compute the batch transpose of matrices
 * @param tensor The input tensor
 * @return sften Shared pointer to the result tensor containing transposed matrices for each channel
 * @details Convenience wrapper for Transposition(). Creates a new tensor with transposed dimensions
 *          (cols, rows, channels) and computes the transpose for all channels.
 */
inline sften transpose(const ften &tensor) {
  sften output =
      std::make_shared<ften>(tensor.channels(), tensor.cols(), tensor.rows());
  Transposition(tensor, *output);
  return output;
}

/**
 * @brief Compute the batch inverse of square matrices
 * @param tensor The input tensor (must be invertible square matrices with multiple channels)
 * @return sften Shared pointer to the result tensor containing inverse matrices for each channel
 * @details Convenience wrapper for Batch_Inverse(). Creates a new tensor with the same shape
 *          as the input and computes the matrix inverse for all channels.
 *          @warning The input tensor must be non-singular square matrices.
 */
inline sften inv(const ften &tensor) {
  sften output = std::make_shared<ften>(tensor.shapes());
  Batch_Inverse(tensor, *output);
  return output;
}

} // namespace ctl::linalg

#endif