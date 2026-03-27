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
 * @details Computes the L2 norm (Euclidean length):
 * @f[
 * \|\text{tensor}\|_2 = \sqrt{\sum_i x_i^2}
 * @f]
 * This is the standard Euclidean distance from the origin.
 */
float Euclidean_norm(const ften &tensor);

/**
 * @brief Compute the L1 norm (absolute value norm) of a tensor
 * @param tensor The input tensor
 * @return float The L1 norm value
 * @details Computes the L1 norm (sum of absolute values):
 * @f[
 * \|\text{tensor}\|_1 = \sum_i |x_i|
 * @f]
 * Also known as the Manhattan norm or taxicab norm.
 */
float Absolute_value_norm(const ften &tensor);

/**
 * @brief Compute the inner product (dot product) of two tensors
 * @param tensor1 The first input tensor
 * @param tensor2 The second input tensor
 * @return float The inner product value
 * @details Computes the inner (dot) product:
 * @f[
 * \text{tensor1} \cdot \text{tensor2} = \sum_i \text{tensor1}_i \times \text{tensor2}_i
 * @f]
 * Both input tensors must have the same shape.
 */
float Transvection(const ften &tensor1, const ften &tensor2);

/**
 * @brief Compute the batch determinant of square matrices
 * @param tensor The input tensor (must be square matrices with multiple channels)
 * @param output Reference to the output tensor storing determinant values for each channel
 * @details Computes the determinant for each channel:
 * @f[
 * \text{output}_k = \det(A_k)
 * @f]
 * where @f$ A_k @f$ is the k-th channel (square matrix).
 * For a 2×2 matrix:
 * @f[
 * \det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc
 * @f]
 * Uses OpenMP for parallel processing when channels > 1.
 * @warning The input tensor must represent square matrices (rows == cols).
 */
void Batch_Determinant(const ften &tensor, ften &output);

/**
 * @brief Compute the batch trace of square matrices in batch
 * @param tensor The input tensor (must be square matrices with multiple channels)
 * @param output Reference to the output tensor storing trace values for each channel
 * @details Computes the trace (sum of diagonal elements) for each channel:
 * @f[
 * \text{output}_k = \text{tr}(A_k) = \sum_i (A_k)_{ii}
 * @f]
 * where @f$ A_k @f$ is the k-th channel (square matrix).
 * Uses OpenMP for parallel processing when channels > 1.
 * @warning The input tensor must represent square matrices (rows == cols).
 */
void Batch_Trace(const ften &tensor, ften &output);

/**
 * @brief Compute the outer product of two tensors
 * @param tensor1 The first input tensor
 * @param tensor2 The second input tensor
 * @param output Reference to the output tensor storing the result
 * @details Computes the outer product, producing a matrix:
 * @f[
 * \text{output}_{ij} = \text{tensor1}_i \otimes \text{tensor2}_j = \text{tensor1}_i \times \text{tensor2}_j
 * @f]
 * If tensor1 has size m and tensor2 has size n, output is an m×n matrix.
 */
void Outer_product(const ften &tensor1, const ften &tensor2, ften &output);

/**
 * @brief Compute the inverse of square matrices in batch
 * @param tensor The input tensor (must be invertible square matrices with multiple channels)
 * @param output Reference to the output tensor storing inverse matrices for each channel
 * @details Computes the matrix inverse for each channel:
 * @f[
 * \text{output}_k = A_k^{-1}, \quad \text{where } A_k \times A_k^{-1} = I
 * @f]
 * where @f$ I @f$ is the identity matrix.
 * Uses OpenMP for parallel processing when channels > 1.
 * @warning The input tensor must be non-singular (determinant ≠ 0).
 */
void Batch_Inverse(const ften &tensor, ften &output);

/**
 * @brief Compute the transpose of matrices in batch
 * @param tensor The input tensor
 * @param output Reference to the output tensor storing the transposed result
 * @details Computes the matrix transpose for each channel:
 * @f[
 * \text{output}_k = A_k^T, \quad \text{where } (A_k^T)_{ij} = (A_k)_{ji}
 * @f]
 * Rows and columns are swapped. Uses OpenMP for parallel processing when channels > 1.
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