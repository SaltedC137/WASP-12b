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

#include "tensor.hpp"

namespace ctl::linalg {

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
 * @brief Compute the determinant of a square matrix tensor
 * @param tensor The input tensor (must be a square matrix)
 * @return float The determinant value
 * @details Computes det(tensor) for a square matrix.
 *          @warning The input tensor must represent a square matrix.
 */
float Determinant(const ften &tensor);

/**
 * @brief Compute the trace of a square matrix tensor
 * @param tensor The input tensor (must be a square matrix)
 * @return float The trace value
 * @details Computes the sum of diagonal elements: sum(a_ii).
 *          @warning The input tensor must represent a square matrix.
 */
float Trace(const ften &tensor);

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
 * @brief Compute the inverse of a square matrix tensor
 * @param tensor The input tensor (must be invertible square matrix)
 * @param output Reference to the output tensor storing the result
 * @details Computes tensor^(-1) such that tensor · tensor^(-1) = I.
 *          @warning The input tensor must be a non-singular square matrix.
 */
void Inverse(const ften &tensor, ften &output);

/**
 * @brief Compute the transpose of a tensor
 * @param tensor The input tensor
 * @param output Reference to the output tensor storing the result
 * @details Computes tensor^T, swapping rows and columns.
 *          For a matrix A, output[i,j] = A[j,i].
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
 * @brief Compute the determinant of a square matrix
 * @param tensor The input tensor (must be a square matrix)
 * @return float The determinant value
 * @details Convenience wrapper for Determinant().
 */
inline float det(const ften &tensor) { return Determinant(tensor); }

/**
 * @brief Compute the trace of a square matrix
 * @param tensor The input tensor (must be a square matrix)
 * @return float The trace value
 * @details Convenience wrapper for Trace().
 */
inline float trace(const ften &tensor) { return Trace(tensor); }

/**
 * @brief Compute the outer product returning a new tensor
 * @param tensor1 The first input tensor
 * @param tensor2 The second input tensor
 * @return sft Shared pointer to the result tensor
 * @details Creates a new tensor and computes the outer product.
 *          Convenient for expression chaining without pre-allocating output.
 */
inline sft outer(const ften &tensor1, const ften &tensor2) {
  sft output = std::make_shared<ften>(1, tensor1.size(), tensor2.size());
  Outer_product(tensor1, tensor2, *output);
  return output;
}

/**
 * @brief Compute the transpose returning a new tensor
 * @param tensor The input tensor
 * @return sft Shared pointer to the result tensor
 * @details Creates a new tensor with transposed dimensions.
 *          Convenient for expression chaining.
 */
inline sft transpose(const ften &tensor) {
  sft output =
      std::make_shared<ften>(tensor.channels(), tensor.cols(), tensor.rows());
  Transposition(tensor, *output);
  return output;
}

/**
 * @brief Compute the inverse returning a new tensor
 * @param tensor The input tensor (must be invertible square matrix)
 * @return sft Shared pointer to the result tensor
 * @details Creates a new tensor and computes the matrix inverse.
 *          @warning The input tensor must be a non-singular square matrix.
 */
inline sft inv(const ften &tensor) {
  sft output = std::make_shared<ften>(tensor.shapes());
  Inverse(tensor, *output);
  return output;
}

} // namespace ctl::linalg

#endif