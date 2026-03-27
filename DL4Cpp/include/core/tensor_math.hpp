/**
 * @file tensor_math.hpp
 * @author Aska Lyn
 * @brief Mathematical operations for tensor objects in deep learning
 * @details This header provides element-wise arithmetic operations, matrix
 * multiplication, and other mathematical transformations for Tensor objects.
 * All operations support both in-place modification (void functions with output
 * parameter) and functional style (inline functions returning shared_ptr).
 * @date 2026-03-12
 */

#ifndef TENSOR_MATH_HPP
#define TENSOR_MATH_HPP

#include "core/tensor.hpp"
#include <armadillo>
#include <memory>

namespace ctl {

/**
 * @brief Element-wise addition of two tensors
 * @param tensor1 The first input tensor
 * @param tensor2 The second input tensor
 * @param output Reference to the output tensor storing the result
 * @details Computes element-wise addition:
 * @f[
 * \text{output}_i = \text{tensor1}_i + \text{tensor2}_i
 * @f]
 * Both input tensors must have the same shape.
 */
void ElementAdd(const ften &tensor1, const ften &tensor2, ften &output); // +

/**
 * @brief Element-wise multiplication of two tensors (Hadamard product)
 * @param tensor1 The first input tensor
 * @param tensor2 The second input tensor
 * @param output Reference to the output tensor storing the result
 * @details Computes the Hadamard (element-wise) product:
 * @f[
 * \text{output}_i = \text{tensor1}_i \odot \text{tensor2}_i = \text{tensor1}_i \times \text{tensor2}_i
 * @f]
 * Both input tensors must have the same shape.
 */
void ElementMultiply(const ften &tensor1, const ften &tensor2,
                     ften &output); // *

/**
 * @brief Matrix multiplication of two tensors
 * @param tensor1 The first input tensor (left operand)
 * @param tensor2 The second input tensor (right operand)
 * @param output Reference to the output tensor storing the result
 * @details Performs matrix multiplication. For 2D tensors (matrices) @f$ A @f$ and @f$ B @f$:
 * @f[
 * \text{output} = A \times B, \quad \text{output}_{ij} = \sum_k A_{ik} B_{kj}
 * @f]
 * For higher-dimensional tensors, the operation is applied per batch/channel.
 * The number of columns in tensor1 must equal the number of rows in tensor2.
 */
void Matmul(const ften &tensor1, const ften &tensor2, ften &output); // .

/**
 * @brief Element-wise subtraction of two tensors
 * @param tensor1 The first input tensor (minuend)
 * @param tensor2 The second input tensor (subtrahend)
 * @param output Reference to the output tensor storing the result
 * @details Computes element-wise subtraction:
 * @f[
 * \text{output}_i = \text{tensor1}_i - \text{tensor2}_i
 * @f]
 * Both input tensors must have the same shape.
 */
void ElementSub(const ften &tensor1, const ften &tensor2, ften &output); // -

/**
 * @brief Element-wise division of two tensors
 * @param tensor1 The numerator tensor
 * @param tensor2 The denominator tensor
 * @param output Reference to the output tensor storing the result
 * @details Computes element-wise division:
 * @f[
 * \text{output}_i = \frac{\text{tensor1}_i}{\text{tensor2}_i}
 * @f]
 * Both input tensors must have the same shape.
 * @warning Division by zero is not handled - caller must ensure
 *          tensor2 contains no zero elements.
 */
void ElementDivide(const ften &tensor1, const ften &tensor2, ften &output); // /

// ==================== Scalar Operations ====================

/**
 * @brief Add a scalar value to each element of a tensor
 * @param tensor The input tensor
 * @param scalar The scalar value to add
 * @param output Reference to the output tensor storing the result
 * @details Computes element-wise scalar addition:
 * @f[
 * \text{output}_i = \text{tensor}_i + c
 * @f]
 * where @f$ c @f$ is the scalar value.
 */
void AddScalar(const ften &tensor, float scalar, ften &output);

/**
 * @brief Multiply each element of a tensor by a scalar
 * @param tensor The input tensor
 * @param scalar The scalar multiplier
 * @param output Reference to the output tensor storing the result
 * @details Computes element-wise scalar multiplication:
 * @f[
 * \text{output}_i = c \times \text{tensor}_i
 * @f]
 * where @f$ c @f$ is the scalar multiplier.
 */
void MultiplyScalar(const ften &tensor, float scalar, ften &output);

/**
 * @brief Divide each element of a tensor by a scalar
 * @param tensor The input tensor
 * @param scalar The scalar divisor
 * @param output Reference to the output tensor storing the result
 * @details Computes element-wise scalar division:
 * @f[
 * \text{output}_i = \frac{\text{tensor}_i}{c}
 * @f]
 * where @f$ c @f$ is the scalar divisor.
 * @warning Caller must ensure scalar is non-zero.
 */
void DivideScalar(const ften &tensor, float scalar, ften &output);

/**
 * @brief Subtract a scalar from each element of a tensor
 * @param tensor The input tensor
 * @param scalar The scalar value to subtract
 * @param output Reference to the output tensor storing the result
 * @details Computes element-wise scalar subtraction:
 * @f[
 * \text{output}_i = \text{tensor}_i - c
 * @f]
 * where @f$ c @f$ is the scalar value.
 */
void SubScalar(const ften &tensor, float scalar, ften &output);

/**
 * @brief Apply exponential function to each element of a tensor
 * @param tensor The input tensor
 * @param output Reference to the output tensor storing the result
 * @details Computes the natural exponential element-wise:
 * @f[
 * \text{output}_i = e^{\text{tensor}_i}
 * @f]
 * where @f$ e \approx 2.71828 @f$ is Euler's number.
 */
void ElementExp(const ften &tensor, ften &output);

/**
 * @brief Clip tensor elements to a specified range
 * @param tensor The input tensor
 * @param min_val The minimum value (lower bound)
 * @param max_val The maximum value (upper bound)
 * @param output Reference to the output tensor storing the result
 * @details Clips each element to the range @f$ [\text{min\_val}, \text{max\_val}] @f$:
 * @f[
 * \text{output}_i = \min(\max(\text{tensor}_i, \text{min\_val}), \text{max\_val})
 * @f]
 * Elements less than min_val are set to min_val.
 * Elements greater than max_val are set to max_val.
 */
void ElementClip(const ften &tensor, float min_val, float max_val,
                 ften &output);

// ==================== Functional Interface ====================
// Inline functions that return shared_ptr for convenient chaining

/**
 * @brief Element-wise addition returning a new tensor
 * @param tensor1 The first input tensor
 * @param tensor2 The second input tensor
 * @return sften Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor1 + tensor2.
 *          Convenient for expression chaining without pre-allocating output.
 */
inline sften add(const ften &tensor1, const ften &tensor2) {
  sften output = std::make_shared<ften>(tensor1.shapes());
  ElementAdd(tensor1, tensor2, *output);
  return output;
}

/**
 * @brief Element-wise subtraction returning a new tensor
 * @param tensor1 The minuend tensor
 * @param tensor2 The subtrahend tensor
 * @return sften Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor1 - tensor2.
 */
inline sften sub(const ften &tensor1, const ften &tensor2) {
  sften output = std::make_shared<ften>(tensor1.shapes());
  ElementSub(tensor1, tensor2, *output);
  return output;
}

/**
 * @brief Element-wise multiplication returning a new tensor
 * @param tensor1 The first input tensor
 * @param tensor2 The second input tensor
 * @return sften Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor1 * tensor2 (Hadamard).
 */
inline sften mul(const ften &tensor1, const ften &tensor2) {
  sften output = std::make_shared<ften>(tensor1.shapes());
  ElementMultiply(tensor1, tensor2, *output);
  return output;
}

/**
 * @brief Element-wise division returning a new tensor
 * @param tensor1 The numerator tensor
 * @param tensor2 The denominator tensor
 * @return sften Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor1 / tensor2.
 */
inline sften div(const ften &tensor1, const ften &tensor2) {
  sften output = std::make_shared<ften>(tensor1.shapes());
  ElementDivide(tensor1, tensor2, *output);
  return output;
}

/**
 * @brief Matrix multiplication returning a new tensor
 * @param tensor1 The left operand tensor
 * @param tensor2 The right operand tensor
 * @return sften Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor1 · tensor2.
 */
inline sften matmul(const ften &tensor1, const ften &tensor2) {
  sften output = std::make_shared<ften>(tensor1.channels(), tensor1.rows(),
                                        tensor2.cols());
  Matmul(tensor1, tensor2, *output);
  return output;
}

/**
 * @brief Add scalar to tensor returning a new tensor
 * @param tensor The input tensor
 * @param scalar The scalar value to add
 * @return sften Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor + scalar.
 */
inline sften add(const ften &tensor, float scalar) {
  sften output = std::make_shared<ften>(tensor.shapes());
  AddScalar(tensor, scalar, *output);
  return output;
}

/**
 * @brief Multiply tensor by scalar returning a new tensor
 * @param tensor The input tensor
 * @param scalar The scalar multiplier
 * @return sften Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor * scalar.
 */
inline sften mul(const ften &tensor, float scalar) {
  sften output = std::make_shared<ften>(tensor.shapes());
  MultiplyScalar(tensor, scalar, *output);
  return output;
}

/**
 * @brief Subtract scalar from tensor returning a new tensor
 * @param tensor The input tensor
 * @param scalar The scalar value to subtract
 * @return sften Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor - scalar.
 */
inline sften sub(const ften &tensor, float scalar) {
  sften output = std::make_shared<ften>(tensor.shapes());
  SubScalar(tensor, scalar, *output);
  return output;
}

/**
 * @brief Divide tensor by scalar returning a new tensor
 * @param tensor The input tensor
 * @param scalar The scalar divisor
 * @return sften Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor / scalar.
 */
inline sften div(const ften &tensor, float scalar) {
  sften output = std::make_shared<ften>(tensor.shapes());
  DivideScalar(tensor, scalar, *output);
  return output;
}

/**
 * @brief Apply exponential to tensor returning a new tensor
 * @param tensor The input tensor
 * @return sften Shared pointer to the result tensor
 * @details Creates a new tensor and computes exp(tensor) element-wise.
 */
inline sften exp(const ften &tensor) {
  sften output = std::make_shared<ften>(tensor.shapes());
  ElementExp(tensor, *output);
  return output;
}

/**
 * @brief Clip tensor values returning a new tensor
 * @param tensor The input tensor
 * @param min_val The minimum value (lower bound)
 * @param max_val The maximum value (upper bound)
 * @return sften Shared pointer to the result tensor
 * @details Creates a new tensor with values clipped to [min_val, max_val].
 */
inline sften clip(const ften &tensor, float min_val, float max_val) {
  sften output = std::make_shared<ften>(tensor.shapes());
  ElementClip(tensor, min_val, max_val, *output);
  return output;
}

} // namespace ctl::math

#endif
