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


#include "tensor.hpp"
#include <armadillo>
#include <memory>


namespace ctl::math {

/**
 * @brief Element-wise addition of two tensors
 * @param tensor1 The first input tensor
 * @param tensor2 The second input tensor
 * @param output Reference to the output tensor storing the result
 * @details Computes output = tensor1 + tensor2 element-wise.
 *          Both input tensors must have the same shape.
 */
void ElementAdd(const ften &tensor1, const ften &tensor2, ften &output); // +

/**
 * @brief Element-wise multiplication of two tensors (Hadamard product)
 * @param tensor1 The first input tensor
 * @param tensor2 The second input tensor
 * @param output Reference to the output tensor storing the result
 * @details Computes output = tensor1 * tensor2 element-wise.
 *          Both input tensors must have the same shape.
 */
void ElementMultiply(const ften &tensor1, const ften &tensor2,
                     ften &output); // *

/**
 * @brief Matrix multiplication of two tensors
 * @param tensor1 The first input tensor (left operand)
 * @param tensor2 The second input tensor (right operand)
 * @param output Reference to the output tensor storing the result
 * @details Performs matrix multiplication output = tensor1 · tensor2.
 *          For 2D tensors, this is standard matrix multiplication.
 *          For higher-dimensional tensors, operation is applied per batch.
 */
void Matmul(const ften &tensor1, const ften &tensor2, ften &output); // .

/**
 * @brief Element-wise subtraction of two tensors
 * @param tensor1 The first input tensor (minuend)
 * @param tensor2 The second input tensor (subtrahend)
 * @param output Reference to the output tensor storing the result
 * @details Computes output = tensor1 - tensor2 element-wise.
 *          Both input tensors must have the same shape.
 */
void ElementSub(const ften &tensor1, const ften &tensor2, ften &output); // -

/**
 * @brief Element-wise division of two tensors
 * @param tensor1 The numerator tensor
 * @param tensor2 The denominator tensor
 * @param output Reference to the output tensor storing the result
 * @details Computes output = tensor1 / tensor2 element-wise.
 *          Both input tensors must have the same shape.
 *          @warning Division by zero is not handled - caller must ensure
 *          tensor2 contains no zero elements.
 */
void ElementDivide(const ften &tensor1, const ften &tensor2, ften &output); // /

// ==================== Scalar Operations ====================

/**
 * @brief Add a scalar value to each element of a tensor
 * @param tensor The input tensor
 * @param scalar The scalar value to add
 * @param output Reference to the output tensor storing the result
 * @details Computes output = tensor + scalar element-wise.
 */
void AddScalar(const ften &tensor, float scalar, ften &output);

/**
 * @brief Multiply each element of a tensor by a scalar
 * @param tensor The input tensor
 * @param scalar The scalar multiplier
 * @param output Reference to the output tensor storing the result
 * @details Computes output = tensor * scalar element-wise.
 */
void MultiplyScalar(const ften &tensor, float scalar, ften &output);

/**
 * @brief Divide each element of a tensor by a scalar
 * @param tensor The input tensor
 * @param scalar The scalar divisor
 * @param output Reference to the output tensor storing the result
 * @details Computes output = tensor / scalar element-wise.
 *          @warning Caller must ensure scalar is non-zero.
 */
void DivideScalar(const ften &tensor, float scalar, ften &output);

/**
 * @brief Subtract a scalar from each element of a tensor
 * @param tensor The input tensor
 * @param scalar The scalar value to subtract
 * @param output Reference to the output tensor storing the result
 * @details Computes output = tensor - scalar element-wise.
 */
void SubScalar(const ften &tensor, float scalar, ften &output);

/**
 * @brief Apply exponential function to each element of a tensor
 * @param tensor The input tensor
 * @param output Reference to the output tensor storing the result
 * @details Computes output = exp(tensor) element-wise, where exp is the
 *          natural exponential function (e^x).
 */
void ElementExp(const ften &tensor, ften &output);

/**
 * @brief Clip tensor elements to a specified range
 * @param tensor The input tensor
 * @param min_val The minimum value (lower bound)
 * @param max_val The maximum value (upper bound)
 * @param output Reference to the output tensor storing the result
 * @details Clips each element to the range [min_val, max_val].
 *          Elements less than min_val are set to min_val.
 *          Elements greater than max_val are set to max_val.
 */
void ElementClip(const ften &tensor, float min_val, float max_val,
                 ften &output);

// ==================== Functional Interface ====================
// Inline functions that return shared_ptr for convenient chaining

/**
 * @brief Element-wise addition returning a new tensor
 * @param tensor1 The first input tensor
 * @param tensor2 The second input tensor
 * @return sft Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor1 + tensor2.
 *          Convenient for expression chaining without pre-allocating output.
 */
inline sft add(const ften &tensor1, const ften &tensor2) {
  sft output = std::make_shared<ften>(tensor1.shapes());
  ElementAdd(tensor1, tensor2, *output);
  return output;
}

/**
 * @brief Element-wise subtraction returning a new tensor
 * @param tensor1 The minuend tensor
 * @param tensor2 The subtrahend tensor
 * @return sft Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor1 - tensor2.
 */
inline sft sub(const ften &tensor1, const ften &tensor2) {
  sft output = std::make_shared<ften>(tensor1.shapes());
  ElementSub(tensor1, tensor2, *output);
  return output;
}

/**
 * @brief Element-wise multiplication returning a new tensor
 * @param tensor1 The first input tensor
 * @param tensor2 The second input tensor
 * @return sft Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor1 * tensor2 (Hadamard).
 */
inline sft mul(const ften &tensor1, const ften &tensor2) {
  sft output = std::make_shared<ften>(tensor1.shapes());
  ElementMultiply(tensor1, tensor2, *output);
  return output;
}

/**
 * @brief Element-wise division returning a new tensor
 * @param tensor1 The numerator tensor
 * @param tensor2 The denominator tensor
 * @return sft Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor1 / tensor2.
 */
inline sft div(const ften &tensor1, const ften &tensor2) {
  sft output = std::make_shared<ften>(tensor1.shapes());
  ElementDivide(tensor1, tensor2, *output);
  return output;
}

/**
 * @brief Matrix multiplication returning a new tensor
 * @param tensor1 The left operand tensor
 * @param tensor2 The right operand tensor
 * @return sft Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor1 · tensor2.
 */
inline sft matmul(const ften &tensor1, const ften &tensor2) {
  sft output = std::make_shared<ften>(tensor1.channels(),tensor1.rows(),tensor2.cols());
  Matmul(tensor1, tensor2, *output);
  return output;
}

/**
 * @brief Add scalar to tensor returning a new tensor
 * @param tensor The input tensor
 * @param scalar The scalar value to add
 * @return sft Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor + scalar.
 */
inline sft add(const ften &tensor, float scalar) {
  sft output = std::make_shared<ften>(tensor.shapes());
  AddScalar(tensor, scalar, *output);
  return output;
}

/**
 * @brief Multiply tensor by scalar returning a new tensor
 * @param tensor The input tensor
 * @param scalar The scalar multiplier
 * @return sft Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor * scalar.
 */
inline sft mul(const ften &tensor, float scalar) {
  sft output = std::make_shared<ften>(tensor.shapes());
  MultiplyScalar(tensor, scalar, *output);
  return output;
}

/**
 * @brief Subtract scalar from tensor returning a new tensor
 * @param tensor The input tensor
 * @param scalar The scalar value to subtract
 * @return sft Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor - scalar.
 */
inline sft sub(const ften &tensor, float scalar) {
  sft output = std::make_shared<ften>(tensor.shapes());
  SubScalar(tensor, scalar, *output);
  return output;
}

/**
 * @brief Divide tensor by scalar returning a new tensor
 * @param tensor The input tensor
 * @param scalar The scalar divisor
 * @return sft Shared pointer to the result tensor
 * @details Creates a new tensor and computes tensor / scalar.
 */
inline sft div(const ften &tensor, float scalar) {
  sft output = std::make_shared<ften>(tensor.shapes());
  DivideScalar(tensor, scalar, *output);
  return output;
}

/**
 * @brief Apply exponential to tensor returning a new tensor
 * @param tensor The input tensor
 * @return sft Shared pointer to the result tensor
 * @details Creates a new tensor and computes exp(tensor) element-wise.
 */
inline sft exp(const ften &tensor) {
  sft output = std::make_shared<ften>(tensor.shapes());
  ElementExp(tensor, *output);
  return output;
}

/**
 * @brief Clip tensor values returning a new tensor
 * @param tensor The input tensor
 * @param min_val The minimum value (lower bound)
 * @param max_val The maximum value (upper bound)
 * @return sft Shared pointer to the result tensor
 * @details Creates a new tensor with values clipped to [min_val, max_val].
 */
inline sft clip(const ften &tensor, float min_val, float max_val) {
  sft output = std::make_shared<ften>(tensor.shapes());
  ElementClip(tensor, min_val, max_val, *output);
  return output;
}

} // namespace ctl::math

#endif
