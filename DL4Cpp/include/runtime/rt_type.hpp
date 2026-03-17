

/**
 * @file rt_type.hpp
 * @author Aska Lyn
 * @brief Runtime type definitions and status codes for computation graph execution
 * @details This header defines enumeration types used throughout the runtime system,
 * including data types for tensor storage (float32, int8, etc.), parameter types for
 * operator configuration, and status codes for error handling. These types ensure
 * type safety and proper data interpretation during model inference.
 * @date 2026-03-17
 */

#ifndef RUNTIME_TYPE_HPP
#define RUNTIME_TYPE_HPP

namespace ctl {

/**
 * @brief Runtime data type enumeration for tensor storage
 * @details Specifies the underlying data format of tensors in the runtime graph.
 * Used for type checking during operator initialization and data propagation.
 * Supports common deep learning precision formats including floating-point and
 * integer quantized types.
 */
enum class RuntimeDataType {
  TypeUnknown = 0,   ///< Unknown or uninitialized data type
  TypeFloat32 = 1,   ///< 32-bit IEEE 754 floating-point
  TypeFloat64 = 2,   ///< 64-bit IEEE 754 floating-point (double precision)
  TypeFloat16 = 3,   ///< 16-bit floating-point (half precision)
  TypeInt32 = 4,     ///< 32-bit signed integer
  TypeInt64 = 5,     ///< 64-bit signed integer
  TypeInt16 = 6,     ///< 16-bit signed integer
  TypeInt8 = 7,      ///< 8-bit signed integer (quantized inference)
  TypeUInt8 = 8,     ///< 8-bit unsigned integer (quantized inference)
};

/**
 * @brief Runtime parameter type enumeration for operator configuration
 * @details Specifies the type of hyperparameters that configure operator behavior.
 * Parameters are distinct from attributes (weights) - they define structural
 * properties like kernel size, stride, padding, etc., rather than learned values.
 */
enum class RuntimeParameterType {
  ParameterUnknown = 0,    ///< Unknown or uninitialized parameter type
  ParameterBool = 1,       ///< Boolean parameter (true/false)
  ParameterInt = 2,        ///< 32-bit signed integer parameter
  ParameterFloat = 3,      ///< 32-bit floating-point parameter
  ParameterString = 4,     ///< String parameter (e.g., activation function name)
  ParameterIntArray = 5,   ///< Array of 32-bit signed integers
  ParameterFloatArray = 6, ///< Array of 32-bit floating-point values
  ParameterStringArray = 7 ///< Array of strings
};

/**
 * @brief Status code enumeration for runtime error handling
 * @details Return codes from runtime operations indicating success or specific
 * error conditions. Used for diagnostic logging and exception handling during
 * graph execution.
 */
enum class StatusCode {
  UnknownCode = -1,       ///< Unknown or uninitialized status code
  Success = 0,            ///< Operation completed successfully
  InferInputsEmpty = 1,   ///< Error: Input tensors are empty or not provided
  InferOutputsEmpty = 2,  ///< Error: Output tensors are empty or not allocated
  InferParamError = 3,    ///< Error: Invalid or missing operator parameters
  InferDimMismatch = 4,   ///< Error: Tensor dimension mismatch in operation
  FunctionNotImplement = 5, ///< Error: Requested function is not implemented
  ParseWeightError = 6,   ///< Error: Failed to parse weight data from model file
  ParseParamError = 7,    ///< Error: Failed to parse parameters from model file
  ParseNullOperator = 8   ///< Error: Encountered null operator during parsing
};

} // namespace ctl
#endif