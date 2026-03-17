

/**
 * @file rt_param.hpp
 * @author Aska Lyn
 * @brief Runtime parameter definitions for operator configuration
 * @details This header defines a hierarchy of parameter types used to configure
 * operator behavior in the runtime computation graph. Parameters represent
 * hyperparameters (e.g., kernel size, stride, padding, activation type) that are
 * distinct from learned weights (attributes). Each parameter type wraps a specific
 * value type (int, float, string, or arrays) with type information for safe access.
 * @date 2026-03-17
 */

#ifndef RUNTIME_PARAMETER_HPP
#define RUNTIME_PARAMETER_HPP

#include "runtime/rt_type.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace ctl {

/**
 * @brief Base class for runtime parameters
 * @details Abstract base class providing type identification for all parameter types.
 * Uses virtual destructor for proper polymorphic deletion. Derived classes wrap
 * specific value types while maintaining a common interface for parameter storage
 * and retrieval in operator parameter maps.
 */
struct RuntimeParameter {
  virtual ~RuntimeParameter() = default;
  
  /**
   * @brief Construct parameter with specified type
   * @param type Parameter type enumeration (default: ParameterUnknown)
   */
  explicit RuntimeParameter(
      RuntimeParameterType type = RuntimeParameterType::ParameterUnknown)
      : type(type) {}
  
  /// Parameter type identifier for runtime type checking
  RuntimeParameterType type = RuntimeParameterType::ParameterUnknown;
};

/**
 * @brief 32-bit signed integer parameter
 * @details Wraps a single int32_t value for parameters like kernel size,
 * stride, padding, num_channels, etc.
 */
struct RuntimeParameterInt : public RuntimeParameter {
  /**
   * @brief Default constructor (value initialized to 0)
   */
  explicit RuntimeParameterInt()
      : RuntimeParameter(RuntimeParameterType::ParameterInt) {}

  /**
   * @brief Construct with specified value
   * @param param_value Integer parameter value
   */
  explicit RuntimeParameterInt(int32_t param_value)
      : RuntimeParameter(RuntimeParameterType::ParameterInt),
        value(param_value) {}

  int32_t value = 0;  ///< Stored integer value
};

/**
 * @brief 32-bit floating-point parameter
 * @details Wraps a single float value for parameters like momentum, epsilon,
 * slope, threshold, etc.
 */
struct RuntimeParameterFloat : public RuntimeParameter {
  /**
   * @brief Default constructor (value initialized to 0.0f)
   */
  explicit RuntimeParameterFloat()
      : RuntimeParameter(RuntimeParameterType::ParameterFloat) {}

  /**
   * @brief Construct with specified value
   * @param param_value Floating-point parameter value
   */
  explicit RuntimeParameterFloat(float param_value)
      : RuntimeParameter(RuntimeParameterType::ParameterFloat),
        value(param_value) {}

  float value = 0.0f;  ///< Stored floating-point value
};

/**
 * @brief String parameter
 * @details Wraps a std::string value for parameters like activation function
 * name, mode selection, or other textual configuration.
 */
struct RuntimeParameterString : public RuntimeParameter {
  /**
   * @brief Default constructor (empty string)
   */
  explicit RuntimeParameterString()
      : RuntimeParameter(RuntimeParameterType::ParameterString) {}

  /**
   * @brief Construct with specified value
   * @param param_value String parameter value
   */
  explicit RuntimeParameterString(std::string param_value)
      : RuntimeParameter(RuntimeParameterType::ParameterString),
        value(std::move(param_value)) {}

  std::string value;  ///< Stored string value
};

/**
 * @brief Array of 32-bit signed integers
 * @details Wraps a std::vector<int32_t> for parameters like kernel_shape,
 * strides, pads, output_shape, etc.
 */
struct RuntimeParameterIntArray : public RuntimeParameter {
  /**
   * @brief Default constructor (empty array)
   */
  explicit RuntimeParameterIntArray()
      : RuntimeParameter(RuntimeParameterType::ParameterIntArray) {}

  /**
   * @brief Construct with specified value
   * @param param_value Vector of integer values
   */
  explicit RuntimeParameterIntArray(std::vector<int32_t> param_value)
      : RuntimeParameter(RuntimeParameterType::ParameterIntArray),
        value(std::move(param_value)) {}

  std::vector<int32_t> value;  ///< Stored integer array
};

/**
 * @brief Array of 32-bit floating-point values
 * @details Wraps a std::vector<float> for parameters like weight scales,
 * bias values, or other floating-point sequences.
 */
struct RuntimeParameterFloatArray : public RuntimeParameter {
  /**
   * @brief Default constructor (empty array)
   */
  explicit RuntimeParameterFloatArray()
      : RuntimeParameter(RuntimeParameterType::ParameterFloatArray) {}

  /**
   * @brief Construct with specified value
   * @param param_value Vector of floating-point values
   */
  explicit RuntimeParameterFloatArray(std::vector<float> param_value)
      : RuntimeParameter(RuntimeParameterType::ParameterFloatArray),
        value(std::move(param_value)) {}

  std::vector<float> value;  ///< Stored floating-point array
};

/**
 * @brief Array of strings
 * @details Wraps a std::vector<std::string> for parameters like multiple
 * activation names, mode lists, or other string sequences.
 */
struct RuntimeParameterStringArray : public RuntimeParameter {
  /**
   * @brief Default constructor (empty array)
   */
  explicit RuntimeParameterStringArray()
      : RuntimeParameter(RuntimeParameterType::ParameterStringArray) {}

  /**
   * @brief Construct with specified value
   * @param param_value Vector of string values
   */
  explicit RuntimeParameterStringArray(std::vector<std::string> param_value)
      : RuntimeParameter(RuntimeParameterType::ParameterStringArray),
        value(std::move(param_value)) {}

  std::vector<std::string> value;  ///< Stored string array
};

/**
 * @brief Boolean parameter
 * @details Wraps a bool value for flags like training mode, bias flag,
 * affine transform enable/disable, etc.
 */
struct RuntimeParameterBool : public RuntimeParameter {
  /**
   * @brief Default constructor (value initialized to false)
   */
  explicit RuntimeParameterBool()
      : RuntimeParameter(RuntimeParameterType::ParameterBool) {}

  /**
   * @brief Construct with specified value
   * @param param_value Boolean parameter value
   */
  explicit RuntimeParameterBool(bool param_value)
      : RuntimeParameter(RuntimeParameterType::ParameterBool),
        value(param_value) {}

  bool value = false;  ///< Stored boolean value
};

} // namespace ctl
#endif