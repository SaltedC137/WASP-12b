

/**
 * @file rt_attr.hpp
 * @author Aska Lyn
 * @brief Runtime attribute definitions for operator weights
 * @details This header defines the RuntimeAttribute structure for storing
 * learned weights (e.g., convolution kernels, batch norm statistics, linear
 * layer weights) in the computation graph. Attributes are distinct from
 * parameters - they contain the trained model weights rather than structural
 * hyperparameters. Supports type-safe access with automatic memory management
 * and optional weight data clearing for memory efficiency.
 * @date 2026-03-17
 */

#ifndef RUNTIME_ATTRIBUTE_HPP
#define RUNTIME_ATTRIBUTE_HPP

#include "check.hpp"
#include "log.hpp"
#include "runtime/rt_type.hpp"
#include <cstdint>
#include <cstring>
#include <vector>

namespace ctl {

/**
 * @brief Runtime attribute structure for storing operator weights
 * @details Encapsulates weight data with shape and type information. The weight
 * data is stored as raw bytes (std::vector<char>) for memory efficiency and
 * type-erased storage. The get<T>() method provides type-safe access with
 * runtime type checking and optional memory release after retrieval.
 */
struct RuntimeAttribute {
  /**
   * @brief Default constructor (empty attribute)
   * @details Creates an uninitialized attribute with default values.
   */
  RuntimeAttribute() = default;

  /**
   * @brief Construct attribute with shape, type, and weight data
   * @param shape Tensor shape dimensions (e.g., [out_channels, in_channels, kH, kW])
   * @param type Runtime data type enumeration
   * @param weight Raw byte array containing weight values
   * @details Initializes attribute with pre-loaded weight data from model file.
   * The weight data is moved (not copied) for efficiency.
   */
  explicit RuntimeAttribute(std::vector<int32_t> shape, RuntimeDataType type,
                            std::vector<char> weight)
      : shape(std::move(shape)), type(type), weight_data(std::move(weight)) {}

  /// Shape dimensions of the weight tensor
  std::vector<int32_t> shape;

  /// Raw byte array containing weight data
  std::vector<char> weight_data;

  /// Runtime data type for interpreting the weight data
  RuntimeDataType type = RuntimeDataType::TypeUnknown;

  /**
   * @brief Retrieve weight data as specified type with optional memory release
   * @tparam T Target type for conversion (must be trivially copyable)
   * @param need_clear_weight If true, release weight data memory after retrieval
   * @return std::vector<T> Vector of typed weight values
   * @details Performs type-safe conversion from raw bytes to specified type.
   * Includes runtime type checking to ensure type compatibility. When
   * need_clear_weight is true, the internal weight_data is cleared after
   * retrieval to reduce memory footprint (useful for large models).
   * @warning Currently only supports float type (enforced via static_assert)
   */
  template <class T>
    requires std::is_trivially_copyable_v<T>
  std::vector<T> get(bool need_clear_weight = true);
};

/**
 * @brief Template implementation for weight data retrieval
 * @tparam T Target type (must be trivially copyable)
 * @param need_clear_weight Whether to clear weight data after retrieval
 * @return std::vector<T> Typed weight values
 * @details Performs memcpy from raw byte storage to typed vector. Includes
 * runtime checks for data validity and type matching. Uses if constexpr for
 * compile-time type branching with static_assert fallback for unsupported types.
 */
template <class T>
  requires std::is_trivially_copyable_v<T>

std::vector<T> RuntimeAttribute::get(bool need_clear_weight) {
  CHECK(!weight_data.empty());
  CHECK(type != RuntimeDataType::TypeUnknown);

  const uint32_t elem_size = sizeof(T);
  CHECK_EQ(weight_data.size() % elem_size, 0);
  const uint32_t elem_count = weight_data.size() / elem_size;

  std::vector<T> weights(elem_count);
  if constexpr (std::is_same_v<T, float>) {
    CHECK(type == RuntimeDataType::TypeFloat32)
        << "Runtime type mismatch! Expected Float32.";
    std::memcpy(weights.data(), weight_data.data(), weight_data.size());
  } else if constexpr (std::is_same_v<T, int32_t>) {
    CHECK(type == RuntimeDataType::TypeInt32)
        << "Runtime type mismatch! Expected Int32.";
    std::memcpy(weights.data(), weight_data.data(), weight_data.size());
  }else {
  static_assert(std::is_same_v<T, float>, "Currently only float is supported in get<T>()");
  }

  // Clear weight data if requested to reduce memory footprint
  if(need_clear_weight){
    std::vector<char> empty_vec = std::vector<char>();
    this->weight_data.swap(empty_vec);

  }else {
    LOG(WARNING) << "The weight data is not cleared!";
  }
  return weights;
}

} // namespace ctl
#endif