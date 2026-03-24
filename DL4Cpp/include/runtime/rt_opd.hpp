

/**
 * @file rt_opd.hpp
 * @author Aska Lyn
 * @brief Runtime operand definitions for computation graph execution
 * @details This header defines the runtime representation of operands (data
 * edges) in a computation graph. Operands carry tensor data between operators,
 * storing shape information, data type, and actual tensor contents. Supports
 * both full precision (float) and quantized (int8) execution modes.
 * @date 2026-03-17
 */

#ifndef RUNTIME_OPERAND_HPP
#define RUNTIME_OPERAND_HPP

#include "core/tensor.hpp"
#include "runtime/rt_type.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

namespace ctl {

/**
 * @brief Base template for runtime operand
 * @tparam T Data type of the operand (float, int8_t, etc.)
 * @details Represents an input/output data edge in the runtime computation
 * graph. Each operand has a unique name, shape information, and may contain
 * multiple tensor instances (for batch processing). The type field indicates
 * the runtime data format (e.g., Float32, Int8).
 */
template <typename T> struct RuntimeOperandBase {
  /**
   * @brief Default constructor
   * @details Creates an empty operand with default values.
   */
  explicit RuntimeOperandBase() = default;

  /**
   * @brief Construct operand with existing tensor data
   * @param name Operand identifier name
   * @param shapes Shape dimensions of the operand
   * @param data Vector of tensor instances containing actual data
   * @param type Runtime data type enumeration
   * @details Initializes operand with pre-existing tensor data. Used when
   * propagating outputs from one operator to inputs of downstream operators.
   */
  explicit RuntimeOperandBase(std::string name, std::vector<int32_t> shapes,
                              std::vector<std::shared_ptr<Tensor<T>>> data,
                              RuntimeDataType type)
      : name(std::move(name)), shapes(std::move(shapes)), datas(std::move(data)), type(type) {}

  /**
   * @brief Construct operand with allocated data slots
   * @param name Operand identifier name
   * @param shapes Shape dimensions of the operand
   * @param data_size Number of tensor slots to allocate
   * @param type Runtime data type enumeration
   * @details Pre-allocates space for tensor data without initializing contents.
   * Used during graph build phase to prepare memory layout before execution.
   */
  explicit RuntimeOperandBase(std::string name, std::vector<int32_t> shapes,
                              uint32_t data_size, RuntimeDataType type)
      : name(std::move(name)), shapes(std::move(shapes)), type(type) {
    datas.resize(data_size);
  }

  /**
   * @brief Get total number of elements in the operand
   * @return size_t Total element count (product of all dimensions)
   * @details Computes the product of all shape dimensions. Returns 0 if
   * shape is empty or not yet initialized.
   */
  size_t size() const;

  /// Operand identifier name (matches producer/consumer operator names)
  std::string name;

  /// Shape dimensions (e.g., [batch, channels, height, width])
  std::vector<int32_t> shapes;

  /// Vector of tensor instances (supports batch processing with multiple slots)
  std::vector<std::shared_ptr<Tensor<T>>> datas;

  /// Runtime data type (Float32, Int8, etc.)
  RuntimeDataType type = RuntimeDataType::TypeUnknown;
};

/**
 * @brief Compute total element count from shape dimensions
 * @tparam T Data type
 * @return size_t Product of all shape dimensions, or 0 if empty
 */
template <typename T> size_t RuntimeOperandBase<T>::size() const {
  if (shapes.empty()) {
    return 0;
  }
  size_t size =
      std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
  return size;
}

/// Float operand type alias (standard precision runtime)
using RuntimeOperand = RuntimeOperandBase<float>;

/// Int8 operand type alias (quantized runtime)
using RuntimeOperandQuantized = RuntimeOperandBase<int8_t>;

/// Shared pointer type alias for operand
using RuntimeOperandPtr = std::shared_ptr<RuntimeOperand>;

} // namespace ctl
#endif // RUNTIME_OPERAND_HPP