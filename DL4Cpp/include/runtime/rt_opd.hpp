

#ifndef RUNTIME_OPERAND_HPP
#define RUNTIME_OPERAND_HPP

#include "tensor.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>
namespace ctl {

enum class RTDataType {
  kTypeUnknown = 0,
  kTypeFloat32 = 1,
  kTypeFloat64 = 2,
  kTypeFloat16 = 3,
  kTypeInt32 = 4,
  kTypeInt64 = 5,
  kTypeInt16 = 6,
  kTypeInt8 = 7,
  kTypeUInt8 = 8,
};

template <typename T>

struct RTOperandBase {
  explicit RTOperandBase() = default;

  explicit RTOperandBase(std::string name, std::vector<int32_t> shapes,
                         std::vector<std::shared_ptr<Tensor<T>>> data,
                         RTDataType type)
      : name(std::move(name)), shapes(std::move(shapes)), type(type) {}

  explicit RTOperandBase(std::string name, std::vector<int32_t> shapes,
                         uint32_t data_size, RTDataType type)
      : name(std::move(name)), shapes(std::move(shapes)), type(type) {
    datas.resize(data_size);
  }

  size_t size() const;

  std::string name;

  std::vector<int32_t> shapes;

  std::vector<std::shared_ptr<Tensor<T>>> datas;

  RTDataType type = RTDataType::kTypeUnknown;
};

template <typename T> size_t RTOperandBase<T>::size() const {
  if (shapes.empty()) {
    return 0;
  }
  size_t size =
      std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
  return size;
}

using RTOperand = RTOperandBase<float>;

} // namespace ctl

#endif // RUNTIME_OPERAND_HPP