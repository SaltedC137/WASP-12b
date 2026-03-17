

#ifndef RUNTIME_ATTRIBUTE_HPP
#define RUNTIME_ATTRIBUTE_HPP

#include "check.hpp"
#include "log.hpp"
#include "runtime/rt_type.hpp"
#include <cstdint>
#include <cstring>
#include <vector>

namespace ctl {

struct RuntimeAttribute {
  RuntimeAttribute() = default;

  explicit RuntimeAttribute(std::vector<int32_t> shape, RuntimeDataType type,
                            std::vector<char> weight)
      : shape(std::move(shape)), type(type), weight_data(std::move(weight)) {}

  std::vector<int32_t> shape;

  std::vector<char> weight_data;

  RuntimeDataType type = RuntimeDataType::TypeUnknown;

  template <class T>
    requires std::is_trivially_copyable_v<T>
  std::vector<T> get(bool need_clear_weight = true);
};

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

  // clear weight data

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