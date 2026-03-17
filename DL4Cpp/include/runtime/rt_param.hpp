

#ifndef RUNTIME_PARAMETER_HPP
#define RUNTIME_PARAMETER_HPP

#include "runtime/rt_type.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace ctl {

// base
struct RuntimeParameter {
  virtual ~RuntimeParameter() = default;
  explicit RuntimeParameter(
      RuntimeParameterType type = RuntimeParameterType::ParameterUnknown)
      : type(type) {}
  RuntimeParameterType type = RuntimeParameterType::ParameterUnknown;
};

// int
struct RuntimeParameterInt : public RuntimeParameter {
  explicit RuntimeParameterInt()
      : RuntimeParameter(RuntimeParameterType::ParameterInt) {}

  explicit RuntimeParameterInt(int32_t param_value)
      : RuntimeParameter(RuntimeParameterType::ParameterInt),
        value(param_value) {}

  int32_t value = 0;
};

// float
struct RuntimeParameterFloat : public RuntimeParameter {
  explicit RuntimeParameterFloat()
      : RuntimeParameter(RuntimeParameterType::ParameterFloat) {}

  explicit RuntimeParameterFloat(float param_value)
      : RuntimeParameter(RuntimeParameterType::ParameterFloat),
        value(param_value) {}

  float value = 0.0f;
};

// string
struct RuntimeParameterString : public RuntimeParameter {
  explicit RuntimeParameterString()
      : RuntimeParameter(RuntimeParameterType::ParameterString) {}

  explicit RuntimeParameterString(std::string param_value)
      : RuntimeParameter(RuntimeParameterType::ParameterString),
        value(std::move(param_value)) {}

  std::string value;
};

// int array
struct RuntimeParameterIntArray : public RuntimeParameter {
  explicit RuntimeParameterIntArray()
      : RuntimeParameter(RuntimeParameterType::ParameterIntArray) {}

  explicit RuntimeParameterIntArray(std::vector<int32_t> param_value)
      : RuntimeParameter(RuntimeParameterType::ParameterIntArray),
        value(std::move(param_value)) {}

  std::vector<int32_t> value;
};

struct RuntimeParameterFloatArray : public RuntimeParameter {
  explicit RuntimeParameterFloatArray()
      : RuntimeParameter(RuntimeParameterType::ParameterFloatArray) {}

  explicit RuntimeParameterFloatArray(std::vector<float> param_value)
      : RuntimeParameter(RuntimeParameterType::ParameterFloatArray),
        value(std::move(param_value)) {}

  std::vector<float> value;
};

// string array

struct RuntimeParameterStringArray : public RuntimeParameter {
  explicit RuntimeParameterStringArray()
      : RuntimeParameter(RuntimeParameterType::ParameterStringArray) {}

  explicit RuntimeParameterStringArray(std::vector<std::string> param_value)
      : RuntimeParameter(RuntimeParameterType::ParameterStringArray),
        value(std::move(param_value)) {}

  std::vector<std::string> value;
};

// bool
struct RuntimeParameterBool : public RuntimeParameter {
  explicit RuntimeParameterBool()
      : RuntimeParameter(RuntimeParameterType::ParameterBool) {}

  explicit RuntimeParameterBool(bool param_value)
      : RuntimeParameter(RuntimeParameterType::ParameterBool),
        value(param_value) {}

  bool value = false;
};



} // namespace ctl

#endif