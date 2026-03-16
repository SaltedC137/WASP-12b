

#ifndef RUNTIME_TYPE_HPP
#define RUNTIME_TYPE_HPP

namespace ctl {

enum class RuntimeDataType {
  TypeUnknown = 0,
  TypeFloat32 = 1,
  TypeFloat64 = 2,
  TypeFloat16 = 3,
  TypeInt32 = 4,
  TypeInt64 = 5,
  TypeInt16 = 6,
  TypeInt8 = 7,
  TypeUInt8 = 8,
};

enum class RuntimeParameterType {
  ParameterUnknown = 0,
  ParameterBool = 1,
  ParameterInt = 2,
  ParameterFloat = 3,
  ParameterString = 4,
  ParameterIntArray = 5,
  ParameterFloatArray = 6,
  ParameterStringArray = 7,
};

enum class StatusCode {
  UnknownCode = -1,
  Success = 0,
  InferInputsEmpty = 1,
  InferOutputsEmpty = 2,
  InferParamError = 3,
  InferDimMismatch = 4,
  FunctionNotImplement = 5,
  ParseWeightError = 6,
  ParseParamError = 7,
  ParseNullOperator = 8,
};
} // namespace ctl
#endif