

#include "nn/layer.hpp"
#include "tensor.hpp"
#include "utils/check.hpp"
#include "utils/log.hpp"
#include <memory>
#include <vector>

namespace ctl {

const std::vector<std::shared_ptr<Tensor<float>>> &
Layer<float>::weights() const {
  LOG(FATAL) << this->layer_name_ << "layer not complement yet!";
}

} // namespace ctl
