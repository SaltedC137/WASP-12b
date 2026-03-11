/**
 * @author Aska Lyn
 * @brief Learn how to implement tensors in C++
 * @date 2026-03-04 11:10:27
 */

#include "tensor.hpp"
#include "check.hpp"
#include <armadillo>
#include <cstdint>


namespace dlc_inf {

Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, channels);
  if (channels == 1 && rows == 1) {
    this->raw_shape_ = std::vector<uint32_t>{cols};
  } else if (channels == 1){
    this->raw_shape_ = std::vector<uint32_t>{rows, cols};
    }else {
    this->raw_shape_ = std::vector<uint32_t>{channels,rows,cols};
    }
}

Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, 1);
  this->raw_shape_ = std::vector<uint32_t>{rows, cols};
}

Tensor<float>::Tensor(uint32_t size) {
  data_ = arma::fcube(1, size, 1);
  this->raw_shape_ = std::vector<uint32_t>{size};
}

Tensor<float>::Tensor(const std::vector<uint32_t>& shapes){
  CHECK(!shapes.empty() && shapes.size() <= 3)
      << "Shape vector cannot be empty";
  uint32_t remaining = 3 - shapes.size();
  std::vector<uint32_t> shapes_(3, 1);
  std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

  uint32_t channels = shapes_.at(0);
  uint32_t rows = shapes_.at(1);
  uint32_t cols = shapes_.at(2);

  data_ = arma::fcube(rows, cols, channels);

  if (channels == 1 && rows == 1) {
    this->raw_shape_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shape_ = std::vector<uint32_t>{rows, cols};
  }else {
    this->raw_shape_ = std::vector<uint32_t>{channels, rows, cols};
  }
}


Tensor<float>::Tensor(const Tensor &tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shape_ = tensor.raw_shape_;
    }
}

Tensor<float>::Tensor(Tensor<float> &&tensor) noexcept{
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shape_ = tensor.raw_shape_;
  }
}

Tensor<float>& Tensor<float>::operator=(Tensor<float>&& tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shape_ = tensor.raw_shape_;
  }
  return *this;
}

Tensor<float>& Tensor<float>::operator=(const Tensor& tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shape_ = tensor.raw_shape_;
  }
  return *this;
}


uint32_t Tensor<float>::channels() const {
  CHECK(!this->data_.empty());
  return this->data_.n_slices;
}


uint32_t Tensor<float>::size() const {
  CHECK(!this->data_.empty());
  return this->data_.size();
}


uint32_t Tensor<float>::rows() const {
  CHECK(!this->data_.empty());
  return this->data_.n_rows;
}

uint32_t Tensor<float>::cols() const {
  CHECK(!this->data_.empty());
  return this->data_.n_cols;
}



}