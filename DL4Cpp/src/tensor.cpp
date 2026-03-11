/**
 * @author Aska Lyn
 * @brief Learn how to implement tensors in C++
 * @date 2026-03-04 11:10:27
 */

#include "tensor.hpp"
#include "check.hpp"
#include <armadillo>
#include <cmath>
#include <cstdint>
#include <vector>



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
  const uint32_t remaining = 3 - shapes.size();
  std::vector<uint32_t> shapes_(3, 1);
  std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

  const uint32_t channels = shapes_.at(0);
  const uint32_t rows = shapes_.at(1);
  const uint32_t cols = shapes_.at(2);

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


uint32_t Tensor<float>::rows() const {
  CHECK(!this->data_.empty());
  return this->data_.n_rows;
}

uint32_t Tensor<float>::cols() const {
  CHECK(!this->data_.empty());
  return this->data_.n_cols;
}

uint32_t Tensor<float>::channels() const {
  CHECK(!this->data_.empty());
  return this->data_.n_slices;
}

uint32_t Tensor<float>::size() const {
  CHECK(!this->data_.empty());
  return this->data_.size();
}

bool Tensor<float>::empty() const {
  return this->data_.empty();
}


std::vector<uint32_t> Tensor<float>::shapes() const {
  CHECK(this->data_.empty());
  return std::vector<uint32_t>{this->channels(),this->rows(),this->cols()};
}

const std::vector<uint32_t> Tensor<float>::sub_shape() const {
  CHECK(this->data_.empty());
  return std::vector<uint32_t>{this->channels(),this->rows(),this->cols()};
}



void Tensor<float>::set_data(const arma::fcube &data) {
  CHECK(data.n_rows == this->data_.n_rows)
      << data.n_rows << " != " << this->data_.n_rows;
  CHECK(data.n_cols == this->data_.n_cols)
      << data.n_cols << " != " << this->data_.n_cols;
  CHECK(data.n_slices == this->data_.n_slices)
      << data.n_slices << " != " << this->data_.n_slices;
  this->data_ = data;
}

float Tensor<float>::index(uint32_t position) const{
  CHECK(position < this->data_.size()) << "Tensor index Out of Bound";
  return this->data_.at(position);
}

float& Tensor<float>::index(uint32_t position) {
  CHECK(position < this->data_.size()) << "Tensor index Out of Bound";
  return this->data_.at(position);
}

float Tensor<float>::posi(uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  return this->data_.at(row,col,1);
}

float &Tensor<float>::posi(uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  return this->data_.at(row,col,1);
}

float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col,channel);
}

float& Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col,channel);
}

arma::fcube &Tensor<float>::data() { return this->data_; }

const arma::fcube& Tensor<float>::data() const {
  return this->data_;
}

arma::fmat &Tensor<float>::slice(uint32_t channel) {
  CHECK_LT(channel,this->channels());
  return this->data_.slice(channel);
}

const arma::fmat& Tensor<float>::slice(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

void Tensor<float>::Padding(const std::vector<uint32_t> &pads,
                            float padding_value) {
  CHECK(!this->data_.empty());
  CHECK_EQ(pads.size(), 4);

  const uint32_t pad_rows1 = pads.at(0);  
  const uint32_t pad_rows2 = pads.at(1);  
  const uint32_t pad_cols1 = pads.at(2);
  const uint32_t pad_cols2 = pads.at(3);

  const uint32_t channels = this->channels();
  const uint32_t rows = this->rows();
  const uint32_t cols = this->cols();

  const uint32_t new_rows = rows + pad_rows1 + pad_rows2;
  const uint32_t new_cols = cols + pad_cols1 + pad_cols2;

  if (pad_rows1 == 0 && pad_rows2 == 0 && pad_cols1 == 0 && pad_cols2 == 0) {
      return;
  }
  arma::fcube new_data;
  if (padding_value == 0.f) {
    }
}


}