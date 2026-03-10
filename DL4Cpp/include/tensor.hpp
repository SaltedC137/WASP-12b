/**
 * @author Aska Lyn
 * @brief Learn how to implement tensors in C++
 * @date 2026-03-04 11:10:27
 */

#include <armadillo>

#include <cstdint>
#include <memory>
#include <vector>

namespace dlc_inf
{
    template<typename T = float>
    class Tensor 
    {
        //
    };

    template<>
    class Tensor<uint8_t>
    {
        //
    };

    template<>
    class Tensor<float>
    {
    public:
      explicit Tensor() = default;
      
      /**
       * @brief Generate a 3D tensor
       * @param channels the number of channels in the tensor
       * @param rows the number of rows in the tensor
       * @param cols the number of columns in the tensor
       */

      explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

      explicit Tensor(uint32_t size);

      explicit Tensor(uint32_t rows, uint32_t cols);

      explicit Tensor(const std::vector<uint32_t> &shapes);

      /**
       * @brief Copy constructor: deep copy the data from another tensor
       */ 
      Tensor(const Tensor &tensor);
            
      Tensor<float> &operator=(const Tensor &tensor);
            
            
      /**
       * @brief Move constructor: take over the data from another temporary 
       *        to avoid data duplication and improve performance
       */
      Tensor(Tensor &&tensor) noexcept;
      
      Tensor<float> &operator=(Tensor &&tensor);

      uint32_t rows() const;
      uint32_t cols() const;
      uint32_t channels() const;
      uint32_t size() const;
      bool empty() const;

      std::vector<uint32_t> shapes() const;
      const std::vector<uint32_t> sub_shape() const;

      void set_data(const arma::fcube &data);

      // 1D linear index visit
      float index(uint32_t position) const;
      float &index(uint32_t position);
      
      // 2D longitude and latitude visit
      float posi(uint32_t rows, uint32_t cols) const;
      float &posi(uint32_t rows, uint32_t cols);

      // 3D channels x y
      float at(uint32_t channels , uint32_t rows, uint32_t cols) const;
      float &at(uint32_t channels , uint32_t rows, uint32_t cols);
     
    };
}
