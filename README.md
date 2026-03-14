# WASP-12b DL4Cpp

A C++ deep learning library implementing tensor operations and neural network components from scratch.

## ⚠️ Status: Under Construction

This project is currently **under active development**. The API is unstable and subject to change. Features are being incrementally added and refined.

## Project Overview

DL4Cpp aims to provide a lightweight, educational implementation of deep learning primitives in modern C++. The core components include:

- **Tensor**: Multi-dimensional tensor data structures (1D/2D/3D) with efficient memory management
- **Math Operations**: Element-wise arithmetic, matrix multiplication, scalar operations, and transformations
- **Linear Algebra**: Norms, dot products, matrix operations (transpose, inverse, determinant, trace)
- **Neural Networks**: Layer abstractions (under development)
- **Utilities**: CHECK macros for logging and assertions, thread configuration management
- **Backend**: Built on top of Armadillo for optimized linear algebra with OpenMP parallelization

## Project Structure

```
WASP-12b/
├── DL4Cpp/
│   ├── include/                  # Header files
│   │   ├── core/
│   │   │   ├── tensor.hpp            # Tensor class definition
│   │   │   ├── tensor_math.hpp       # Tensor mathematical operations
│   │   │   └── tensor_linalg.hpp     # Linear algebra operations
│   │   ├── nn/
│   │   │   └── layer.hpp             # Neural network layer components
│   │   └── utils/
│   │       ├── check.hpp             # CHECK macros for assertions
│   │       ├── log.hpp               # Logging utilities
│   │       ├── thread_config.hpp     # Thread management configuration
│   │       └── ubench.h              # Micro-benchmark framework
│   ├── src/                      # Implementation files
│   │   ├── core/
│   │   │   ├── tensor.cpp            # Tensor class implementation
│   │   │   ├── tensor_math.cpp       # Tensor math operations
│   │   │   └── tensor_linalg.cpp     # Linear algebra operations
│   │   ├── nn/
│   │   │   └── layer.cpp             # Neural network layer implementation
│   │   └── utils/
│   │       ├── log.cpp               # Logging implementation
│   │       └── thread_config.cpp     # Thread management implementation
│   ├── example/                  # Example code
│   │   ├── ex.cpp                # Feature demonstration examples
│   │   └── bench.cpp             # Performance benchmarks
│   └── CMakeLists.txt            # Build configuration
├── out/                          # Build output directory
│   └── bin/                      # Final output files
├── build/                        # CMake build directory
└── CMakeLists.txt                # Root project configuration
```

## Building

This project uses CMake as its build system.

### Quick Start

```bash
# Configure and build
mkdir build && cd build
cmake ..
cmake --build .
```

### Full Build Process (with Installation)

```bash
# 1. Create build directory
mkdir build && cd build

# 2. Configure project
cmake ..

# 3. Build project
cmake --build .

# 4. Install to out/bin/
cmake --install . --prefix ../out/bin
```

### Using Presets (Recommended)

The project provides `CMakePresets.json` for convenient configuration:

```bash
# Configure using preset
cmake --preset=default

# Build using preset
cmake --build --preset=default
```

### Build Artifacts

After building, output files are located in the `out/bin/` directory:

| File | Description |
|------|-------------|
| `libDL4Cpp.lib` / `libDL4Cpp.a` | Static library |
| `dl4cpp_test.exe` | Test executable |
| `dl4cpp_bench.exe` | Benchmark executable |
| `include/` | Header files copy |

### Running Tests

```bash
cd build
ctest --output-on-failure
```

Or run the test executable directly:

```bash
./out/bin/dl4cpp_test.exe
```

### Running Benchmarks

```bash
./out/bin/dl4cpp_bench.exe
```

## Requirements

- **C++23** or later
- **Armadillo** linear algebra library
- **OpenMP** for parallel computing
- **CMake 3.14+**

### Installing Dependencies

Using vcpkg (recommended):

```bash
# Set VCPKG_ROOT environment variable
set VCPKG_ROOT=C:\path\to\vcpkg  # Windows (cmd)
$env:VCPKG_ROOT="C:\path\to\vcpkg"  # Windows (PowerShell)

# Install Armadillo
vcpkg install armadillo
```

Or use system package managers:

```bash
# Ubuntu/Debian
sudo apt-get install libarmadillo-dev

# macOS (Homebrew)
brew install armadillo

# Windows (vcpkg)
vcpkg install armadillo:x64-windows
```

## Implemented Features

### Core Tensor Operations
- [x] Tensor construction (1D, 2D, 3D)
- [x] Copy/Move semantics
- [x] Element access (`index`, `posi`, `at`)
- [x] Shape manipulation (`Reshape`, `Flatten`)
- [x] Padding operations (`Padding`)
- [x] Fill operations (`Fill`, `One`, `Rand`)
- [x] Transformation operations (`Transform`)
- [x] Data export (`values`, `Show`)
- [x] Raw pointer access (`raw_ptr`, `matrix_raw_ptr`)

### Mathematical Operations
- [x] Element-wise operations (`ElementAdd`, `ElementSub`, `ElementMultiply`, `ElementDivide`)
- [x] Broadcasting operations (tensor + per-channel bias)
- [x] Matrix multiplication (`Matmul`)
- [x] Scalar operations (`AddScalar`, `SubScalar`, `MultiplyScalar`, `DivideScalar`)
- [x] Element-wise exponential (`ElementExp`)
- [x] Element-wise clipping (`ElementClip`)
- [x] Functional interface (inline functions returning `std::shared_ptr`)

### Linear Algebra Operations
- [x] Euclidean norm (L2 norm)
- [x] L1 norm (absolute value norm)
- [x] Dot product / inner product
- [x] Batch determinant
- [x] Batch trace
- [x] Batch matrix inverse
- [x] Matrix transpose
- [x] Outer product

### Neural Network Components
- [x] Layer abstraction (base class)

### Thread Management
- [x] Global thread count configuration
- [x] Scheduling policy control (STATIC, DYNAMIC, GUIDED, AUTO)
- [x] Parallel execution enable/disable switch
- [x] Nested parallelism control
- [x] RAII thread guard for scoped configuration

### Utilities
- [x] CHECK macros (`CHECK`, `CHECK_EQ`, `CHECK_LT`, `CHECK_LE`, `CHECK_GE`)
- [x] Logging system (`LOG(INFO)`, `LOG(DEBUG)`, `LOG(WARNING)`, `LOG(ERROR)`, `LOG(FATAL)`)
- [x] OpenMP parallelization with strict thread management

## Usage Examples

### Basic Tensor Operations

```cpp
#include "tensor.hpp"
#include "tensor_math.hpp"
#include "tensor_linalg.hpp"

using namespace ctl;
using namespace ctl::math;
using namespace ctl::linalg;

int main() {
    // Create tensors
    Tensor<float> a(2, 3, 4);  // 3D tensor: 2 channels, 3x4
    Tensor<float> b(2, 3, 4);
    
    // Fill with random values
    a.Rand();
    b.Rand();
    
    // Element-wise addition (functional style)
    auto result = add(a, b);
    
    // Element-wise multiplication (in-place)
    Tensor<float> output(2, 3, 4);
    ElementMultiply(a, b, output);
    
    // Matrix operations
    Tensor<float> mat(4, 10, 10);  // 4 batched 10x10 matrices
    mat.Rand();
    
    // Compute determinant for each matrix
    auto det_result = det(mat);
    
    // Compute L2 norm
    float norm = norm(a);
    
    return 0;
}
```

### Thread Configuration

```cpp
#include "thread_config.hpp"

using namespace ctl;

int main() {
    // Get available CPU cores
    uint32_t cores = ThreadConfig::getNumCores();
    
    // Set thread count
    ThreadConfig::getInstance().set_thread_count(4);
    
    // Disable parallel execution (for debugging)
    ThreadConfig::getInstance().setParallelEnabled(false);
    
    // Use RAII guard for scoped thread configuration
    {
        ThreadGuard guard(8);  // Use 8 threads in this scope
        // ... parallel operations ...
    }  // Automatically restores previous settings
    
    return 0;
}
```

## TODO

### High Priority
- [ ] Convolution operations (Conv2D, Conv3D)
- [ ] Pooling layers (MaxPool, AvgPool)
- [ ] Activation functions (ReLU, Sigmoid, Tanh, Softmax)
- [ ] Fully connected layers (Dense/Linear)
- [ ] Backpropagation support and gradient computation
- [ ] Loss functions (MSE, CrossEntropy)
- [ ] Optimizers (SGD, Adam)

### Medium Priority
- [ ] Batch normalization
- [ ] Dropout layer
- [ ] Data loading utilities
- [ ] Model serialization (save/load weights)
- [ ] Unit tests for all components
- [ ] Improved thread pool implementation

### Low Priority
- [ ] GPU acceleration support (CUDA backend)
- [ ] Documentation website (Doxygen)
- [ ] Python bindings (pybind11)
- [ ] Example tutorials and extended benchmarks

## Performance Notes

- **Parallel Operations**: Element-wise operations, broadcasting, and batch linear algebra use OpenMP for parallel execution
- **Thread Management**: Centralized `ThreadConfig` provides strict control over thread count and scheduling
- **Memory Layout**: Row-major ordering for compatibility with standard C++ conventions
- **Armadillo Backend**: Leverages optimized BLAS/LAPACK routines for matrix operations

## License

This project is licensed under the MIT License.

---

*Last updated: 2026-03-14*
