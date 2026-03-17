# WASP-12b DL4Cpp

A modern C++ deep learning inference library with tensor operations, neural network components, and runtime computation graph execution.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-23-blue)](https://en.cppreference.com/w/cpp/compiler_support)
[![CMake](https://img.shields.io/badge/CMake-3.14+-blue.svg)](https://cmake.org/)

> **Project Name**: WASP-12b - A high-performance deep learning inference engine inspired by the exoplanet's intensity.

---

## ⚠️ Status: Under Active Development

This project is currently **under active development**. Core tensor operations and runtime graph infrastructure are implemented. The API is subject to change as the architecture evolves.

---

## Features Overview

### Core Tensor Engine
- Multi-dimensional tensor storage (1D/2D/3D) with efficient memory layout
- Element-wise mathematical operations with broadcasting support
- Comprehensive linear algebra operations (norms, determinant, inverse, transpose)
- OpenMP-parallelized batch operations for performance

### Runtime Computation Graph
- PNNX model format support for loading pre-trained networks
- Topological sorting for execution order optimization
- Operator/Operand abstraction for flexible graph construction
- Type-safe parameter and attribute management
- Memory-efficient weight loading with optional data clearing

### Neural Network Infrastructure
- Abstract layer base class for custom layer implementation
- Layer factory pattern for automatic registration
- Support for common operations (Conv, Pool, Activation - *in progress*)

### Utilities
- Robust logging system with configurable levels
- CHECK macros for runtime assertions
- Centralized thread configuration with RAII guards

---

## Project Structure

```
WASP-12b/
├── DL4Cpp/
│   ├── include/                    # Public header files
│   │   ├── core/
│   │   │   ├── tensor.hpp          # Tensor class definition
│   │   │   ├── tensor_math.hpp     # Element-wise operations
│   │   │   └── tensor_linalg.hpp   # Linear algebra operations
│   │   ├── nn/
│   │   │   └── layer.hpp           # Neural network layer abstraction
│   │   ├── runtime/
│   │   │   ├── rt_ir.hpp           # Runtime computation graph
│   │   │   ├── rt_op.hpp           # Runtime operator definition
│   │   │   ├── rt_opd.hpp          # Runtime operand definition
│   │   │   ├── rt_attr.hpp         # Runtime attribute (weights)
│   │   │   ├── rt_param.hpp        # Runtime parameters (hyperparams)
│   │   │   └── rt_type.hpp         # Type enumerations
│   │   ├── pnnx/
│   │   │   └── ir.h                # PNNX graph format
│   │   ├── utils/
│   │   │   ├── check.hpp           # CHECK assertion macros
│   │   │   ├── log.hpp             # Logging utilities
│   │   │   └── thread_config.hpp   # Thread management
│   │   └── status_code.hpp         # Status code definitions
│   ├── src/                        # Implementation files
│   │   ├── core/
│   │   ├── nn/
│   │   ├── runtime/
│   │   ├── pnnx/
│   │   └── utils/
│   ├── example/
│   │   ├── ex.cpp                  # Comprehensive test examples
│   │   └── bench.cpp               # Performance benchmarks
│   └── CMakeLists.txt
├── build/                          # CMake build directory
├── out/                            # Build artifacts
│   └── bin/                        # Executables and libraries
├── CMakeLists.txt                  # Root configuration
└── README.md
```

---

## Quick Start

### Prerequisites

- **Compiler**: GCC 11+, Clang 14+, or MSVC 2022+
- **CMake**: 3.14 or higher
- **C++ Standard**: C++23
- **Dependencies**:
  - [Armadillo](http://arma.sourceforge.net/) - Linear algebra backend
  - [OpenMP](https://www.openmp.org/) - Parallel computing

### Installation (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y libarmadillo-dev cmake g++ libomp-dev

# Clone repository
git clone https://github.com/yourusername/WASP-12b.git
cd WASP-12b

# Build
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)

# Run tests
./dl4cpp_test

# Run benchmarks
./dl4cpp_bench
```

### Installation (macOS)

```bash
# Install dependencies via Homebrew
brew install armadillo cmake libomp

# Build
mkdir build && cd build
cmake ..
cmake --build . -j$(sysctl -n hw.ncpu)
```

### Installation (Windows with vcpkg)

```powershell
# Install vcpkg and dependencies
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install armadillo:x64-windows

# Set environment variable
$env:VCPKG_ROOT="C:\path\to\vcpkg"

# Build
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build . --config Release
```

---

## Usage Examples

### Tensor Creation and Operations

```cpp
#include "core/tensor.hpp"
#include "core/tensor_math.hpp"
#include "core/tensor_linalg.hpp"

using namespace ctl;
using namespace ctl::math;
using namespace ctl::linalg;

int main() {
    // Create 3D tensor: 2 channels, 3 rows, 4 columns
    Tensor<float> tensor3d(2, 3, 4);
    tensor3d.Rand();  // Fill with random values

    // Create another tensor
    Tensor<float> other(2, 3, 4);
    other.Fill(2.0f);

    // Element-wise addition (functional style)
    auto sum = add(tensor3d, other);

    // Element-wise multiplication (in-place)
    Tensor<float> product(2, 3, 4);
    ElementMultiply(tensor3d, other, product);

    // Matrix operations
    Tensor<float> matrices(4, 10, 10);  // 4 batched 10x10 matrices
    matrices.Rand();

    // Compute determinant for each matrix in batch
    auto det_result = det(matrices);

    // Compute L2 norm
    float norm_value = norm(tensor3d);

    // Compute transpose
    auto transposed = transpose(matrices);

    return 0;
}
```

### Thread Configuration

```cpp
#include "utils/thread_config.hpp"

using namespace ctl;

int main() {
    // Get available CPU cores
    uint32_t cores = ThreadConfig::getNumCores();

    // Set thread count
    ThreadConfig::getInstance().set_thread_count(8);

    // Use RAII guard for scoped configuration
    {
        ThreadGuard guard(16);  // Temporarily use 16 threads
        // ... parallel operations ...
    }  // Automatically restores to 8 threads

    return 0;
}
```

### Runtime Graph (Model Inference)

```cpp
#include "runtime/rt_ir.hpp"

using namespace ctl;

int main() {
    // Load PNNX model
    RuntimeGraph graph("model.param", "model.bin");

    // Build computation graph (topology sort, layer creation)
    graph.Build();

    // Prepare input tensor
    auto input = std::make_shared<Tensor<float>>(1, 3, 224, 224);
    input->Fill(1.0f);

    // Set graph input
    graph.set_inputs("input", {input});

    // Execute forward pass
    graph.Forward(false);

    // Get output
    auto outputs = graph.get_outputs("output");

    return 0;
}
```

---

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `DL4CPP_BUILD_TESTS` | `ON` | Build test executables |
| `DL4CPP_BUILD_BENCHMARKS` | `ON` | Build benchmark executables |
| `DL4CPP_BUILD_DOCS` | `OFF` | Build Doxygen documentation |

### Example: Minimal Build (No Tests/Benchmarks)

```bash
cmake .. -DDL4CPP_BUILD_TESTS=OFF -DDL4CPP_BUILD_BENCHMARKS=OFF
cmake --build .
```

### Generate Documentation

```bash
# Configure with documentation enabled
cmake .. -DDL4CPP_BUILD_DOCS=ON

# Generate documentation
cmake --build . --target docs

# Open in browser
# Linux: xdg-open build/docs/html/index.html
# macOS: open build/docs/html/index.html
# Windows: start build\docs\html\index.html
```

---

## Implemented Features

### ✅ Core Tensor Operations
- [x] Tensor construction (1D, 2D, 3D)
- [x] Copy/Move semantics with proper memory management
- [x] Element access (`index`, `posi`, `at`, `operator()`)
- [x] Shape manipulation (`Reshape`, `Flatten`)
- [x] Padding operations
- [x] Fill operations (`Fill`, `Ones`, `Rand`, `Zeros`)
- [x] Transformation operations (`Transform`)
- [x] Raw pointer access for interoperability

### ✅ Mathematical Operations
- [x] Element-wise arithmetic (Add, Sub, Mul, Div)
- [x] Broadcasting operations (per-channel bias)
- [x] Matrix multiplication (`Matmul`)
- [x] Scalar operations
- [x] Element-wise exponential and clipping
- [x] Functional interface (returning `std::shared_ptr`)

### ✅ Linear Algebra Operations
- [x] Euclidean norm (L2) and L1 norm
- [x] Dot product / inner product
- [x] Batch determinant and trace
- [x] Batch matrix inverse and transpose
- [x] Outer product

### ✅ Runtime Graph Infrastructure
- [x] PNNX model loading
- [x] Graph initialization and building
- [x] Reverse topological sort for execution order
- [x] Operator/Operand abstraction
- [x] Parameter and attribute management
- [x] Type-safe weight retrieval with memory optimization
- [x] Forward execution with data propagation
- [x] Input/Output tensor management

### ✅ Utilities
- [x] CHECK assertion macros
- [x] Logging system (INFO, DEBUG, WARNING, ERROR, FATAL)
- [x] OpenMP parallelization
- [x] Thread configuration with RAII guards

---

## Roadmap

### High Priority (v0.2)
- [ ] Conv2D layer implementation
- [ ] Pooling layers (MaxPool, AvgPool)
- [ ] Activation functions (ReLU, Sigmoid, GELU)
- [ ] Fully connected / Linear layer
- [ ] Softmax and normalization
- [ ] Complete runtime layer implementations

### Medium Priority (v0.3)
- [ ] Batch normalization
- [ ] Dropout layer
- [ ] Model serialization (save/load)
- [ ] Comprehensive unit tests
- [ ] CI/CD pipeline integration

### Future Considerations
- [ ] GPU acceleration (CUDA backend)
- [ ] Quantization support (INT8 inference)
- [ ] Python bindings (pybind11)
- [ ] ONNX runtime support
- [ ] Performance profiling tools

---

## Performance Notes

- **Parallel Execution**: All batch operations leverage OpenMP for multi-core utilization
- **Memory Layout**: Row-major ordering for cache-friendly access patterns
- **Zero-Copy Operations**: Functional interfaces use `std::shared_ptr` to minimize allocations
- **Backend**: Armadillo provides optimized BLAS/LAPACK routines for matrix operations
- **Thread Control**: Centralized `ThreadConfig` enables fine-grained parallelism management

---

## License

This project is licensed under the [MIT License](LICENSE).

```
Copyright (c) 2026 WASP-12b Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
```

---

## Acknowledgments

- [Armadillo](http://arma.sourceforge.net/) - Efficient linear algebra library
- [PNNX](https://github.com/pnnxsoftware/pnnx) - PyTorch model export format
- [ncnn](https://github.com/Tencent/ncnn) - Inspiration for runtime graph design

---

*Last updated: 2026-03-17*
