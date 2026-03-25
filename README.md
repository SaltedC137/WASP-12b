# WASP-12b DL4Cpp

A modern C++23 deep learning inference library featuring tensor operations, neural network components, and runtime computation graph execution with PNNX model support.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-23-blue)](https://en.cppreference.com/w/cpp/compiler_support)
[![CMake](https://img.shields.io/badge/CMake-3.14+-blue.svg)](https://cmake.org/)

> **Project Name**: WASP-12b — Named after the ultra-hot Jupiter exoplanet, symbolizing high-performance inference intensity.

---

## ⚠️ Development Status

**Active Development** — Core tensor operations and runtime graph infrastructure are production-ready. The API may evolve as the architecture matures.

---

## Features

### 🧮 Tensor Engine
- **Multi-dimensional tensors** (1D/2D/3D) with efficient Armadillo-backed storage
- **Element-wise operations**: add, sub, mul, div with broadcasting support
- **Linear algebra**: norms, dot product, determinant, inverse, transpose, trace
- **OpenMP parallelization** for batch operations
- **Zero-copy memory wrapping** for external data interoperability

### 🕸️ Runtime Computation Graph
- **PNNX model format** support for loading PyTorch-exported networks
- **Topological sorting** for optimal execution order
- **Operator/Operand abstraction** for flexible graph construction
- **Type-safe parameter/attribute management**
- **Memory-efficient weight loading** with optional data clearing

### 🧠 Neural Network Infrastructure
- **Abstract layer base class** for custom implementations
- **Factory pattern** for automatic layer registration
- **Activation functions**: ReLU, Sigmoid, SiLU, HardSwish, HardSigmoid, ReLU6
- **Extensible architecture** for Conv2D, Pooling, Linear layers (*in progress*)

### 🛠️ Utilities
- **Configurable logging** system (INFO, DEBUG, WARNING, ERROR, FATAL)
- **CHECK macros** for runtime assertions
- **Centralized thread configuration** with RAII guards
- **Micro-benchmark framework** (ubench)

---

## Project Structure

```
WASP-12b/
├── DL4Cpp/
│   ├── include/                    # Public headers
│   │   ├── core/
│   │   │   ├── tensor.hpp          # Tensor class (1D/2D/3D)
│   │   │   ├── tensor_math.hpp     # Element-wise operations
│   │   │   └── tensor_linalg.hpp   # Linear algebra
│   │   ├── nn/
│   │   │   ├── layer.hpp           # Layer base class
│   │   │   ├── layer_factory.hpp   # Layer registry
│   │   │   ├── param_layer.hpp     # Parameterized layers
│   │   │   └── ops/
│   │   │       ├── activation.hpp  # Activation layer
│   │   │       └── sigmoid.hpp     # Sigmoid implementation
│   │   ├── runtime/
│   │   │   ├── rt_ir.hpp           # Runtime computation graph
│   │   │   ├── rt_op.hpp           # Runtime operator
│   │   │   ├── rt_opd.hpp          # Runtime operand
│   │   │   ├── rt_attr.hpp         # Attributes (weights)
│   │   │   ├── rt_param.hpp        # Parameters (hyperparams)
│   │   │   └── rt_type.hpp         # Type enumerations
│   │   ├── pnnx/
│   │   │   ├── ir.h                # PNNX IR format
│   │   │   └── store_zip.hpp       # Compressed model storage
│   │   ├── utils/
│   │   │   ├── check.hpp           # CHECK assertion macros
│   │   │   ├── log.hpp             # Logging utilities
│   │   │   ├── thread_config.hpp   # Thread management
│   │   │   ├── layer_bench.hpp     # Layer benchmarking
│   │   │   └── fmath.hpp           # Fast math utilities
│   │   └── status_code.hpp         # Status codes
│   ├── src/                        # Implementations
│   │   ├── core/                   # Tensor engine
│   │   ├── nn/                     # Neural network layers
│   │   ├── runtime/                # Graph execution
│   │   ├── pnnx/                   # Model loading
│   │   └── utils/                  # Utilities
│   ├── test/                       # Unit tests
│   │   ├── test_tensor.cpp         # Tensor tests
│   │   ├── test_runtime.cpp        # Graph tests
│   │   ├── test_attr.cpp           # Attribute tests
│   │   ├── test_param.cpp          # Parameter tests
│   │   ├── test_sigmoid.cpp        # Activation tests
│   │   └── bench_calcu.cpp         # Benchmarks
│   └── CMakeLists.txt
├── build/                          # Build directory
├── out/                            # Build artifacts
│   └── bin/                        # Executables
├── example/                        # Usage examples
├── CMakeLists.txt                  # Root configuration
└── README.md
```

---

## Quick Start

### Prerequisites

| Component | Requirement |
|-----------|-------------|
| **Compiler** | GCC 11+, Clang 14+, or MSVC 2022+ |
| **CMake** | 3.14+ |
| **C++ Standard** | C++23 |
| **Dependencies** | Armadillo, OpenMP |

### Installation

#### Ubuntu/Debian

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y libarmadillo-dev cmake g++ libomp-dev

# Clone and build
git clone https://github.com/yourusername/WASP-12b.git
cd WASP-12b
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)

# Run tests
./DL4Cpp/test_tensor
./DL4Cpp/test_sigmoid
```

#### macOS

```bash
# Install dependencies
brew install armadillo cmake libomp

# Build
mkdir build && cd build
cmake ..
cmake --build . -j$(sysctl -n hw.ncpu)
```

#### Windows (vcpkg)

```powershell
# Install vcpkg and dependencies
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install armadillo:x64-windows

# Build
$env:VCPKG_ROOT="C:\path\to\vcpkg"
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build . --config Release
```

---

## Usage Examples

### Tensor Operations

```cpp
#include "core/tensor.hpp"
#include "core/tensor_math.hpp"
#include "core/tensor_linalg.hpp"

using namespace ctl;

int main() {
    // Create 3D tensor: 2 channels × 3 rows × 4 columns
    Tensor<float> tensor3d(2, 3, 4);
    tensor3d.Rand();

    // Element-wise operations
    auto sum = add(tensor3d, tensor3d);           // tensor + tensor
    auto scaled = mul(tensor3d, 2.0f);            // tensor * scalar
    auto shifted = add(tensor3d, 1.0f);           // tensor + scalar

    // Linear algebra
    Tensor<float> matrices(4, 10, 10);            // 4 batched 10×10 matrices
    matrices.Rand();

    auto det_result = det(matrices);              // Batch determinant
    auto norm_value = norm(tensor3d);             // L2 norm
    auto transposed = transpose(matrices);        // Batch transpose

    return 0;
}
```

### Activation Functions

```cpp
#include "core/tensor.hpp"
#include "nn/ops/activation.hpp"

using namespace ctl;
using namespace ctl::nn;

int main() {
    auto input = std::make_shared<Tensor<float>>(1, 32, 32);
    auto output = std::make_shared<Tensor<float>>(1, 32, 32);
    input->Rand();

    // Apply sigmoid activation
    auto sigmoid_func = ApplySSEActivation(ActivationType::ActivationSigmoid);
    sigmoid_func(input, output);

    // Apply ReLU activation
    auto relu_func = ApplySSEActivation(ActivationType::ActivationRelu);
    relu_func(input, output);

    return 0;
}
```

### Runtime Graph Inference

```cpp
#include "runtime/rt_ir.hpp"

using namespace ctl;

int main() {
    // Load PNNX model (exported from PyTorch)
    RuntimeGraph graph("model.param", "model.bin");

    // Build computation graph
    graph.Build();

    // Prepare input
    auto input = std::make_shared<Tensor<float>>(1, 3, 224, 224);
    input->Fill(1.0f);

    // Set input and run inference
    graph.set_inputs("input", {input});
    graph.Forward(false);

    // Get output
    auto outputs = graph.get_outputs("output");

    return 0;
}
```

### Thread Configuration

```cpp
#include "utils/thread_config.hpp"

using namespace ctl;

int main() {
    // Configure global thread count
    ThreadConfig::getInstance().set_thread_count(8);

    // Use RAII guard for scoped override
    {
        ThreadGuard guard(16);  // Temporarily use 16 threads
        // ... parallel operations ...
    }  // Automatically restores to 8 threads

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

### Minimal Build (No Tests/Benchmarks)

```bash
cmake .. -DDL4CPP_BUILD_TESTS=OFF -DDL4CPP_BUILD_BENCHMARKS=OFF
cmake --build .
```

### Generate Documentation

```bash
cmake .. -DDL4CPP_BUILD_DOCS=ON
cmake --build . --target docs
```

---

## Implemented Features

### ✅ Tensor Core
- [x] Tensor construction (1D/2D/3D)
- [x] Copy/Move semantics with proper memory management
- [x] Element access (`index`, `posi`, `at`, `operator()`)
- [x] Shape manipulation (`Reshape`, `Flatten`)
- [x] Padding operations
- [x] Fill operations (`Fill`, `Ones`, `Rand`, `Zeros`)
- [x] Transformation (`Transform`)
- [x] Raw pointer access for interoperability
- [x] External memory wrapping (zero-copy)

### ✅ Mathematical Operations
- [x] Element-wise arithmetic (Add, Sub, Mul, Div)
- [x] Broadcasting (per-channel bias/scale)
- [x] Matrix multiplication (`Matmul`)
- [x] Scalar operations
- [x] Exponential and clipping

### ✅ Linear Algebra
- [x] L2 norm and L1 norm
- [x] Dot product / inner product
- [x] Batch determinant and trace
- [x] Batch matrix inverse and transpose
- [x] Outer product

### ✅ Runtime Graph
- [x] PNNX model loading
- [x] Graph initialization and building
- [x] Reverse topological sort
- [x] Operator/Operand abstraction
- [x] Parameter and attribute management
- [x] Type-safe weight retrieval
- [x] Forward execution with data propagation

### ✅ Neural Network Layers
- [x] Layer base class
- [x] Layer factory pattern
- [x] Activation functions (ReLU, Sigmoid, SiLU, HardSwish, HardSigmoid, ReLU6)
- [ ] Conv2D layer (*in progress*)
- [ ] Pooling layers (*planned*)
- [ ] Linear/FC layer (*planned*)

### ✅ Utilities
- [x] CHECK assertion macros
- [x] Logging system (5 levels)
- [x] OpenMP parallelization
- [x] Thread configuration with RAII guards
- [x] Micro-benchmark framework

---

## Roadmap

### v0.2 (High Priority)
- [ ] Conv2D layer implementation
- [ ] Pooling layers (MaxPool, AvgPool)
- [ ] Fully connected / Linear layer
- [ ] Batch normalization
- [ ] Softmax and LayerNorm
- [ ] Complete runtime layer implementations

### v0.3 (Medium Priority)
- [ ] Dropout layer
- [ ] Model serialization (save/load)
- [ ] Comprehensive unit tests (90%+ coverage)
- [ ] CI/CD pipeline integration
- [ ] Performance profiling tools

### Future Considerations
- [ ] GPU acceleration (CUDA backend)
- [ ] Quantization support (INT8 inference)
- [ ] Python bindings (pybind11)
- [ ] ONNX runtime support
- [ ] Dynamic shape inference

---

## Performance Notes

- **Parallel Execution**: All batch operations leverage OpenMP for multi-core utilization
- **Memory Layout**: Row-major ordering for cache-friendly access patterns
- **Zero-Copy Operations**: Functional interfaces use `std::shared_ptr` to minimize allocations
- **Backend**: Armadillo provides optimized BLAS/LAPACK routines
- **Thread Control**: Centralized `ThreadConfig` enables fine-grained parallelism management
- **SIMD Optimization**: AVX2/FMA instructions enabled for supported compilers

---

## Testing

Run all tests:

```bash
cd build
ctest --output-on-failure
```

Individual test executables:
- `test_tensor` — Tensor creation, access, and operations
- `test_runtime` — Runtime graph execution
- `test_attr` — Runtime attribute management
- `test_param` — Runtime parameter handling
- `test_sigmoid` — Sigmoid activation function (16 test cases)
- `bench_calcu` — Performance benchmarks

---

## License

This project is licensed under the [MIT License](LICENSE).

```
Copyright (c) 2026 WASP-12b Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## Acknowledgments

- [Armadillo](http://arma.sourceforge.net/) — Efficient linear algebra library
- [PNNX](https://github.com/pnnxsoftware/pnnx) — PyTorch model export format
- [ncnn](https://github.com/Tencent/ncnn) — Inspiration for runtime graph design
- [KuiperInfer](https://github.com/zjhellofss/KuiperInfer) — Reference for inference engine architecture

---

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

*Last updated: 2026-03-25*
