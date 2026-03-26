# DL4Cpp API Reference

## Overview

**DL4Cpp** is a modern C++23 deep learning inference library featuring tensor operations, neural network components, and runtime computation graph execution with PNNX model support.

**Version:** 0.1.0  
**C++ Standard:** C++23  

---

## Quick Links

- [Tensor Engine] - Multi-dimensional tensor operations
- [Neural Network Layers] - Activation functions and layer abstractions
- [Runtime Graph] - Computation graph execution
- [PNNX Support] - PyTorch model loading
- [Utilities] - Logging, assertions, and thread management

---

## Project Structure

```
DL4Cpp/
├── include/
│   ├── core/           # Tensor engine (tensor.hpp, tensor_math.hpp, tensor_linalg.hpp)
│   ├── nn/             # Neural network (layer.hpp, layer_factory.hpp, ops/)
│   ├── runtime/        # Graph execution (rt_ir.hpp, rt_op.hpp, rt_opd.hpp)
│   ├── pnnx/           # Model format (ir.h, store_zip.hpp)
│   ├── optim/          # Optimizers
│   └── utils/          # Utilities (log.hpp, check.hpp, thread_config.hpp)
├── src/                # Implementation files
├── test/               # Unit tests and benchmarks
└── CMakeLists.txt
```

---

## Namespaces

| Namespace | Description |
|-----------|-------------|
| `ctl` | Core library namespace containing all public APIs |
| `ctl::nn` | Neural network components (layers, activations) |
| `ctl::optim` | Optimization algorithms |

---

## Key Components

### Tensor Engine (`ctl::Tensor`)

The foundation of DL4Cpp, providing efficient multi-dimensional array operations.

**Features:**
- 1D/2D/3D tensor support with Armadillo-backed storage
- Element-wise operations: `add`, `sub`, `mul`, `div`
- Linear algebra: `norm`, `dot`, `det`, `inv`, `transpose`, `trace`
- OpenMP parallelization for batch operations
- Zero-copy memory wrapping

**Example:**
```cpp
#include <core/tensor.hpp>
#include <core/tensor_math.hpp>

ctl::Tensor<float> mat(4, 10, 10);  // 4 batched 10×10 matrices
mat.Rand();
auto result = ctl::det(mat);        // Batch determinant
```

---

### Neural Network Layers (`ctl::nn`)

Abstract layer base class with factory pattern for automatic registration.

**Available Activations:**
- `ReLU` - Rectified Linear Unit
- `Sigmoid` - Logistic function
- `SiLU` - Sigmoid Linear Unit
- `HardSwish` - Hard Swish activation
- `HardSigmoid` - Hard Sigmoid approximation
- `ReLU6` - Clipped ReLU

**Example:**
```cpp
#include <nn/ops/activation.hpp>

auto sigmoid = ctl::nn::ApplySSEActivation(ctl::nn::ActivationType::ActivationSigmoid);
sigmoid(input_tensor, output_tensor);
```

---

### Runtime Computation Graph (`ctl::RuntimeGraph`)

Executes PNNX-exported models with topological sorting and efficient memory management.

**Features:**
- PNNX model format support (`.param` + `.bin`)
- Automatic topological sorting
- Type-safe parameter and attribute retrieval
- Memory-efficient weight loading

**Example:**
```cpp
#include <runtime/rt_ir.hpp>

ctl::RuntimeGraph graph("model.param", "model.bin");
graph.Build();
graph.set_inputs("input", {input_tensor});
graph.Forward(false);
auto outputs = graph.get_outputs("output");
```

---

### PNNX Model Format

Load PyTorch-exported models directly for inference.

**File Structure:**
- `model.param` - Network architecture (JSON-like text)
- `model.bin` - Binary weights (optionally compressed in `.pnnxparam.zip`)

---

### Utilities

| Header | Description |
|--------|-------------|
| `<utils/log.hpp>` | Configurable logging (INFO, DEBUG, WARNING, ERROR, FATAL) |
| `<utils/check.hpp>` | Runtime assertion macros (`CHECK`, `CHECK_EQ`, etc.) |
| `<utils/thread_config.hpp>` | Global thread management with RAII guards |
| `<status_code.hpp>` | Standardized return codes |

---

## Build Requirements

| Component | Requirement |
|-----------|-------------|
| **Compiler** | GCC 11+, Clang 14+, or MSVC 2022+ |
| **CMake** | 3.14+ |
| **C++ Standard** | C++23 |
| **Dependencies** | Armadillo, OpenMP |

---

## Build Instructions

### Windows (MSYS2 MinGW)

```cmd
# Install dependencies (via MSYS2)
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-armadillo mingw-w64-x86_64-openmp

# Configure and build using CMake presets
cmake --preset windows-msys2
cmake --build --preset default-build -j$(nproc)
```

### Windows (vcpkg)

```cmd
# Install dependencies
vcpkg install armadillo:x64-windows

# Build (note: vcpkg requires manual toolchain specification)
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake"
cmake --build . --config Release
```

### Linux

```bash
# Install dependencies
sudo apt-get install libarmadillo-dev libomp-dev

# Configure and build using CMake presets
cmake --preset linux
cmake --build --preset linux-build -j$(nproc)
```

### macOS

```bash
# Install dependencies
brew install armadillo libomp

# Configure and build using CMake presets
cmake --preset macos
cmake --build --preset macos-build -j$(sysctl -n hw.ncpu)
```

### Using Default Preset (Auto-detect Platform)

```bash
# Automatically selects the appropriate preset for your platform
cmake --preset default
cmake --build --preset default-build -j 32
```

---

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `DL4CPP_BUILD_TESTS` | `ON` | Build unit tests |
| `DL4CPP_BUILD_BENCHMARKS` | `ON` | Build benchmarks |
| `DL4CPP_BUILD_DOCS` | `OFF` | Build documentation |

### Generate Documentation

```bash
# Windows (MSYS2)
cmake --preset windows-msys2-docs
cmake --build build --target docs

# Linux
cmake --preset linux-docs
cmake --build build --target docs

# macOS
cmake --preset macos-docs
cmake --build build --target docs
```

### Minimal Build (No Tests/Benchmarks)

```bash
cmake --preset windows-msys2 -DDL4CPP_BUILD_TESTS=OFF -DDL4CPP_BUILD_BENCHMARKS=OFF
cmake --build --preset default-build
```

---

## Testing

Run all tests:
```cmd
cd build
ctest --output-on-failure
```

Individual tests:
- `test_tensor` - Tensor operations
- `test_runtime` - Runtime graph execution
- `test_attr` / `test_param` - Attributes and parameters
- `test_sigmoid` - Activation functions
- `bench_calcu` - Performance benchmarks

---


## Performance Notes

- **Parallel Execution:** OpenMP for multi-core utilization
- **Memory Layout:** Row-major ordering for cache efficiency
- **SIMD Optimization:** AVX2/FMA instructions enabled
- **Zero-Copy:** `std::shared_ptr` minimizes allocations
- **Backend:** Armadillo provides optimized BLAS/LAPACK

---

## Thread Configuration

```cpp
#include <utils/thread_config.hpp>

// Set global thread count
ctl::ThreadConfig::getInstance().set_thread_count(8);

// Scoped override with RAII guard
{
    ctl::ThreadGuard guard(16);  // Use 16 threads in this scope
    // ... operations ...
}  // Automatically restores to 8 threads
```

---

## Logging System

```cpp
#include <utils/log.hpp>

LOG_INFO << "Application started";
LOG_DEBUG << "Debug value: " << value;
LOG_WARNING << "Potential issue detected";
LOG_ERROR << "Error occurred: " << error_msg;
LOG_FATAL << "Critical failure";  // Terminates program
```

---

## CHECK Macros

```cpp
#include <utils/check.hpp>

CHECK(tensor.rows() > 0);
CHECK_EQ(tensor.channels(), 3);
CHECK_NE(ptr, nullptr);
CHECK_GE(value, 0);
CHECK_LT(index, size);
```

---

## Acknowledgments

- **Armadillo** - Linear algebra library
- **PNNX** - PyTorch export format
- **ncnn** - Runtime graph design inspiration
- **KuiperInfer** - Inference engine reference

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and open a Pull Request

---

*Last updated: 2026-03-25*
