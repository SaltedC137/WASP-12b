#include "runtime/rt_ir.hpp"
#include "core/tensor.hpp"
#include "log.hpp"
#include <iostream>
#include <memory>
#include <string>

int main() {
    std::string param_path = "unet.pnnx.param";
    std::string bin_path = "unet.pnnx.bin";

    ctl::RuntimeGraph graph(param_path, bin_path);

    LOG(INFO) << "Building UNet graph from PNNX model...";
    
    try {
        graph.Build();
        LOG(INFO) << "Graph built successfully!";
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to build graph: " << e.what();
    }

    return 0;
}
