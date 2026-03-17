

#ifndef RUNTIME_IR_HPP
#define RUNTIME_IR_HPP

#include <string>

namespace ctl {



class RuntimeGraph {

  public:


    RuntimeGraph(std::string param_path, std::string bin_path) ;

    void Build();

    

};

} // namespace ctl

#endif