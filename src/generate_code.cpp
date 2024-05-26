#include <unordered_map>
#include <vector>
#include <string>
#include <memory>

#include "util.h"
#include "generate_code.h"

extern std::map<std::string, std::unique_ptr<op::Node>> operatorMap;
extern std::unordered_map<std::string, TensorLifeSpan> tensor_lifetimes;




void launchKernel(const std::string& opName,const int Input_index,const int Output_index)
{
    auto& CurrentOperator = *operatorMap[opName];
    std::string opType = CurrentOperator.type;

    if (opType == "Conv")
    {
        auto conv_Ptr = dynamic_cast<op::Conv*>(&CurrentOperator);
        conv_Ptr->Execute();
    }

    else if (opType == "LeakyRelu")
    {
        dim3 grid(1, 1, 1);
        dim3 block(1024);
        
        // Fetch parameters
        float *A, *C, B;
        int numElements;
        
        // Launch kernel
        LeakyRelu<<<grid, block>>>(A, B, C, numElements);
    }

    else if (opType == "Add")
    {
        
        auto add_Ptr = dynamic_cast<op::Add*>(&CurrentOperator);
        int numElements = tensor_lifetimes[add_Ptr->inputs[0]].tensor_size;
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        if(add_Ptr->add_value != 0)
        {
            AddKernel_1<<<grid, block>>>(POOL[Input_index], add_const, POOL[Output_index], numElements);

        }
        else
        {
            AddKernel_2(const float *A, const float *B, float *C, int numElements);
        }
        
        
        // Fetch parameters
        
        // Launch kernel
        AddKernel_1<<<grid, block>>>(A, add_const, C, numElements);
    }
    
    // else if (opType == "Div")
    // {
    //     dim3 grid(1, 1, 1);
    //     dim3 block(1024);
        
    //     // Fetch parameters
    //     float *A, *C, div_const;
    //     int numElements;
        
    //     // Launch kernel
    //     Div<<<grid, block>>>(A, div_const, C, numElements);
    // }
    
    // else if (opType == "Abs")
    // {
    //     dim3 grid(1, 1, 1);
    //     dim3 block(1024);
        
    //     // Fetch parameters
    //     float *A, *C;
    //     int numElements;
        
    //     // Launch kernel
    //     Abs<<<grid, block>>>(A, C, numElements);
    // }
    // else if (opName == "Tanh")
    // {
    //     dim3 grid(1, 1, 1);
    //     dim3 block(1024);
        
    //     // Fetch parameters
    //     float *A, *C;
    //     int numElements;
        
    //     // Launch kernel
    //     Tanh<<<grid, block>>>(A, C, numElements);
    // }
}

void generateAllKernels()
{
    // Loop through the topological order and launch kernels
    for (const auto& opName : topologicalOrder)
    {
        launchKernel(opName);
    }
}

void BuildCudaIndex()
{

}
