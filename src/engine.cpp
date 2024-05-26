#include <iostream>

#include "operator.h"
#include "util.h"
#include "cudaop.h"
#include "engine.h"



extern std::vector<std::string> topologicalOrder;
extern std::map<std::string, graphNode> graph; 
extern std::map<std::string, std::unique_ptr<op::Node>> operatorMap;
extern std::map<std::string, std::unique_ptr<cuda::Node>> cudaMap;
extern std::map<std::string, size_t> paraOffsets;
extern size_t totalParaSize;


void BuildCudaOperator()
{
    for (const auto& operatorName : topologicalOrder)
    {
        if((graph[operatorName].in_degree == 0) || (graph[operatorName].dependents.empty()))
        {
            continue;
        }

        else
        {
            const auto& CurrentOperator = operatorMap[operatorName];
            std::string opType = CurrentOperator->type;
            if(opType == "Concat" || opType == "Slice") continue;
            CreateCudaOperator(opType,operatorName);
            auto& CudaOperator = cudaMap[operatorName];
            
            if(PRINTCUDAOP)
            {
                CudaOperator->PrintCudaNode();
                CudaOperator->printArgInfo();
                std::cout<<std::endl;
            }
        }
    }
}

void CreateCudaOperator(const std::string& opType, const std::string& operatorName)
{
    int flag = 1;
    if(opType == "LeakyRelu")
    {
        cudaMap[operatorName] = std::make_unique<cuda::LeakyRelu>(opType,operatorName);
    }
    else if(opType == "Add")
    {
        cudaMap[operatorName] = std::make_unique<cuda::Add>(opType,operatorName);

    }
    else if(opType == "Abs")
    {
        cudaMap[operatorName] = std::make_unique<cuda::Abs>(opType,operatorName);
    }
    else if(opType == "Tanh")
    {
        cudaMap[operatorName] = std::make_unique<cuda::Tanh>(opType,operatorName);

    }
    else if(opType == "Div")
    {
        cudaMap[operatorName] = std::make_unique<cuda::Div>(opType,operatorName);
    }
    else if(opType == "Conv")
    {
        cudaMap[operatorName] = std::make_unique<cuda::Conv>(opType,operatorName);
    }
    else
    {
        std::cout<<"CUDA算子库里没有该算子"<<std::endl;
        flag = 0;
    }
    if(flag)
    {
        auto& cudaOperator = cudaMap[operatorName];
        cudaOperator->para_index = totalParaSize;
        int  kernelpara_size = cudaOperator->SetKernelPara(); 
        paraOffsets[operatorName] = totalParaSize;
        totalParaSize += kernelpara_size;
        return ;
    }
    
}

void PrintParaOffsets()
{
    std::cout << "Parameter Offsets:" << std::endl;
    for (const auto& pair : paraOffsets)
    {
        std::cout << "Key: " << pair.first << ", Offset: " << pair.second << std::endl;
    }
    std::cout << "Total Parameter Size: " << totalParaSize << std::endl;
}