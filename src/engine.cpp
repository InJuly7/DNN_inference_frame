#include <iostream>

#include "operator.h"
#include "util.h"
#include "cudaop.h"
#include "engine.h"

#define PRINTCUDAOP 1

extern std::vector<std::string> topologicalOrder;
extern std::map<std::string, graphNode> graph; 
extern std::map<std::string, std::unique_ptr<op::Node>> operatorMap;
extern std::map<std::string, std::unique_ptr<cuda::Node>> cudaMap;

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
            CreateCudaOperator(opType,operatorName);
            auto& CudaOperator = cudaMap[operatorName];
            if(PRINTCUDAOP)
            {
                CudaOperator->PrintCudaNode();
            }
        }
    }
}

void CreateCudaOperator(const std::string& opType, const std::string& operatorName)
{
    if(opType == "Concat" || opType == "Slice") return;

        else if(opType == "LeakyRelu")
        {
            cudaMap[operatorName] = std::make_unique<cuda::LeakyRelu>(opType,operatorName);
            // auto& cudaOperator = cudaMap[operatorName];
            // cudaOperator
            // cudaOperator->SetKernelPara()
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

        else
        {
            std::cout<<"CUDA算子库里没有该算子"<<std::endl;
        }
}