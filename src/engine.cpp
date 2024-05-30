#include <iostream>

#include "operator.h"
#include "util.h"
#include "cudaop.h"
#include "engine.h"

extern std::vector<std::string> topologicalOrder;
extern std::map<std::string, graphNode> graph; 
extern std::map<std::string, std::unique_ptr<op::Node>> operatorMap;
extern std::map<std::string, std::unique_ptr<cuda::Node>> cudaMap;
extern std::unordered_map<std::string, TensorLifeSpan> tensor_lifetimes;
extern std::multimap<size_t, std::string> tensorOffsets;
extern std::map<std::string, size_t> paraOffsets;
extern size_t totalParaSize;
extern size_t totalMemorySize;

std::unordered_map<std::string, float*> graphinitMap;

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
    if(PRINTPARAOFFSET)
    {
        PrintParaOffsets();
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

void engine()
{
    float *cudaTensor_ptr, *cudaPara_ptr; 
    cudaMalloc(&cudaTensor_ptr, totalMemorySize*sizeof(float));
    cudaMalloc(&cudaPara_ptr, totalParaSize*sizeof(float));

    HostinitializeTensors();
   
    float* d_vis;
    float* d_ir;
    float* d_output_1;

    // 将指针存储在映射表中
    tensorMap["input1"] = d_input1;
    tensorMap["output1"] = d_output1;

    // 将所有kernel用到的参数 传递给设备内存
    for (const auto& graph_node : topologicalOrder)
    {
        // 图的输入节点 将主机端的数据按照索引传递给GPU
        if(graph[graph_node].in_degree == 0)
        {
            for(const auto& pair : tensorOffsets)
            {
                if(pair.second == graph_node)
                {
                    size_t input_idx = pair.first;
                    float *host_data_ptr = graphinitMap[graph_node];
                    float *device_data_ptr = cudaTensor_ptr + input_idx;
                    size_t data_size = tensor_lifetimes[graph_node].tensor_size * sizeof(float);
                    cudaMemcpy(device_data_ptr, host_data_ptr, data_size, cudaMemcpyHostToDevice);
                }
            }
        }
        const std::string op_type = operatorMap[graph_node]->type;
        if(op_type == "Concat" || op_type == "Slice") continue;
        // 按照各个算子的内存布局 传递参数到GPU paraPool
        else if(op_type == "Conv")
        {
            /*
                内存布局
                0                               --> weights
                ... + weight_size               --> bias
                ... + bias_size                 --> pads
                ... + pads.size()               --> edag
                ... + edag.size()               --> output_shape
                ... + output_shape.size()       --> kshape
                ... + kshape.size()             --> strides
                ... + strides.size()            --> input_shape
            */
            auto& node = *cudaMap[graph_node];
            const auto& conv_ptr =  dynamic_cast<cuda::Conv*>(&node);
            float* device_para_ptr = cudaPara_ptr + conv_ptr->para_index;
            size_t offset = 0;
            cudaMemcpy(device_para_ptr + offset, conv_ptr->weight.data(), conv_ptr->weight_size * sizeof(float), cudaMemcpyHostToDevice);
            offset += conv_ptr->weight_size;
            if(conv_ptr->bias_size != 0)
            {
                cudaMemcpy(device_para_ptr + offset, conv_ptr->bias.data(), conv_ptr->bias_size * sizeof(float), cudaMemcpyHostToDevice);
            }
            offset += conv_ptr->bias_size;
            cudaMemcpy(device_para_ptr + offset, conv_ptr->pads.data(), conv_ptr->pads.size() * sizeof(int), cudaMemcpyHostToDevice);
            offset += conv_ptr->pads.size();
            cudaMemcpy(device_para_ptr + offset, conv_ptr->edag.data(), conv_ptr->edag.size() * sizeof(int), cudaMemcpyHostToDevice);
            offset += conv_ptr->edag.size();
            cudaMemcpy(device_para_ptr + offset, conv_ptr->output_shape.data(), conv_ptr->output_shape.size() * sizeof(int), cudaMemcpyHostToDevice);
            offset += conv_ptr->output_shape.size();
            cudaMemcpy(device_para_ptr + offset, conv_ptr->kshape.data(), conv_ptr->kshape.size() * sizeof(int), cudaMemcpyHostToDevice);
            offset += conv_ptr->kshape.size;
            cudaMemcpy(device_para_ptr + offset, conv_ptr->strides.data(), conv_ptr->strides.size() * sizeof(int), cudaMemcpyHostToDevice);
            offset += conv_ptr->strides.size();
            cudaMemcpy(device_para_ptr + offset, conv_ptr->input_shape.data(), conv_ptr->input_shape.size() * sizeof(int), cudaMemcpyHostToDevice);
            offset += conv_ptr->input_shape.size();
            cudaDeviceSynchronize();
        }
        else if(op_type == "Div")
        {
            /*
                内存布局
                0   --> div_const
                4   --> numElements
            */
            
        }
        else if(op_type == "Add")
        {
            /*
                内存布局
                0   --> add_const
                4   --> numElements
            */
            




        }
        else if(op_type == "LeakyRelu")
        {
            /*
                内存布局
                0   --> alpha
                4   --> numElements
            */
            auto& node = *cudaMap[graph_node];
            const auto& leakyrelu_ptr = dynamic_cast<cuda::LeakyRelu*>(&node);
            float* device_para_ptr = cudaPara_ptr + leakyrelu_ptr->para_index;
            size_t offset = 0;
            cudaMemcpy(device_para_ptr + offset, &(leakyrelu_ptr->alpha), sizeof(float), cudaMemcpyHostToDevice);
            offset += 1;
            cudaMemcpy(device_para_ptr + offset, &(leakyrelu_ptr->numElements), sizeof(int), cudaMemcpyHostToDevice);

        }
        else if(op_type == "Tanh")
        {
            /*
                内存布局
                0   --> numElements
            */
           auto& node = *cudaMap[graph_node];
           const auto& tanh_ptr = dynamic_cast<cuda::Tanh*>(&node);
           float* device_para_ptr = cudaPara_ptr + tanh_ptr->para_index;
           cudaMemcpy(device_para_ptr, &(tanh_ptr->numElements), sizeof(int), cudaMemcpyHostToDevice);            
        }
        else if(op_type == "Abs")
        {
            /*
                内存布局
                0   --> numElements
            */
            auto& node = *cudaMap[graph_node];
            const auto& abs_ptr = dynamic_cast<cuda::Abs*>(&node);
            float* device_para_ptr = cudaPara_ptr + abs_ptr->para_index;
            cudaMemcpy(device_para_ptr, &(abs_ptr->numElements), sizeof(int), cudaMemcpyHostToDevice);
        }
    }    
}

void HostinitializeTensors()
{
    float* h_vis;
    float* h_ir;
    float* h_output_1;
    
    h_vis = malloc(tensor_lifetimes["vis"].tensor_size*sizeof(float));
    // h_ir = malloc(tensor_lifetimes["ir"].tensor_size*sizeof(float));
    h_output_1 = malloc(tensor_lifetimes["output_1"].tensor_size*sizeof(float));

    // 数据初始化
    for (size_t i = 0; i < tensor_lifetimes["vis"].tensor_size; ++i)
    {
        h_vis[i] = 1.0f;
    }
    
    // 将指针存储在映射表中
    graphinitMap["vis"] = h_vis;
    // graphinitMap["ir"] = h_ir
    graphinitMap["output_1"] = h_output_1;
}

void paraMemcpy()
{

}