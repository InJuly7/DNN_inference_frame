#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

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
extern size_t max_pad_temp;

extern int input_vis_C;
extern int input_ir_C;
extern int input_H;
extern int input_W;

std::unordered_map<std::string, float*> graphinitMap;
std::unordered_map<std::string, float*> cudainitMap;

#define PRINTCUDAOP 0
#define PRINTPARAOFFSET 0

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
            // if(opType == "Concat" || opType == "Slice") continue;
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
        cudaMap[operatorName] = std::make_unique<cuda::ConcreteNode>(opType,operatorName);
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
    
    float *cudaTensor_ptr, *cudaPara_ptr, *cudaPadTemp_ptr;
    cudaMalloc(&cudaTensor_ptr, totalMemorySize*sizeof(float));
    cudaMalloc(&cudaPara_ptr, totalParaSize*sizeof(float));
    cudaMalloc(&cudaPadTemp_ptr, max_pad_temp*sizeof(float));

    HostinitializeTensors();
    writeArrayToFile(graphinitMap["vis"],input_H,input_W,"input_vis.txt");
    writeArrayToFile(graphinitMap["ir"],input_H,input_W,"input_ir.txt");
    paraMemcpy(cudaTensor_ptr,cudaPara_ptr,cudaPadTemp_ptr);
    // 执行
    for(const auto& graph_node : topologicalOrder)
    {
        if(graph_node == "vis" || graph_node == "ir" || graph_node == "output_1") continue;
        auto& op_node = cudaMap[graph_node];
        if(op_node->type == "Slice" || op_node->type == "Concat") continue;
        else
        {
            op_node->Execute();
        }
    }
    
    size_t data_size = tensor_lifetimes["output_1"].tensor_size * sizeof(float);
    cudaMemcpy(graphinitMap["output_1"], cudainitMap["output_1"], data_size, cudaMemcpyDeviceToHost);
    writeArrayToFile(graphinitMap["output_1"],input_H,input_W,"output.txt");

    freeHostTensors();
    cudaFree(cudaTensor_ptr);
    cudaFree(cudaPara_ptr);
    cudaFree(cudaPadTemp_ptr);
}   

void freeHostTensors()
{
    for(auto it = graphinitMap.begin(); it != graphinitMap.end(); ++it)
    {
        free(it->second);
    }
}

void HostinitializeTensors()
{
    float* h_vis;
    float* h_ir;
    float* h_output_1;
    
    h_vis = (float*)malloc(tensor_lifetimes["vis"].tensor_size*sizeof(float));
    h_ir = (float*)malloc(tensor_lifetimes["ir"].tensor_size*sizeof(float));
    h_output_1 = (float*)malloc(tensor_lifetimes["output_1"].tensor_size*sizeof(float));
    // 数据初始化
    for (size_t i = 0; i < tensor_lifetimes["vis"].tensor_size; ++i)
    {
        h_vis[i] = 1.0;
    }
    for (size_t i = 0; i < tensor_lifetimes["ir"].tensor_size; ++i)
    {
        h_ir[i]= 0.0;
    }
    // 将指针存储在映射表中
    graphinitMap["vis"] = h_vis;
    graphinitMap["ir"] = h_ir;
    graphinitMap["output_1"] = h_output_1;
    
}

void paraMemcpy(float* cudaTensor_ptr,float* cudaPara_ptr,float* cudaPadTemp_ptr)
{
    // 将所有kernel用到的参数 传递给设备内存
    // 将输入节点 vis 传入到GPU中
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
                    float* host_data_ptr = graphinitMap[graph_node];
                    float* device_data_ptr = cudaTensor_ptr + input_idx;
                    cudainitMap[graph_node] = device_data_ptr;
                    size_t data_size = tensor_lifetimes[graph_node].tensor_size * sizeof(float);
                    cudaMemcpy(device_data_ptr, host_data_ptr, data_size, cudaMemcpyHostToDevice);
                }
            }
            continue;
        }
        // 输出节点 
        else if(graph[graph_node].dependents.empty())
        {
            for(const auto& pair : tensorOffsets)
            {
                if(pair.second == graph_node)
                {
                    size_t output_idx = pair.first;
                    float* device_data_ptr = cudaTensor_ptr + output_idx; 
                    
                    cudainitMap[graph_node] = device_data_ptr;
                }
            }
            continue;
        }
        //  算子
        const std::string op_type = operatorMap[graph_node]->type;
        if(op_type == "Concat" || op_type == "Slice") continue;
        // 按照各个算子的内存布局 传递参数到GPU 内存池
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
            conv_ptr->d_A = cudaTensor_ptr + conv_ptr->inputs_idx[0];
            conv_ptr->d_C = cudaTensor_ptr + conv_ptr->outputs_idx[0];

            size_t offset = 0;
            cudaMemcpy(device_para_ptr + offset, conv_ptr->weight.data(), conv_ptr->weight_size * sizeof(float), cudaMemcpyHostToDevice);
            conv_ptr->d_weight = device_para_ptr+offset;

            offset += conv_ptr->weight_size;
            if(conv_ptr->bias_size != 0)
            {
                cudaMemcpy(device_para_ptr + offset, conv_ptr->bias.data(), conv_ptr->bias_size * sizeof(float), cudaMemcpyHostToDevice);
            }
            (conv_ptr->bias_size == 0) ? conv_ptr->d_bias = NULL : conv_ptr->d_bias = device_para_ptr + offset;

            offset += conv_ptr->bias_size;
            cudaMemcpy(device_para_ptr + offset, conv_ptr->pads.data(), conv_ptr->pads.size() * sizeof(int), cudaMemcpyHostToDevice);
            conv_ptr->d_pads = reinterpret_cast<int*>(device_para_ptr + offset);
            
            offset += conv_ptr->pads.size();
            cudaMemcpy(device_para_ptr + offset, conv_ptr->edag.data(), conv_ptr->edag.size() * sizeof(int), cudaMemcpyHostToDevice);
            conv_ptr->d_edag = reinterpret_cast<int*>(device_para_ptr + offset);
            
            offset += conv_ptr->edag.size();
            cudaMemcpy(device_para_ptr + offset, conv_ptr->output_shape.data(), conv_ptr->output_shape.size() * sizeof(int), cudaMemcpyHostToDevice);
            conv_ptr->d_output_shape = reinterpret_cast<int*>(device_para_ptr + offset);
            
            offset += conv_ptr->output_shape.size();
            cudaMemcpy(device_para_ptr + offset, conv_ptr->kshape.data(), conv_ptr->kshape.size() * sizeof(int), cudaMemcpyHostToDevice);
            conv_ptr->d_kshape = reinterpret_cast<int*>(device_para_ptr + offset);
            
            offset += conv_ptr->kshape.size();
            cudaMemcpy(device_para_ptr + offset, conv_ptr->strides.data(), conv_ptr->strides.size() * sizeof(int), cudaMemcpyHostToDevice);
            conv_ptr->d_strides = reinterpret_cast<int*>(device_para_ptr + offset);
            
            offset += conv_ptr->strides.size();
            cudaMemcpy(device_para_ptr + offset, conv_ptr->input_shape.data(), conv_ptr->input_shape.size() * sizeof(int), cudaMemcpyHostToDevice);
            conv_ptr->d_input_shape = reinterpret_cast<int*>(device_para_ptr + offset);
            
            offset += conv_ptr->input_shape.size();
            conv_ptr->d_pad_temp = cudaPadTemp_ptr;
            cudaDeviceSynchronize();
        }
        else if(op_type == "Div")
        {
            auto& node = *cudaMap[graph_node];
            const auto& div_ptr = dynamic_cast<cuda::Div*>(&node);

            div_ptr->d_A = cudaTensor_ptr + div_ptr->inputs_idx[0];
            div_ptr->d_C = cudaTensor_ptr + div_ptr->outputs_idx[0];
        }
        else if(op_type == "Add")
        {
            auto& node = *cudaMap[graph_node];
            const auto& add_ptr = dynamic_cast<cuda::Add*>(&node);
            
            add_ptr->d_A = cudaTensor_ptr + add_ptr->inputs_idx[0];
            add_ptr->d_C = cudaTensor_ptr + add_ptr->outputs_idx[0];
            (add_ptr->add_value != 0) ? add_ptr->d_B = NULL : add_ptr->d_B = cudaTensor_ptr + add_ptr->inputs_idx[1];
        }
        else if(op_type == "LeakyRelu")
        {
            auto& node = *cudaMap[graph_node];
            const auto& leakyrelu_ptr = dynamic_cast<cuda::LeakyRelu*>(&node);
            
            leakyrelu_ptr->d_A = cudaTensor_ptr + leakyrelu_ptr->inputs_idx[0];
            leakyrelu_ptr->d_C = cudaTensor_ptr + leakyrelu_ptr->outputs_idx[0];
        }
        else if(op_type == "Tanh")
        {
            auto& node = *cudaMap[graph_node];
            const auto& tanh_ptr = dynamic_cast<cuda::Tanh*>(&node);
            
            tanh_ptr->d_A = cudaTensor_ptr + tanh_ptr->inputs_idx[0];
            tanh_ptr->d_C = cudaTensor_ptr + tanh_ptr->outputs_idx[0];
        }
        else if(op_type == "Abs")
        {
            auto& node = *cudaMap[graph_node];
            const auto& abs_ptr = dynamic_cast<cuda::Abs*>(&node);
            
            abs_ptr->d_A = cudaTensor_ptr + abs_ptr->inputs_idx[0];
            abs_ptr->d_C = cudaTensor_ptr + abs_ptr->outputs_idx[0];
        }
    }

}

void writeArrayToFile(float* d_output_1, int rows, int cols, const char* filename)
{
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // 写入每个元素，假设数组是按行存储的
            fprintf(file, "%.2f ", d_output_1[i * cols + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}