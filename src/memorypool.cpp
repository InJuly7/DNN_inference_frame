#include <iostream>
#include <list>
#include <vector>
#include <string>


#include "util.h"
#include "memorypool.h"



extern std::list<MemoryBlock> memoryPool;
extern std::vector<std::string> topologicalOrder;
extern std::map<std::string, graphNode> graph; 
extern std::unordered_map<std::string, TensorLifeSpan> tensor_lifetimes;
extern std::map<std::string, std::unique_ptr<op::Node>> operatorMap;
extern size_t totalMemorySize = 0;


void MemoryPoolImplementation()
{
    std::vector<std::string> inputTensors;
    std::string outputTensor;
    int current_time = 0;
    for (const auto& operatorName : topologicalOrder)
    {
        // 对于 vis ir 图节点 input_tensor = NULL output_tensor = operatorName 
        if(graph[operatorName].in_degree == 0)
        {
            inputTensors = {};
            outputTensor = operatorName;
        }
        else
        {
            for(int i = 0; i < graph[operatorName].in_degree;i++)
            {
                if(graph[operatorName].inputs[i] == "vis" || graph[operatorName].inputs[i] == "ir")
                {
                    inputTensors.push_back(graph[operatorName].inputs[i]);
                }
                else
                {
                    inputTensors.push_back(graph[operatorName].inputs[i] + "_output_0");
                }
                
            }
            outputTensor = operatorName + "_output_0";
        }
        processOperator(operatorName, inputTensors, outputTensor,current_time);
        current_time++;
        // printMemoryPool();
        inputTensors = {};
        outputTensor = {};
    }

}

std::list<MemoryBlock>::iterator findBlockByName(const std::string& name)
{
    for (auto it = memoryPool.begin(); it != memoryPool.end(); ++it)
    {
        if(it->tensor_name == name)
        {
            return it;
        }
    }
    return  memoryPool.end();
}

void processOperator(const std::string& operator_name, const std::vector<std::string>& inputTensors, const std::string& outputTensor,int current_time)
{
    // std::cout<<operator_name<<std::endl;
    // for(const auto& inputTensor : inputTensors)
    // {
    //     std::cout<<inputTensor<<" ";
    // }
    // std::cout<<std::endl;
    // std::cout<<outputTensor<<std::endl;

    // 图输入节点
    if(graph[operator_name].in_degree == 0)
    {
        allocateMemory(tensor_lifetimes[operator_name].tensor_size,operator_name,tensor_lifetimes[operator_name].end_time);
        return;
    }

    // 不需要执行的 算子节点 concat 
    else if(operatorMap[operator_name]->type == "Concat")
    {
        return ;
    }

    
    else if(canTensorBeOverwritten(operator_name,inputTensors,current_time))
    {
        // 需要执行的可覆盖算子节点, 每次执行完都需要考虑释放块的问题
        if(operatorMap[operator_name]->type == "Slice")
        {

        }
        // 单输入 直接覆盖
        else if(operatorMap[operator_name]->type == "LeakyRelu" || 
                operatorMap[operator_name]->type == "Abs" || 
                operatorMap[operator_name]->type == "Tanh")
        {

        }





        freeMemory(current_time);
    }
    else
    {

    }


}


bool canTensorBeOverwritten(const std::string& operator_name, const std::vector<std::string>& inputTensors, int current_time)
{
    // 强依赖算子 算子类型为conv
    if(operatorMap[operator_name]->type == "Conv")
    {
        return false;
    }
    // 弱依赖算子 add Leakyrelu Tanh div abs 
    else
    {
        // 单/多输入情形
        for (const auto& inputTensor : inputTensors)
        {
            if(tensor_lifetimes[inputTensor].end_time > current_time)
            return false
        }
        // 弱依赖算子的输入Tensor为单依赖情形
        return true;
    }
}

// 分配内存块
void allocateMemory(size_t size, std::string tensor_name)
{
    // 已经是预分配状态的Tensor
    if(findBlockByName(tensor_name) != memoryPool.end())
    {
        return;
    }
    
    for (auto it = memoryPool.begin(); it != memoryPool.end(); ++it)
    {
        if (!it->isAllocated && it->size >= size)
        {  
            if (it->size > size)
            {  
                memoryPool.insert(it, MemoryBlock(size, true, tensor_lifetimes[tensor_name].end_time, tensor_name)); // 在分配块时存储 tensor_name
                it->size -= size;
            }
            else
            {  
                it->isAllocated = true;
                it->releaseTime = tensor_lifetimes[tensor_name].end_time;
                it->tensorName = tensor_name; // 在分配块时存储 tensor_name
            }
            totalMemorySize += size; // 更新内存池总大小
            return;
        }
    }

    memoryPool.emplace_back(size, true, tensor_lifetimes[tensor_name].end_time, tensor_name);
    totalMemorySize += size; // 更新内存池总大小
}

// 释放内存块
void freeMemory(int releaseTime)
{
    for (auto it = memoryPool.begin(); it != memoryPool.end(); ++it)
    {
        if (it->isAllocated && it->releaseTime <= releaseTime)
        {
            it->isAllocated = false;
            it->releaseTime = -1;
            it->tensor_name = {};
            // 尝试与前后块合并
            if (it != memoryPool.begin())
            {
                auto prev = std::prev(it);
                if (!prev->isAllocated)
                {
                    prev->size += it->size;
                    prev->releaseTime = -1;
                    prev->tensor_name = {};
                    memoryPool.erase(it);
                    it = prev;
                }
            }
            auto next = std::next(it);
            if (next != memoryPool.end() && !next->isAllocated)
            {
                it->size += next->size;
                memoryPool.erase(next);
            }
        }
    }
}

void coverageMemory(const std::vector<std::string>& inputTensors, const std::string& outputTensor, )
{
    size_t outputTensorSize = tensor_lifetimes[outputTensor].tensor_size;

    if (inputTensors.size() == 1)
    {
        // 单输入单输出 (如 abs, leakyrelu, slice, tanh)
        auto blockIt = findBlockByName(inputTensors[0]);
        if (blockIt != memoryPool.end())
        {
            if (blockIt->size == outputTensorSize)
            {
                blockIt->tensor_name = outputTensor;  // 直接覆盖
                blockIt->releaseTime = tensor_lifetimes[outputTensor].end_time;
            }
            else if (blockIt->size > outputTensorSize)
            {
                // 拆分内存块, 简化slice 算子的操作
                blockIt->size = outputTensorSize;
                blockIt->tensor_name = outputTensor;
                blockIt->releaseTime = tensor_lifetimes[outputTensor].end_time;
                memoryPool.insert(std::next(blockIt), MemoryBlock("", false, blockIt->size - outputTensorSize));
            }
        }
    }
    
    else if (inputTensors.size() > 1) 
    { 
        // 多输入单输出情形 
        size_t inputTensorTotalSize = 0;
        // 每个输入tensor size 都与输出tensor size 匹配
        bool allMatchOutputSize = true;

        for (const auto& inputTensor : inputTensors)
        {
            size_t size = tensor_lifetimes[inputTensor].tensor_size;
            inputTensorTotalSize += size;
            if (size != outputTensorSize)
            {
                allMatchOutputSize = false;
            }
        }

        if (!allMatchOutputSize && inputTensorTotalSize == outputTensorSize)
        {
            // 多输入单输出 concat
            // 假设内存块已经连续
            auto it = findBlockByName(inputTensors[0]);
            if (it != memoryPool.end())
            {
                it->tensor_name = outputTensor;  // 更改第一个块的名字
                it->size = inputTensorTotalSize;  // 调整大小为总和
                it->releaseTime = tensor_lifetimes[outputTensor].end_time;
                // 移除其他块
                auto next = std::next(it);
                while (next != memoryPool.end() && std::find(inputTensors.begin(), inputTensors.end(), next->tensor_name) != inputTensors.end())
                {
                    next = memoryPool.erase(next);  // 删除其他相关块
                }
            }
        }

        else if (allMatchOutputSize)
        {
            // 多输入单输出 (如 div, add)
            // 选择覆盖一个块，释放其他块，并使释放后的连续空闲块最大化
            std::list<MemoryBlock>::iterator bestBlockIt = memoryPool.end();
            size_t maxFreeSpace = 0;  // 记录最大连续空闲空间大小

            // 检查每个块作为覆盖块时的最大连续空闲空间
            for (const auto& coverCandidate : inputTensors)
            {
                auto candidateIt = findBlockByName(coverCandidate);
                if (candidateIt != memoryPool.end() && candidateIt->size == outputTensorSize)
                {
                    size_t currentFreeSpace = 0;
                    // 计算假设覆盖这个块后的最大连续空闲空间
                    // 释放除了coverCandidate 以外的所有块
                    for (const auto& inputTensor : inputTensors)
                    {
                        if (inputTensor != coverCandidate)
                        {
                            auto blockIt = findBlockByName(inputTensor);
                            if (blockIt != memoryPool.end())
                            {
                                // 假设释放这个块
                                size_t spaceBefore = (blockIt != memoryPool.begin() && !std::prev(blockIt)->isAllocated) ? std::prev(blockIt)->size : 0;
                                size_t spaceAfter = (std::next(blockIt) != memoryPool.end() && !std::next(blockIt)->isAllocated) ? std::next(blockIt)->size : 0;
                                size_t totalSpace = blockIt->size + spaceBefore + spaceAfter;
                                currentFreeSpace = std::max(currentFreeSpace, totalSpace);
                            }
                        }
                    }
                    // 如果找到了更大的连续空间，更新最佳块
                    if (currentFreeSpace > maxFreeSpace)
                    {
                        maxFreeSpace = currentFreeSpace;
                        bestBlockIt = candidateIt;
                    }
                }
            }

            // 执行最佳覆盖并释放其他块
            if (bestBlockIt != memoryPool.end())
            {
                bestBlockIt->tensor_name = outputTensor;  // 覆盖块名
                bestBlockIt->releaseTime = tensor_lifetimes[outputTensor].end_time;
            }
        }
    }
}

void printMemoryPool()
{
    std::cout << "Memory Pool Status:\n";
    std::cout << "-------------------------------------------------------------------\n";
    std::cout << "| Allocated | Size (Bytes) | Release Time | Tensor Name            |\n";
    std::cout << "-------------------------------------------------------------------\n";
    
    for (const auto& block : memoryPool)
    {
        std::cout << "| " << (block.isAllocated ? "Yes" : "No ")
                  << "       | " << block.size
                  << "          | ";
        if (block.releaseTime == -1)
        {
            std::cout << "Free        ";
        }
        else
        {
            std::cout << block.releaseTime;
        }
        std::cout << "         | " << (block.tensor_name.empty() ? "None" : block.tensor_name);
        std::cout << " |\n";
    }
    
    std::cout << "-------------------------------------------------------------------\n";
}
