#include <iostream>
#include <list>
#include <vector>
#include <string>
#include <algorithm>

#include "util.h"
#include "memorypool.h"

#define PRINT_MEMORYPOOL 1
#define PRINT_TENSOROFFSET 0

extern std::list<MemoryBlock> memoryPool;
extern std::vector<std::string> topologicalOrder;
extern std::map<std::string, graphNode> graph; 
extern std::unordered_map<std::string, TensorLifeSpan> tensor_lifetimes;
extern std::map<std::string, std::unique_ptr<op::Node>> operatorMap;
extern size_t totalMemorySize;;
extern std::multimap<size_t, std::string> tensorOffsets;

extern std::string getNodeName(const std::string& outputName);


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
        else if(graph[operatorName].dependents.empty())
        {
            continue;
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
            outputTensor = operatorMap[operatorName]->outputs[0];
        }
        processOperator(operatorName, inputTensors, outputTensor,current_time);
        processOutputTensor(operatorName,outputTensor);
        updateTensorOffsets();
        current_time++;

        if (PRINT_MEMORYPOOL)
        {
            printMemoryPool();
        }
        
        inputTensors = {};
        outputTensor = {};
    }
    if(PRINT_TENSOROFFSET)
    {
        printTensorOffsets();
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

size_t calculateOffset(std::list<MemoryBlock>::iterator blockIt)
{
    size_t offset = 0;
    for (auto it = memoryPool.begin(); it != blockIt; ++it)
    {
        offset += it->size;
    }
    return offset;
}

void processOperator(const std::string& operator_name, const std::vector<std::string>& inputTensors, const std::string& outputTensor,int current_time)
{
    // 图输入节点
    if(graph[operator_name].in_degree == 0)
    {
        allocateMemory(tensor_lifetimes[outputTensor].tensor_size,outputTensor);
        return;
    }

    // 不需要执行的 算子节点 concat 
    else if(operatorMap[operator_name]->type == "Concat")
    {
        coverageMemory(inputTensors,outputTensor);
    }
    
    else if(canTensorBeOverwritten(operator_name,inputTensors,current_time))
    {
        // 需要执行的可覆盖算子节点, 每次执行完都需要考虑释放块的问题
        if(operatorMap[operator_name]->type == "Slice")
        {
            coverageMemory(inputTensors,outputTensor);
        }
        // 单输入 直接覆盖
        else if(operatorMap[operator_name]->type == "LeakyRelu" || 
                operatorMap[operator_name]->type == "Abs" || 
                operatorMap[operator_name]->type == "Tanh")
        {
            coverageMemory(inputTensors,outputTensor);
        }

        else if(operatorMap[operator_name]->type == "Add" || 
                operatorMap[operator_name]->type == "Div")
        {
            coverageMemory(inputTensors,outputTensor);
        }

        freeMemory(current_time);
    }
    else if (operatorMap[operator_name]->type == "Conv")
    {
        allocateMemory(tensor_lifetimes[outputTensor].tensor_size,outputTensor);
        freeMemory(current_time);
    }


}

void processOutputTensor(const std::string& operator_name, const std::string& outputTensor)
{
    std::string partner_Tensor = {};
    std::string partner_Node = {};
    int index = -1; 
    bool found = false;

    // 输出Tensor 的依赖是concat 算子
    if(tensor_lifetimes[outputTensor].special_flag == true)
    {
        // 是否执行预分配 
        // 内存池中没有 concat输入Tensor
        for(const auto& dependent : graph[operator_name].dependents)
        {
            if(operatorMap[dependent]->type == "Concat")
            {
                for(index = 0; index < operatorMap[dependent]->inputs.size(); index++)
                {
                    if(operatorMap[dependent]->inputs[index] != outputTensor )
                    {
                        partner_Tensor = operatorMap[dependent]->inputs[index];
                        if(tensor_lifetimes[partner_Tensor].start_time > tensor_lifetimes[outputTensor].start_time)
                        {
                            found = true;
                            break;
                        }
                    }
                        
                }
            }
            if(found) break;
            // 否则 不需要预分配
            else return;
        }
        // 如果新块覆盖了其输入，应该使用输入的名称作为新块的名称
        
        partner_Node = getNodeName(partner_Tensor);
        bool flag = canTensorBeOverwritten(partner_Node, graph[partner_Node].inputs,tensor_lifetimes[partner_Node].start_time);
        
        while(flag)
        {
            partner_Tensor = graph[partner_Node].inputs[0] + "_output_0";
            partner_Node = getNodeName(partner_Tensor);
            flag = canTensorBeOverwritten(partner_Node, graph[partner_Node].inputs,tensor_lifetimes[partner_Node].start_time);
        }

        auto blockIt = findBlockByName(outputTensor);
        if (blockIt != memoryPool.end())
        {
            size_t insertPosition = calculateOffset(blockIt);
            // 根据index决定新块的位置
            if (index == 0)
            {
                // 在outputTensor前面分配块
                memoryPool.insert(blockIt, MemoryBlock(tensor_lifetimes[partner_Tensor].tensor_size , true, tensor_lifetimes[partner_Tensor].end_time , partner_Tensor));
                updateOffsets(insertPosition, tensor_lifetimes[partner_Tensor].tensor_size, partner_Tensor);
            }
            else if (index == 1)
            {
                // 在outputTensor后面分配块
                auto nextIt = std::next(blockIt);
                size_t nextPosition = insertPosition + blockIt->size;
                memoryPool.insert(nextIt, MemoryBlock(tensor_lifetimes[partner_Tensor].tensor_size, true, tensor_lifetimes[partner_Tensor].end_time, partner_Tensor));
                updateOffsets(nextPosition, tensor_lifetimes[partner_Tensor].tensor_size, partner_Tensor);
            }
        }  
    }
    else
    {
        return ;
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
            return false;
        }
        // 弱依赖算子的输入Tensor为单依赖情形
        return true;
    }
}

// 分配内存块
void allocateMemory(size_t size, std::string tensor_name)
{
    // 检查是否已经分配
    if(findBlockByName(tensor_name) != memoryPool.end())
    {
        return;
    }

    // 查找最合适的空闲块
    auto bestFit = memoryPool.end();
    size_t minSizeDiff = std::numeric_limits<size_t>::max();

    for (auto it = memoryPool.begin(); it != memoryPool.end(); ++it)
    {
        if (!it->isAllocated && it->size >= size)
        {
            size_t sizeDiff = it->size - size;
            if (sizeDiff < minSizeDiff)
            {
                minSizeDiff = sizeDiff;
                bestFit = it;
                if (sizeDiff == 0) break; // 完美匹配，直接结束查找
            }
        }
    }

    if (bestFit != memoryPool.end())
    {
        // 发现适合的空闲块
        if (bestFit->size > size)
        {
            // 拆分块
            memoryPool.insert(bestFit, MemoryBlock(size, true, tensor_lifetimes[tensor_name].end_time, tensor_name));
            bestFit->size -= size;
        }
        else
        {
            // 大小完全匹配，直接使用
            bestFit->isAllocated = true;
            bestFit->tensor_name = tensor_name;
            bestFit->releaseTime = tensor_lifetimes[tensor_name].end_time;
        }
    }
    else
    {
        // 没有足够大的空闲块，找最大的空闲块并扩展
        std::list<MemoryBlock>::iterator largestFreeBlock = memoryPool.end();
        size_t largestFreeSize = 0;
        size_t largestFreePosition = 0;
        size_t current_position = 0;
        for (auto it = memoryPool.begin(); it != memoryPool.end(); ++it)
        {
            if (!it->isAllocated && it->size > largestFreeSize)
            {
                largestFreeSize = it->size;
                largestFreeBlock = it;
                largestFreePosition = current_position;
            }
            current_position += it->size;
        }

        if (largestFreeBlock != memoryPool.end() && !largestFreeBlock->isAllocated)
        {
            // 扩展最大的空闲块
            updateOffsets(largestFreePosition,size-largestFreeBlock->size,tensor_name);
            largestFreeBlock->size = size; // 假设这里可以直接扩展到所需大小
            largestFreeBlock->isAllocated = true;
            largestFreeBlock->tensor_name = tensor_name;
            largestFreeBlock->releaseTime = tensor_lifetimes[tensor_name].end_time;
        }
        else
        {
            // 如果所有块都已分配，或找不到空闲块，则新建块
            memoryPool.emplace_back(size, true, tensor_lifetimes[tensor_name].end_time, tensor_name);
        }
    }
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

void coverageMemory(const std::vector<std::string>& inputTensors, const std::string& outputTensor)
{
    size_t outputTensorSize = tensor_lifetimes[outputTensor].tensor_size;

    if (inputTensors.size() == 1)
    {
        // 单输入单输出 (如 abs, leakyrelu, slice, tanh)
        // 常数输入 add div
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
                size_t block_size = blockIt->size;
                // 拆分内存块, 简化slice 算子的操作
                blockIt->size = outputTensorSize;
                blockIt->tensor_name = outputTensor;
                blockIt->releaseTime = tensor_lifetimes[outputTensor].end_time;
                memoryPool.insert(std::next(blockIt), MemoryBlock(block_size-outputTensorSize, false));
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
                it->releaseTime = tensor_lifetimes[outputTensor].end_time;
                // 更改concat的其余输入的Tensor的 end_time
                auto next = std::next(it);
                while (next != memoryPool.end() && std::find(inputTensors.begin(), inputTensors.end(), next->tensor_name) != inputTensors.end())
                {
                    next->releaseTime = tensor_lifetimes[outputTensor].end_time;
                    next = std::next(next);
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

void updateOffsets(size_t insertPosition, size_t blockSize, const std::string& outputTensor)
{
    std::vector<std::pair<size_t, std::string>> toBeUpdated;

    // 收集需要更新的元素
    for (auto it = tensorOffsets.lower_bound(insertPosition); it != tensorOffsets.end(); ++it)
    {
        toBeUpdated.push_back({it->first + blockSize, it->second});
    }

    // 删除旧的元素
    tensorOffsets.erase(tensorOffsets.lower_bound(insertPosition), tensorOffsets.end());

    // 插入更新后的元素和新的Tensor
    for (const auto& elem : toBeUpdated)
    {
        tensorOffsets.insert(elem);
    }
    
    // 插入新的Tensor
    tensorOffsets.insert(std::make_pair(insertPosition, outputTensor));
}

void updateTensorOffsets()
{
    size_t currentOffset = 0;

    for (const auto& block : memoryPool)
    {
        if (block.isAllocated)
        {
            // 获取当前偏移量下的所有条目的范围
            auto range = tensorOffsets.equal_range(currentOffset);
            bool found = false;
            for (auto it = range.first; it != range.second; ++it)
            {
                if (it->second == block.tensor_name)
                {
                    found = true; // 找到匹配的Tensor，无需更新
                    break;
                }
            }

            if (!found)
            {
                // 如果没有找到匹配的Tensor，添加新的条目
                tensorOffsets.insert(std::make_pair(currentOffset, block.tensor_name));
            }
        }
        currentOffset += block.size; // 更新偏移量，以包含当前块的大小
    }
}

void printTensorOffsets()
{
    std::cout << "Tensor Offsets:\n";
    std::cout << "---------------------\n";
    for (const auto& pair : tensorOffsets)
    {
        std::cout << "Tensor: " << pair.second << ", Offset: " << pair.first << '\n';
    }
    std::cout << "---------------------\n";
}