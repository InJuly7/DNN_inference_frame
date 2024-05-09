#include <iostream>
#include <list>

#include "memorypool.h"

extern std::list<MemoryBlock> memoryPool;
extern std::vector<std::string> topologicalOrder;



void MemoryPoolImplementation()
{
    for (const auto& operatorName : topologicalOrder)
    {
        if()
        std::vector<std::string> inputTensors = ;  // 根据算子确定输入张量
        std::string outputTensor = ...;  // 根据算子确定输出张量

        processOperator(operatorName, inputTensors, outputTensor);
        currentTime++;  // 假设每个算子处理增加时间
    }

}

void processOperator(const std::string& operator_name, const std::vector<std::string>& inputTensors, const std::string& outputTensor)
{
    if (operatorType == "Slice")
    {
        // 特殊处理Slice算子
        // 释放不再需要的Tensor部分
    }
    else if (operatorType == "Concat")
    {
        // 处理Concat算子，保证所有输入Tensor直到Concat完成都不被释放
    }
    else
    {
        // 处理常规计算型算子
        if (canTensorBeOverwritten(inputTensors[0], tensor_lifetimes[inputTensors[0]], dependencies))
        {
            // 如果输入Tensor可以被覆盖
            reuseMemory(inputTensors[0], outputTensor);
        }
        else 
        {
            allocateNewMemory(outputTensor);
        }
    }
}

bool canTensorBeOverwritten()
{
    if (tensorLife.special_flag) {  // 例如：如果标记为特殊类型（如concat），则可能不可覆盖
        return false;
    }

    if (dependencies.find(tensorName) != dependencies.end() && dependencies.at(tensorName).size() > 1) {
        return false;  // 如果Tensor被多个算子依赖
    }

    // 此处还可以增加其他判断逻辑

    return true;  // 如果没有触发上述条件，假设Tensor可覆盖
}




// 分配内存块
void allocateMemory(size_t size, int releaseTime)
{
    for (auto it = memoryPool.begin(); it != memoryPool.end(); ++it)
    {
        if (!it->isAllocated && it->size >= size)
        {  // 找到第一个足够大的空闲块
            if (it->size > size)
            {  // 如果块比需要的大，拆分它
                memoryPool.insert(it, MemoryBlock(size, true, releaseTime));
                it->size -= size;
            }
            else
            {  // 正好符合大小，直接分配
                it->isAllocated = true;
                it->releaseTime = releaseTime;
            }
            return;
        }
    }

    // 没有足够的空闲块，创建新块
    memoryPool.emplace_back(size, true, currentTime);
}

// 释放内存块
void freeMemory(int currentTime)
{
    for (auto it = memoryPool.begin(); it != memoryPool.end(); ++it)
    {
        if (it->isAllocated && it->releaseTime <= currentTime)
        {
            it->isAllocated = false;
            it->releaseTime = -1;
            // 尝试与前后块合并
            if (it != memoryPool.begin())
            {
                auto prev = std::prev(it);
                if (!prev->isAllocated)
                {
                    prev->size += it->size;
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

int main()
{
    // 模拟内存分配和释放
    allocateMemory(100, 0);  // 在时间0分配100字节
    allocateMemory(200, 0);  // 同时分配200字节
    freeMemory(10);  // 在时间10释放所有应该释放的内存

    // 打印内存池状态
    for (const auto& block : memoryPool)
    {
        std::cout << "Block size: " << block.size << ", Allocated: " << block.isAllocated << std::endl;
    }

    return 0;
}
