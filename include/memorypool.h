#ifndef MEMORYPOOL_H
#define MEMORYPOOL_H

struct MemoryBlock
{
    bool isAllocated;
    size_t size;
    int releaseTime;  // -1 表示空闲
    std::string tensor_name;  // 存储与此内存块关联的Tensor名称

    MemoryBlock(size_t sz, bool allocated, int release = -1, std::string name = "")
        : size(sz), isAllocated(allocated), releaseTime(release), tensor_name(name) {}
};


std::list<MemoryBlock>::iterator findBlockByName(const std::string &name);
void MemoryPoolImplementation();
void processOperator(const std::string &operator_name, const std::vector<std::string> &inputTensors, const std::string &outputTensor, int current_time);
bool canTensorBeOverwritten(const std::string &operator_name, const std::vector<std::string> &inputTensors, int current_time);
void processOutputTensor(const std::string &operator_name, const std::string &outputTensor);
void allocateMemory(size_t size, std::string tensor_name);
void freeMemory(int releaseTime);
void coverageMemory(const std::vector<std::string> &inputTensors, const std::string &outputTensor);
void printMemoryPool();

#endif // MEMORYPOOL_H


