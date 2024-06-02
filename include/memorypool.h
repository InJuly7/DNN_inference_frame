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
size_t calculateOffset(std::list<MemoryBlock>::iterator blockIt);
void calculateTotalMemorySize();

void MemoryPoolImplementation();

void processOperator(const std::string &operator_name, const std::vector<std::string> &inputTensors, const std::string &outputTensor, int current_time);
void processOutputTensor(const std::string& operator_name, std::string outputTensor);
bool canTensorBeOverwritten(const std::string &operator_name, const std::vector<std::string> &inputTensors, int current_time);
void allocateMemory(size_t size, std::string tensor_name);
void freeMemory(int releaseTime);
void coverageMemory(const std::vector<std::string> &inputTensors, const std::string &outputTensor);
void printMemoryPool();
void updateOffsets(size_t insertPosition, size_t blockSize, const std::string &outputTensor);
void updateTensorOffsets();
void printTensorOffsets();



#endif // MEMORYPOOL_H


