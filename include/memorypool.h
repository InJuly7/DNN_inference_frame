#ifndef MEMORYPOOL_H
#define MEMORYPOOL_H

struct MemoryBlock
{
    bool isAllocated;
    size_t size;
    int releaseTime;  // -1 表示空闲

    MemoryBlock(size_t sz, bool allocated, int release = -1)
        : size(sz), isAllocated(allocated), releaseTime(release) {}
};

#endif // MEMORYPOOL_H