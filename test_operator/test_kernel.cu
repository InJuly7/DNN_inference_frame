#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <cstring>

#include "verify_kernel.h"

extern "C" __global__ void AddKernel_1(const float *A, const float add_const, float *C, int numElements);
extern "C" __global__ void AddKernel_2(const float *A, const float *B, float *C, int numElements);
extern "C" __global__ void LeakyRelu(const float *A, const float B, float *C, int numElements);
extern "C" __global__ void Abs(const float *A, float *C, int numElements);
extern "C" __global__ void Div(const float *A, float div_const, float *C, int numElements);
extern "C" __global__ void Tanh(const float *A, float *C, int numElements);


int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <kernel_function>" << std::endl;
        return 1;
    }

    float *h_A, *h_B, *h_C;  
    float *d_A, *d_B, *d_C;  
    int numElements = 1024;  
    
    size_t size = numElements * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    
    
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    
    

    if(strcmp(argv[1], "AddKernel_1") == 0)
    {
        h_C = (float*)malloc(size);
        cudaMalloc((void**)&d_C, size);
    
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        float add_const = 0.5;
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        AddKernel_1<<<blocksPerGrid, threadsPerBlock>>>(d_A, add_const, d_C, numElements);
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        verify_AddKernel_1(h_A,add_const,h_C,10);
    }
    else if(strcmp(argv[1], "AddKernel_2") == 0)
    {
        h_C = (float*)malloc(size);
        cudaMalloc((void**)&d_C, size);

        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        AddKernel_2<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        verify_AddKernel_2(h_A,h_B,h_C,10);
    }
    else if(strcmp(argv[1], "LeakyRelu") == 0)
    {
        h_C = (float*)malloc(size);
        cudaMalloc((void**)&d_C, size);

        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
               
        float alpha = 0.2;
        h_A[9] = -0.9;
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        LeakyRelu<<<blocksPerGrid, threadsPerBlock>>>(d_A, alpha, d_C, numElements);
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        verify_LeakyReluKernel(h_A,alpha,h_C,10);
    }
    else if(strcmp(argv[1], "Abs") == 0)
    {
        h_C = (float*)malloc(size);
        cudaMalloc((void**)&d_C, size);

        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        h_A[5] = -h_A[5];
        h_A[9] = -h_A[9];
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        Abs<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, numElements);
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        verify_AbsKernel(h_A,h_C,10);
    }
    else if(strcmp(argv[1], "Div") == 0)
    {
        h_C = (float*)malloc(size);
        cudaMalloc((void**)&d_C, size);

        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        float div_const = 2;
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        Div<<<blocksPerGrid, threadsPerBlock>>>(d_A, div_const, d_C, numElements);
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        verify_DivKernel(h_A,div_const,h_C,10);
    }
    else if (strcmp(argv[1], "Tanh") == 0)
    {
        h_C = (float*)malloc(size);
        cudaMalloc((void**)&d_C, size);
        
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        h_A[0] = 0;
        h_A[9] = 10;
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        Tanh<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, numElements);
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        verify_TanhKernel(h_A,h_C,10);
    }
    else if (strcmp(argv[1], "Slice") == 0)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
        Slice<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, 2, 5, 3, numElements);  // Example parameters
    }
    else if (strcmp(argv[1], "Concat") == 0)
    {   
        h_C = (float*)malloc(2*size);
        cudaMalloc((void**)&d_C, 2*size);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((16+threadsPerBlock.x-1) / threadsPerBlock.x,
                        (16+threadsPerBlock.y-1) / threadsPerBlock.y);

        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        ConcatNCHW<<<numBlocks,threadsPerBlock>>>(d_A,4,d_B,4,d_C,16,16);
        cudaMemcpy(h_C, d_C, 2*size, cudaMemcpyDeviceToHost);
        verify_ConcatKernel(h_A,h_B,h_C);
    }
    else
    {
        std::cerr << "Invalid kernel function specified." << std::endl;
        return 1;
    }

    
    // cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

