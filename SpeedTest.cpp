#include "cuda.h"
#include "cuda_runtime.h"
#include <random>
#include <chrono>
#include "OpKernel.cu"

#define NUMBER 20

extern "C"{
    void cuda_add_2(float *a, int numsize);
    void cuda_add_1(float *a, int numsize, float addinit);
    void cuda_leaky_1(float *a, int numsize, float leakyinit);
    void cuda_abs_1(float *a, int numsize);
    void cuda_div_1(float *a, int numsize, float divinit);
    void cuda_tanh_1(float *a, int numsize);
    void cuda_slice_1(float *a, int *numsize, int *argc);
    void cuda_concat_1(float *a, int **input_shape, int rows, int *cols, int aixs, int *output_shape);
    void cuda_conv2d_1(float *a, int group, int *input_shape, int *output_shape, int *kernel_shape, 
                        int *pads, int *stride, float *weight);
    void cuda_conv2d_2(float *a, int group, int *input_shape, int *output_shape, int *kernel_shape, 
                        int *pads, int *stride, float *weight, float *bias);
}

void cuda_add_2(float *a, int numsize)
{
    float *A = new float[numsize];
    float *B = new float[numsize];
    float *C = new float[numsize];
    float maxtime = 0.0;
    std::default_random_engine e(time(0));
    std::uniform_real_distribution<float> u(-1,1);
    for (int i = 0; i < numsize; ++i)
    {
        A[i] = u(e);
        B[i] = u(e);
    }
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, numsize * sizeof(float));
    cudaMalloc((void**)&d_b, numsize * sizeof(float));
    cudaMalloc((void**)&d_c, numsize * sizeof(float));
    cudaMemcpy(d_a, A, numsize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, numsize * sizeof(float), cudaMemcpyHostToDevice);
    AddKernel_2<<<(numsize + 255)/256, 256>>> (d_a, d_b, d_c, numsize);
    cudaDeviceSynchronize();
    for (int i = 0; i < NUMBER; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        AddKernel_2<<<(numsize + 255)/256, 256>>> (d_a, d_b, d_c, numsize);
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        float duration_seconds = static_cast<float>(duration);
        if (duration_seconds > maxtime)
            maxtime = duration_seconds;
    }
    cudaMemcpy(C, d_c, numsize * sizeof(float), cudaMemcpyDeviceToHost);
    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    a[0] = maxtime;
}

void cuda_add_1(float *a, int numsize, float addinit)
{
    float *A = new float[numsize];
    float *C = new float[numsize];
    float maxtime = 0.0;
    std::default_random_engine e(time(0));
    std::uniform_real_distribution<float> u(-1,1);
    for (int i = 0; i < numsize; ++i)
    {
        A[i] = u(e);
    }
    float *d_a, *d_c;
    cudaMalloc((void**)&d_a, numsize * sizeof(float));
    cudaMalloc((void**)&d_c, numsize * sizeof(float));
    cudaMemcpy(d_a, A, numsize * sizeof(float), cudaMemcpyHostToDevice);
    AddKernel_1<<<(numsize + 255)/256, 256>>> (d_a, addinit, d_c, numsize);
    cudaDeviceSynchronize();
    for (int i = 0; i < NUMBER; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        AddKernel_1<<<(numsize + 255)/256, 256>>> (d_a, addinit, d_c, numsize);
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        float duration_seconds = static_cast<float>(duration);
        if (duration_seconds > maxtime)
            maxtime = duration_seconds;
    }
    cudaMemcpy(C, d_c, numsize * sizeof(float), cudaMemcpyDeviceToHost);
    delete[] A;
    delete[] C;
    cudaFree(d_a);
    cudaFree(d_c);
    a[0] = maxtime;
}

void cuda_leaky_1(float *a, int numsize, float leakyinit)
{
    float *A = new float[numsize];
    float *C = new float[numsize];
    float maxtime = 0.0;
    std::default_random_engine e(time(0));
    std::uniform_real_distribution<float> u(-1,1);
    for (int i = 0; i < numsize; ++i)
    {
        A[i] = u(e);
    }
    float *d_a, *d_c;
    cudaMalloc((void**)&d_a, numsize * sizeof(float));
    cudaMalloc((void**)&d_c, numsize * sizeof(float));
    cudaMemcpy(d_a, A, numsize * sizeof(float), cudaMemcpyHostToDevice);
    LeakyRelu<<<(numsize + 255)/256, 256>>> (d_a, leakyinit, d_c, numsize);
    cudaDeviceSynchronize();
    for (int i = 0; i < NUMBER; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        LeakyRelu<<<(numsize + 256)/256, 256>>> (d_a, leakyinit, d_c, numsize);
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        float duration_seconds = static_cast<float>(duration);
        a[i] = duration_seconds;
        if (duration_seconds > maxtime)
            maxtime = duration_seconds;
    }
    cudaMemcpy(C, d_c, numsize * sizeof(float), cudaMemcpyDeviceToHost);
    delete[] A;
    delete[] C;
    cudaFree(d_a);
    cudaFree(d_c);
    a[0] = maxtime;
}

void cuda_abs_1(float *a, int numsize)
{
    float *A = new float[numsize];
    float *C = new float[numsize];
    float maxtime = 0.0;
    std::default_random_engine e(time(0));
    std::uniform_real_distribution<float> u(-1,1);
    for (int i = 0; i < numsize; ++i)
    {
        A[i] = u(e);
    }
    float *d_a, *d_c;
    cudaMalloc((void**)&d_a, numsize * sizeof(float));
    cudaMalloc((void**)&d_c, numsize * sizeof(float));
    cudaMemcpy(d_a, A, numsize * sizeof(float), cudaMemcpyHostToDevice);
    Abs<<<(numsize + 255)/256, 256>>> (d_a, d_c, numsize);
    cudaDeviceSynchronize();
    for (int i = 0; i < NUMBER; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        Abs<<<(numsize + 255)/256, 256>>> (d_a, d_c, numsize);
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        float duration_seconds = static_cast<float>(duration);
        if (duration_seconds > maxtime)
            maxtime = duration_seconds;
    }
    cudaMemcpy(C, d_c, numsize * sizeof(float), cudaMemcpyDeviceToHost);
    delete[] A;
    delete[] C;
    cudaFree(d_a);
    cudaFree(d_c);
    a[0] = maxtime;
}

void cuda_div_1(float *a, int numsize, float divinit)
{
    float *A = new float[numsize];
    float *C = new float[numsize];
    float maxtime = 0.0;
    std::default_random_engine e(time(0));
    std::uniform_real_distribution<float> u(-1,1);
    for (int i = 0; i < numsize; ++i)
    {
        A[i] = u(e);
    }
    float *d_a, *d_c;
    cudaMalloc((void**)&d_a, numsize * sizeof(float));
    cudaMalloc((void**)&d_c, numsize * sizeof(float));
    cudaMemcpy(d_a, A, numsize * sizeof(float), cudaMemcpyHostToDevice);
    Div<<<(numsize + 255)/256, 256>>> (d_a, divinit, d_c, numsize);
    cudaDeviceSynchronize();
    for (int i = 0; i < NUMBER; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        Div<<<(numsize + 255)/256, 256>>> (d_a, divinit, d_c, numsize);
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        float duration_seconds = static_cast<float>(duration);
        if (duration_seconds > maxtime)
            maxtime = duration_seconds;
    }
    cudaMemcpy(C, d_c, numsize * sizeof(float), cudaMemcpyDeviceToHost);
    delete[] A;
    delete[] C;
    cudaFree(d_a);
    cudaFree(d_c);
    a[0] = maxtime;
}

void cuda_tanh_1(float *a, int numsize)
{
    float *A = new float[numsize];
    float *C = new float[numsize];
    float maxtime = 0.0;
    std::default_random_engine e(time(0));
    std::uniform_real_distribution<float> u(-1,1);
    for (int i = 0; i < numsize; ++i)
    {
        A[i] = u(e);
    }
    float *d_a, *d_c;
    cudaMalloc((void**)&d_a, numsize * sizeof(float));
    cudaMalloc((void**)&d_c, numsize * sizeof(float));
    cudaMemcpy(d_a, A, numsize * sizeof(float), cudaMemcpyHostToDevice);
    Tanh<<<(numsize + 255)/256, 256>>> (d_a, d_c, numsize);
    cudaDeviceSynchronize();
    for (int i = 0; i < NUMBER; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        Tanh<<<(numsize + 255)/256, 256>>> (d_a, d_c, numsize);
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        float duration_seconds = static_cast<float>(duration);
        if (duration_seconds > maxtime)
            maxtime = duration_seconds;
    }
    cudaMemcpy(C, d_c, numsize * sizeof(float), cudaMemcpyDeviceToHost);
    delete[] A;
    delete[] C;
    cudaFree(d_a);
    cudaFree(d_c);
    a[0] = maxtime;
}

void cuda_slice_1(float *a, int *numsize, int *argc)
{
    float maxtime = 0.0;
    int NumSize = 1;
    for (int i = 0; i < 4; ++i)
    {
        NumSize *= numsize[i];
    }
    float *A = new float[NumSize];
    std::default_random_engine e(time(0));
    std::uniform_real_distribution<float> u(-1,1);
    for (int i = 0; i < NumSize; ++i)
    {
        A[i] = u(e);
    }
    int start = argc[0];
    int end = argc[1];
    int axes = argc[2];
    int step = argc[3];
    int num_block = 1;
    for (int i = 0; i < axes; ++i)
    {
        num_block *= numsize[i];
    }
    int unit = 1;
    for (int i = axes+1; i < 4; ++i)
    {
        unit *= numsize[i];
    }
    int src_block_size = unit * numsize[axes];
    int dst_block_size = unit * (end - start);
    int src_offset = unit * start;
    float *d_a, *d_c;
    cudaMalloc((void**)&d_a, NumSize * sizeof(float));
    cudaMalloc((void**)&d_c, num_block * dst_block_size * sizeof(float));
    cudaMemcpy(d_a, A, NumSize * sizeof(float), cudaMemcpyHostToDevice);
    Slice<<<num_block, 256>>>(d_a, d_c, src_block_size, dst_block_size, src_offset, num_block);
    for (int i = 0; i < NUMBER; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        Slice<<<num_block, 256>>>(d_a, d_c, src_block_size, dst_block_size, src_offset, num_block);
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        float duration_seconds = static_cast<float>(duration);
        if (duration_seconds > maxtime)
            maxtime = duration_seconds;
    }
    float *C = new float[num_block * dst_block_size];
    cudaMemcpy(C, d_c, num_block * dst_block_size * sizeof(float), cudaMemcpyDeviceToHost);
    delete[] A;
    delete[] C;
    cudaFree(d_a);
    cudaFree(d_c);
    a[0] = maxtime;
}

//Concat简易版
void cuda_concat_1(float *a, int **input_shape, int rows, int *cols, int aixs, int *output_shape)
{
    int Numsize = 1;
    for (int i = 0; i < 4; ++i)
    {
        Numsize *= output_shape[i];
    }
    float *A = new float[Numsize];
    float *C = new float[Numsize];
    float maxtime = 0.0;
    std::default_random_engine e(time(0));
    std::uniform_real_distribution<float> u(-1,1);
    for (int i = 0; i < Numsize; ++i)
    {
        A[i] = u(e);
    }
    float *d_a, *d_c;
    cudaMalloc((void**)&d_a, Numsize * sizeof(float));
    cudaMalloc((void**)&d_c, Numsize * sizeof(float));
    cudaMemcpy(d_a, A, Numsize * sizeof(float), cudaMemcpyHostToDevice);
    Concat<<<(Numsize+255)/256, 256>>> (d_a, d_c, Numsize);
    for (int i = 0; i < NUMBER; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        Concat<<<(Numsize+255)/256, 256>>> (d_a, d_c, Numsize);
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        float duration_seconds = static_cast<float>(duration);
        if (duration_seconds > maxtime)
            maxtime = duration_seconds;
    }
    cudaMemcpy(C, d_c, Numsize * sizeof(float), cudaMemcpyDeviceToHost);
    delete[] A;
    delete[] C;
    cudaFree(d_a);
    cudaFree(d_c);
    a[0] = maxtime;
}

void cuda_conv2d_1(float *a, int group, int *input_shape, int *output_shape, int *kernel_shape, 
                        int *pads, int *stride, float *weight, float *bias)
{
    int inNumsize = 1;
    int outNumsize = 1;
    for (int i = 0; i < 4; ++i)
    {
        inNumsize *= input_shape[i];
        outNumsize *= output_shape[i];
    }
    float *A = new float[inNumsize];
    float *C = new float[outNumsize];
    float maxtime = 0.0;
    std::default_random_engine e(time(0));
    std::uniform_real_distribution<float> u(-1,1);
    for (int i = 0; i < inNumsize; ++i)
    {
        A[i] = u(e);
    }
    float *pad = new float[3];
    pad[0] = input_shape[1];
    pad[1] = input_shape[2] + pads[0] + pads[1];
    pad[2] = input_shape[3] + pads[2] + pads[3];
    float *edag = new float[4];
    edag[0] = pads[0];
    edag[1] = input_shape[2] + pads[1];
    edag[2] = pads[2];
    edag[3] = input_shape[3] + pads[3];
    float *d_a, *d_c, *d_weight, *pad_temp;
    int *d_pads, *d_edag, *d_outshape, *d_kshape, *d_strides;
    cudaMalloc((void**)&d_a, inNumsize * sizeof(float));
    cudaMalloc((void**)&d_c, outNumsize * sizeof(float));
    cudaMalloc((void**)&d_weight, kernel_shape[0] * kernel_shape[1] *kernel_shape[2] *kernel_shape[3] * sizeof(float));
    cudaMalloc((void**)&d_pads, 3 * sizeof(int));
    cudaMalloc((void**)&d_edag, 4 * sizeof(int));
    cudaMalloc((void**)&d_outshape, 4 * sizeof(int));
    cudaMalloc((void**)&d_kshape, 4 * sizeof(int));
    cudaMalloc((void**)&d_strides, 2 * sizeof(int));
    cudaMalloc((void**)&pad_temp, pad[0]*pad[1]*pad[2] * sizeof(float));

    cudaMemcpy(d_a, A, inNumsize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, kernel_shape[0] * kernel_shape[1] *kernel_shape[2] *kernel_shape[3] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pads, pad, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edag, edag, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outshape, output_shape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kshape, kernel_shape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, stride, 2 * sizeof(int), cudaMemcpyHostToDevice);
    if (group == 1)
        Conv2d<<<256,256>>>(d_a, d_c, d_weight, d_pads, d_edag, d_outshape, d_kshape, d_strides, pad_temp, input_shape);
    else 
        Conv2dg<<<256,256>>>(d_a, d_c, d_weight, d_pads, d_edag, d_outshape, d_kshape, d_strides, pad_temp, input_shape, bias);
    for (int i = 0; i < NUMBER; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        if (group == 1)
            Conv2d<<<256,256>>>(d_a, d_c, d_weight, d_pads, d_edag, d_outshape, d_kshape, d_strides, pad_temp, input_shape);
        else
            Conv2dg<<<256,256>>>(d_a, d_c, d_weight, d_pads, d_edag, d_outshape, d_kshape, d_strides, pad_temp, input_shape, bias);
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        float duration_seconds = static_cast<float>(duration);
        if (duration_seconds > maxtime)
            maxtime = duration_seconds;
    }
    cudaMemcpy(C, d_c,  outNumsize * sizeof(float), cudaMemcpyDeviceToHost);
    delete[] A;
    delete[] C;
    cudaFree(d_a);
    cudaFree(d_c);
    a[0] = maxtime;
}

void cuda_conv2d_2(float *a, int group, int *input_shape, int *output_shape, int *kernel_shape, 
                        int *pads, int *stride, float *weight, float *bias)
{
    int inNumsize = 1;
    int outNumsize = 1;
    for (int i = 0; i < 4; ++i)
    {
        inNumsize *= input_shape[i];
        outNumsize *= output_shape[i];
    }
    float *A = new float[inNumsize];
    float *C = new float[outNumsize];
    float maxtime = 0.0;
    std::default_random_engine e(time(0));
    std::uniform_real_distribution<float> u(-1,1);
    for (int i = 0; i < inNumsize; ++i)
    {
        A[i] = u(e);
    }
    float *pad = new float[3];
    pad[0] = input_shape[1];
    pad[1] = input_shape[2] + pads[0] + pads[1];
    pad[2] = input_shape[3] + pads[2] + pads[3];
    float *edag = new float[4];
    edag[0] = pads[0];
    edag[1] = input_shape[2] + pads[1];
    edag[2] = pads[2];
    edag[3] = input_shape[3] + pads[3];
    float *d_a, *d_c, *d_weight, *d_bias, *pad_temp;
    int *d_pads, *d_edag, *d_outshape, *d_kshape, *d_strides;
    cudaMalloc((void**)&d_a, inNumsize * sizeof(float));
    cudaMalloc((void**)&d_c, outNumsize * sizeof(float));
    cudaMalloc((void**)&d_weight, kernel_shape[0] * kernel_shape[1] *kernel_shape[2] *kernel_shape[3] * sizeof(float));
    cudaMalloc((void**)&d_pads, 3 * sizeof(int));
    cudaMalloc((void**)&d_edag, 4 * sizeof(int));
    cudaMalloc((void**)&d_outshape, 4 * sizeof(int));
    cudaMalloc((void**)&d_kshape, 4 * sizeof(int));
    cudaMalloc((void**)&d_strides, 2 * sizeof(int));
    cudaMalloc((void**)&d_bias, output_shape[1] * sizeof(float));
    cudaMalloc((void**)&pad_temp, pad[0]*pad[1]*pad[2] * sizeof(float));

    cudaMemcpy(d_a, A, inNumsize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, kernel_shape[0] * kernel_shape[1] *kernel_shape[2] *kernel_shape[3] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pads, pad, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edag, edag, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outshape, output_shape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kshape, kernel_shape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, stride, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, output_shape[1] * sizeof(float), cudaMemcpyHostToDevice);
    Conv2db<<<256,256>>>(d_a, d_c, d_weight, d_pads, d_edag, d_outshape, d_kshape, d_strides, pad_temp, input_shape, d_bias);
    for (int i = 0; i < NUMBER; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        Conv2db<<<256,256>>>(d_a, d_c, d_weight, d_pads, d_edag, d_outshape, d_kshape, d_strides, pad_temp, input_shape, d_bias);
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        float duration_seconds = static_cast<float>(duration);
        if (duration_seconds > maxtime)
            maxtime = duration_seconds;
    }
    cudaMemcpy(C, d_c,  outNumsize * sizeof(float), cudaMemcpyDeviceToHost);
    delete[] A;
    delete[] C;
    cudaFree(d_a);
    cudaFree(d_c);
    a[0] = maxtime;
}