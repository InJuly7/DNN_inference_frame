#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "OpKernel.cuh"


extern "C" __global__ void AddKernel_1(const float *A, const float add_const, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + add_const;
    }
}

extern "C" __global__ void AddKernel_2(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

extern "C" __global__ void Div(const float *A, const float div_const, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] / div_const;
    }
}

extern "C" __global__ void LeakyRelu(const float *A, const float alpha, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        float val = A[i];
        C[i] = (val < 0.0f) ? alpha*val: val;
    }
}

extern "C" __global__ void Abs(const float *A, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = fabs(A[i]);
    }
}

extern "C" __global__ void Tanh(const float *A, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        float val = A[i];
        C[i] = (expf(2.0f * val) - 1.0f) / (expf(2.0f * val) + 1.0f);
    }
}

extern "C" __global__ void Slice(const float *A, float *C, int src_block_size, int dst_block_size, 
                                                                            int offset, int num_blocks)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (num_blocks * dst_block_size); index += blockDim.x * gridDim.x)
    {
        int chunk = index % dst_block_size;
        int block = index / dst_block_size;
        C[index] = A[block * src_block_size + chunk + offset];
    }
}

// extern "C" __global__ void Concat(const float *A, float *C, int numElements)
// {
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     if (i < numElements)
//     {
//         C[i] = A[i];
//     }
// }

extern "C" __global__ void ConcatNCHW(const float* A, int A_C, const float* B, int B_C, float* C,int H, int W)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x; 
    int iy = threadIdx.y + blockIdx.y * blockDim.y; 

    if (ix<W && iy<H)
    {
        for (int c = 0; c < A_C + B_C; c++)
        {
            int idx = c*H*W + iy*W + ix;
            if (c < A_C)
            {
                C[idx] = A[c*H*W + iy*W + ix];
            }
            else
            {
                C[idx] = B[(c-A_C)*H*W + iy*W + ix];
            }
        }
    }
}

extern "C" __global__ void Conv2d(const float *A, float *C, const float *weight, int *pads, int *edag, 
                            int *outshape, int *kshape, int *strides, float *pad_temp, int *input_shape)
{
    __shared__ float weight_s[16*1*3*3];
    register int pads_1 = pads[1], pads_2 = pads[2];
    register int ek_0 = edag[0], ek_1 = edag[1], ek_2 = edag[2], ek_3 = edag[3];
    if (threadIdx.x < 16*1*3*3)
        weight_s[threadIdx.x] = weight[threadIdx.x];
    
    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < pads[0]*pads_1*pads_2; index += blockDim.x * gridDim.x)
    {
        int ic = index / (pads_1 * pads_2);
        int index_p = index % (pads_1 * pads_2);
        int pads_h = index_p / pads_2;
        int pads_w = index_p % pads_2;
        pad_temp[index] = 
            ((ek_0 <= pads_h) && (pads_h < ek_1) && (ek_2 <= pads_w) && (pads_w < ek_3))
            ? A[((((ic * input_shape[2] * input_shape[3]) + ((pads_h - ek_0) * input_shape[3]) + pads_w - ek_2)))] : 0.0f;
    }
    __syncthreads();
    
    __shared__ float output[256];
    ek_0 = kshape[0];
    ek_1 = kshape[1];
    ek_2 = kshape[2];
    ek_3 = kshape[3];
    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < outshape[1] * outshape[2] * outshape[3]; index += blockDim.x * gridDim.x)
    {
        int oc = index / (outshape[2] * outshape[3]);
        int index_o = index % (outshape[2] * outshape[3]);
        int oh = index_o / outshape[3];
        int ow = index_o % outshape[3];
        output[threadIdx.x] = 0.0f;
        for (int i = 0; i < ek_1; ++i)
        {
            for (int j = 0; j < ek_2; ++j)
            {
                for (int k = 0; k < ek_3; ++k)
                {
                    //if ((i == 0) && (j == 0) && (k == 0)) output[threadIdx.x] = 0.0f;
                    output[threadIdx.x] = output[threadIdx.x] + weight_s[(oc*ek_1*ek_2*ek_3) + (i*ek_2*ek_3) + (j*ek_3) + k] *
                               pad_temp[(i*pads_1*pads_2) + ((oh + j)*pads_2) + (ow + k)];
                }
            }
        }
        
        C[index] = output[threadIdx.x];
    }
}

extern "C" __global__ void Conv2dg(const float *A, float *C, float *weight, int *pads, int *edag, 
        int *outshape, int *kshape, int *strides, float *pad_temp, int *input_shape, const float *bias)
{
    __shared__ float weight_s[16*1*3*3];
    register int pads_1 = pads[1], pads_2 = pads[2];
    register int ek_0 = edag[0], ek_1 = edag[1], ek_2 = edag[2], ek_3 = edag[3];
    if (threadIdx.x < 16*1*3*3)
        weight_s[threadIdx.x] = weight[threadIdx.x];
    
    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < pads[0]*pads_1*pads_2; index += blockDim.x * gridDim.x)
    {
        int ic = index / (pads_1 * pads_2);
        int index_p = index % (pads_1 * pads_2);
        int pads_h = index_p / pads_2;
        int pads_w = index_p % pads_2;
        pad_temp[index] = 
            ((ek_0 <= pads_h) && (pads_h < ek_1) && (ek_2 <= pads_w) && (pads_w < ek_3))
            ? A[((((ic * input_shape[2] * input_shape[3]) + ((pads_h - ek_0) * input_shape[3]) + pads_w - ek_2)))] : 0.0f;
    }
    __syncthreads();
    
    __shared__ float output[256];
    ek_0 = kshape[0];
    ek_1 = kshape[1];
    ek_2 = kshape[2];
    ek_3 = kshape[3];
    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < outshape[1] * outshape[2] * outshape[3]; index += blockDim.x * gridDim.x)
    {
        int oc = index / (outshape[2] * outshape[3]);
        int index_o = index % (outshape[2] * outshape[3]);
        int oh = index_o / outshape[3];
        int ow = index_o % outshape[3];
        output[threadIdx.x] = 0.0f;
        for (int i = 0; i < ek_1; ++i)
        {
            for (int j = 0; j < ek_2; ++j)
            {
                for (int k = 0; k < ek_3; ++k)
                {
                    //if ((i == 0) && (j == 0) && (k == 0)) output[threadIdx.x] = 0.0f;
                    output[threadIdx.x] = output[threadIdx.x] + weight_s[(oc*ek_1*ek_2*ek_3) + (i*ek_2*ek_3) + (j*ek_3) + k] *
                               pad_temp[(i*pads_1*pads_2) + ((oh + j)*pads_2) + (ow + k)];
                }
            }
        }
        /*
        for (int i = 0; i < kshape[1]; ++i)
        {
            for (int j = 0; j < kshape[2]; ++j)
            {
                for (int k = 0; k < kshape[3]; ++k)
                {
                    if ((i == 0) && (j == 0) && (k == 0)) output[threadIdx.x] = 0.0f;
                    output[threadIdx.x] = output[threadIdx.x] + weight_s[(oc*kshape[1]*kshape[2]*kshape[3]) + (i*kshape[2]*kshape[3]) + (j*kshape[3]) + k]* 
                                pad_temp[(i*pads_1*pads_2) + ((oh*strides[0] + j)*pads_2) + (ow*strides[1] + k)];
                }
            }
        }
        */
        C[index] = output[threadIdx.x] + bias[oc];
    }
}

extern "C" __global__ void Conv2db(const float *A, float *C, const float *weight, int *pads, int *edag,
        int *outshape, int *kshape, int *strides, float *pad_temp, int *input_shape, const float *bias)
{
    
    __shared__ float weight_s[16*1*3*3];
    register int pads_1 = pads[1], pads_2 = pads[2];
    register int ek_0 = edag[0], ek_1 = edag[1], ek_2 = edag[2], ek_3 = edag[3];
    if (threadIdx.x < 16*1*3*3)
        weight_s[threadIdx.x] = weight[threadIdx.x];
    
    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < pads[0]*pads_1*pads_2; index += blockDim.x * gridDim.x)
    {
        int ic = index / (pads_1 * pads_2);
        int index_p = index % (pads_1 * pads_2);
        int pads_h = index_p / pads_2;
        int pads_w = index_p % pads_2;
        pad_temp[index] = 
            ((ek_0 <= pads_h) && (pads_h < ek_1) && (ek_2 <= pads_w) && (pads_w < ek_3))
            ? A[((((ic * input_shape[2] * input_shape[3]) + ((pads_h - ek_0) * input_shape[3]) + pads_w - ek_2)))] : 0.0f;
    }
    __syncthreads();
    
    __shared__ float output[256];
    ek_0 = kshape[0];
    ek_1 = kshape[1];
    ek_2 = kshape[2];
    ek_3 = kshape[3];
    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < outshape[1] * outshape[2] * outshape[3]; index += blockDim.x * gridDim.x)
    {
        int oc = index / (outshape[2] * outshape[3]);
        int index_o = index % (outshape[2] * outshape[3]);
        int oh = index_o / outshape[3];
        int ow = index_o % outshape[3];
        output[threadIdx.x] = 0.0f;
        for (int i = 0; i < ek_1; ++i)
        {
            for (int j = 0; j < ek_2; ++j)
            {
                for (int k = 0; k < ek_3; ++k)
                {
                    //if ((i == 0) && (j == 0) && (k == 0)) output[threadIdx.x] = 0.0f;
                    output[threadIdx.x] = output[threadIdx.x] + weight_s[(oc*ek_1*ek_2*ek_3) + (i*ek_2*ek_3) + (j*ek_3) + k] *
                               pad_temp[(i*pads_1*pads_2) + ((oh + j)*pads_2) + (ow + k)];
                }
            }
        }
        /*
        for (int i = 0; i < kshape[1]; ++i)
        {
            for (int j = 0; j < kshape[2]; ++j)
            {
                for (int k = 0; k < kshape[3]; ++k)
                {
                    if ((i == 0) && (j == 0) && (k == 0)) output[threadIdx.x] = 0.0f;
                    output[threadIdx.x] = output[threadIdx.x] + weight_s[(oc*kshape[1]*kshape[2]*kshape[3]) + (i*kshape[2]*kshape[3]) + (j*kshape[3]) + k]* 
                                pad_temp[(i*pads_1*pads_2) + ((oh*strides[0] + j)*pads_2) + (ow*strides[1] + k)];
                }
            }
        }
        */
       if(bias != NULL) C[index] = output[threadIdx.x] + bias[oc];
       else C[index] = output[threadIdx.x];

    }
}



void LeakyReluKernel(const float *A, const float alpha, float *C, int numElements)
{
    dim3 dimBlock = (256);
    dim3 dimGrid = ((numElements+256-1)/256);
    LeakyRelu<<<dimGrid,dimBlock>>>(A,alpha,C,numElements);
}

void TanhKernel(const float *A, float *C, int numElements)
{
    dim3 dimBlock = (256);
    dim3 dimGrid = ((numElements+256-1)/256);
    Tanh<<<dimGrid,dimBlock>>>(A,C,numElements);
    cudaDeviceSynchronize();
}

void AbsKernel(const float *A, float *C, int numElements)
{
    dim3 dimBlock = (256);
    dim3 dimGrid = ((numElements+256-1)/256);
    Abs<<<dimGrid,dimBlock>>>(A,C,numElements);
    cudaDeviceSynchronize();
}

void DivKernel(const float *A, float *C, const float div_const, int numElements)
{
    dim3 dimBlock = (256);
    dim3 dimGrid = ((numElements+256-1)/256);
    Div<<<dimGrid,dimBlock>>>(A,div_const,C,numElements);
    cudaDeviceSynchronize();
}

void AddKernel(const float *A, const float *B, float *C, const float add_const, int numElements)
{
    dim3 dimBlock = (256);
    dim3 dimGrid = ((numElements+256-1)/256);
    
    if(add_const == 0)
    {
        AddKernel_2<<<dimGrid,dimBlock>>>(A,B,C,numElements);
    }
    else
    {
        AddKernel_1<<<dimGrid,dimBlock>>>(A,add_const,C,numElements);
    }
    cudaDeviceSynchronize();
}

void ConvKernel(const float *A, float *C, float *weight, int *pads, int *edag, 
        int *outshape, int *kshape, int *strides, float *pad_temp, int *input_shape, int group,const float *bias)
{
    dim3 dimBlock = (256);
    int output_size = outshape[0] * outshape[1] * outshape[2] * outshape[3];
    dim3 dimGrid = ((output_size+256-1)/256);

    if(group != 1)
    {
        Conv2dg<<<dimGrid,dimBlock>>>(A,C,weight,pads,edag,outshape,kshape,strides,pad_temp,input_shape,bias);
    }
    else
    {
        Conv2db<<<dimGrid,dimBlock>>>(A,C,weight,pads,edag,outshape,kshape,strides,pad_temp,input_shape,bias);
    }
}