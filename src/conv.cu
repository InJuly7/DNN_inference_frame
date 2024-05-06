/*
 convolution supported by CUDA implements
 matrix is orgnized in 1D array
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>


extern "C" __global__ void Conv2d(const float *A, float *C, const float *weight, int *pads, int *edag, int *outshape, int *kshape, int *strides, float *pad_temp, int *input_shape)
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
        C[index] = output[threadIdx.x];
    }
}

extern "C" __global__ void Conv2db(const float *A, float *C, const float *weight, int *pads, int *edag, int *outshape, int *kshape, int *strides, float *pad_temp, int *input_shape, const float *bias)
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

extern "C" __global__ void Conv2dg(const float *A, float *C, float *weight, int *pads, int *edag, int *outshape, int *kshape, int *strides, float *pad_temp, int *input_shape,const float *bias)
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