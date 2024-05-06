#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel function declaration
__global__ void Conv2d(const float *A, float *C, const float *weight, int *pads, int *edag, 
                                  int *outshape, int *kshape, int *strides, float *pad_temp, int *input_shape)
{
    __shared__ float weight_s[16*1*3*3];
    register int pads_1 = pads[1], pads_2 = pads[2];
    register int ek_0 = edag[0], ek_1 = edag[1], ek_2 = edag[2], ek_3 = edag[3];
    if (threadIdx.x < 16*1*3*3)
        weight_s[threadIdx.x] = weight[threadIdx.x];
    
    // pads_1 下边界(不包括) pads_2 右边界(不包括) pads[0] inC
    // 有效区域 ek_0 左边界 ek_1 右边界(不包括) ek_2 上边界 ek_3 下边界(不包括)
    // 二维块的线程组织结构
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

int main()
{
    // 输入Tensor大小
    int input_shape[4] = {1, 1, 5, 5}; // 批量大小、通道数、高度、宽度
    int input_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];

    // 分配输入数据并初始化
    float *h_input = new float[input_size];
    std::fill_n(h_input, input_size, 1.0f); // 全部初始化为1

    // 输出Tensor大小
    int outshape[4] = {1, 1, 5, 5}; // 批量大小、通道数、高度、宽度
    int output_size = outshape[0] * outshape[1] * outshape[2] * outshape[3];

    float *h_output = new float[output_size];

    // 卷积核大小
    int kshape[4] = {1, 1, 3, 3}; // 输出通道数、输入通道数、内核高度、内核宽度
    int kernel_size = kshape[0] * kshape[1] * kshape[2] * kshape[3];

    float *h_weight = new float[kernel_size];
    std::fill_n(h_weight, kernel_size, 1.0f); // 全部初始化为1

    // 其它参数
    int pads[4] = {1,1,1,1};
    int *pad = new int[3];
    pad[0] = input_shape[1];
    pad[1] = input_shape[2] + pads[0] + pads[1];
    pad[2] = input_shape[3] + pads[2] + pads[3];
    int *edag = new int[4];
    edag[0] = pads[0];
    edag[1] = input_shape[2] + pads[1];
    edag[2] = pads[2];
    edag[3] = input_shape[3] + pads[3];

    int strides[2] = {1, 1}; // 步长

    // 临时的padding后结果
    int pad_temp_size = pads[0] * pads[1] * pads[2];
    float *h_pad_temp = new float[pad_temp_size];

    // 在设备上分配
    float *d_input, *d_output, *d_weight, *d_pad_temp;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_weight, kernel_size * sizeof(float));
    cudaMalloc(&d_pad_temp, pad_temp_size * sizeof(float));

    int *d_pads, *d_edag, *d_outshape, *d_kshape, *d_strides, *d_input_shape;
    cudaMalloc(&d_pads, 3 * sizeof(int));
    cudaMalloc(&d_edag, 4 * sizeof(int));
    cudaMalloc(&d_outshape, 4 * sizeof(int));
    cudaMalloc(&d_kshape, 4 * sizeof(int));
    cudaMalloc(&d_strides, 2 * sizeof(int));
    cudaMalloc(&d_input_shape, 4 * sizeof(int));

    // 复制数据到设备
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pads, pad, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edag, edag, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outshape, outshape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kshape, kshape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, strides, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_shape, input_shape, 4 * sizeof(int), cudaMemcpyHostToDevice);

    // 调用CUDA kernel
    int block_size = 256; // 每个block中的线程数量
    int grid_size = (output_size + block_size - 1) / block_size; // 计算grid大小

    Conv2d<<<grid_size, block_size>>>(d_input, d_output, d_weight, d_pads, d_edag, d_outshape, d_kshape, d_strides, d_pad_temp, d_input_shape);

    // 从设备复制数据到host
    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印输出
    std::cout << "Output:\n";
    for (int i = 0; i < output_size; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // 释放资源
    delete[] h_input;
    delete[] h_output;
    delete[] h_weight;
    delete[] h_pad_temp;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weight);
    cudaFree(d_pad_temp);
    cudaFree(d_pads);
    cudaFree(d_edag);
    cudaFree(d_outshape);
    cudaFree(d_kshape);
    cudaFree(d_strides);
    cudaFree(d_input_shape);

    return 0;
}
