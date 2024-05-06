#include <stdio.h>
#include <stdlib.h>

#define element_type float
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

/*
    @brief: 串行卷积实现 CPU代码 NCHW
    @param in inC inH inW: 输入矩阵(数组) channel height width
    @param out outC outH outW: 输出矩阵 channel height width
    @param kernel kernelH kernelW: 卷积核 height width
*/
void serial_convolution(element_type *in, element_type *out, element_type *kernel,
                        int inC, int inH, int inW,
                        int outC, int outH, int outW,
                        int kernelH, int kernelW)
{
    float val;
    int out_pos, in_pos, kernel_pos;
    for (int oc = 0; oc < outC; oc++) // 每个输出通道
    {
        // 对一个位置的操作 用当前输入channel卷积去对相应的输出channel
        // 保证每个outChannel都是多inChannel累积的结果
        for (int i = 0; i < outH; i++)
        {
            for (int j = 0; j < outW; j++)
            {
                val = 0; // 避免累积和需要多次读取写入
                out_pos = oc * outH * outW + OFFSET(i, j, outW);
                for (int ic = 0; ic < inC; ic++) // 对每个输入通道
                {
                    // 卷积运算
                    for (int ii = 0; ii < kernelH; ii++)
                    {
                        for (int jj = 0; jj < kernelW; jj++)
                        {
                            in_pos = ic * inH * inW + OFFSET(i + ii, j + jj, inW);
                            kernel_pos = oc * ic * kernelH * kernelW + ic * kernelH * kernelW + OFFSET(ii, jj, kernelW);
                            val += in[in_pos] * kernel[kernel_pos];
                        }
                    }
                }
                out[out_pos] = val; // 与cudnn计算结果为相反数
            }
        }
    }
}

int main()
{
    // int inC = 1, inH = 7, inW = 7;
    // int kernelH = 3, kernelW = 3;
    // int outC = 1, outH = 5, outW = 5;
    // int batch_size = 1;

    // // 分配内存
    // float *input = (float *)malloc(inC * inH * inW * sizeof(float));
    // float *output = (float *)malloc(outC * outH * outW * sizeof(float));
    // float *kernel = (float *)malloc(inC * outC * kernelH * kernelW * sizeof(float));

    // // 初始化输入和卷积核
    // for (int i = 0; i < inC * inH * inW; i++)
    // {
    //     input[i] = 1.0; // 可以改为其他值或随机数以便测试
    // }
    // for (int i = 0; i < inC * outC * kernelH * kernelW; i++)
    // {
    //     kernel[i] = 1.0; // 单位卷积核
    // }

    // // 调用卷积函数
    // serial_convolution(input, output, kernel, inC, inH, inW, outC, outH, outW, kernelH, kernelW);

    // // 打印输出矩阵
    // printf("Output:\n");
    // for (int i = 0; i < outC; i++)
    // {
    //     for (int j = 0; j < outH; j++)
    //     {
    //         for (int k = 0; k < outW; k++)
    //         {
    //             printf("%.1f ", output[i*outH*outW + j*outW + k]);
    //         }
    //         printf("\n");
    //     }
    // }

    // // 释放内存
    // free(input);
    // free(output);
    // free(kernel);

    // return 0;
    
}