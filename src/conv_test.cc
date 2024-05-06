#include <sys/time.h>
#include <iostream>
#include "cuda.h"
#include "cudnn.h"
#include "dldnn_ext.h"
#include "cu_ext.h"
#include <chrono>
#include <fstream>
#include <string.h>
#include <random>

int main()
{
    std::srand(std::time(0));
    std::cout << "CUDNN_VERSION:" << CUDNN_VERSION << std::endl;
    // 设定输入输出tensor的维度参数
    const int batch_size = 1;
    const int channel_in = 3;
    const int height_in = 768;
    const int width_in = 512;
    const int channel_out = 3;
    const int height_out = 768;
    const int width_out = 512;
    const int kernel_h = 3;
    const int kernel_w = 3;
    // 构造相关Tensor
    // input
    float *in_tensor = new float[batch_size*channel_in*height_in*width_in];
    // kernel input
    float *kernel_tensor = new float[channel_out*channel_in*kernel_h*kernel_w];
    // bias
    float *bias_tensor = new float[channel_out];
    float *z_tensor = new float[batch_size*channel_out*height_out*width_out];
    // output
    float *out_tensor = new float[batch_size*channel_out*height_out*width_out];
    for(int i = 0; i < batch_size*channel_in*height_in*width_in; ++i)
    {
        in_tensor[i] = float(rand() % 100000) / 100000;
    }
    for(int i = 0; i < channel_out; ++i)
    {
        bias_tensor[i] = float(rand() % 100000) / 100000;
    }
    for(int i = 0; i < channel_out*channel_in*kernel_h*kernel_w; ++i)
    {
        kernel_tensor[i] = float(rand() % 100000) / 10000;
    }
    for (int i = 0; i < batch_size*channel_out*height_out*width_out; ++i)
    {
        z_tensor[i] = 1.0f;
        out_tensor[i] = 1.0f;
    }
    float *qptr_gpu, *bias_gpu, *kernel_gpu, *outptr_gpu, *z_gpu;
    cudaMalloc((void**)&qptr_gpu, batch_size*channel_in*height_in*width_in*sizeof(float));
    cudaMalloc((void**)&bias_gpu, channel_out * sizeof(float));
    cudaMalloc((void**)&kernel_gpu, channel_out*channel_in*kernel_h*kernel_w*sizeof(float));
    cudaMalloc((void**)&outptr_gpu, batch_size*channel_out*height_out*width_out*sizeof(float));
    cudaMalloc((void**)&z_gpu, batch_size*channel_out*height_out*width_out*sizeof(float));
    cudaMemcpy(qptr_gpu, in_tensor, batch_size*channel_in*height_in*width_in*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_gpu, bias_tensor, channel_out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_gpu, kernel_tensor, channel_out*channel_in*kernel_h*kernel_w*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(z_gpu, z_tensor, batch_size*channel_out*height_out*width_out*sizeof(float), cudaMemcpyHostToDevice);
    // 创建cudnn句柄并设置
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudnnSetStream(cudnn, stream);
    // y = act ( alpha1 * conv(x) + alpha2 * z + bias )
    const float alpha1 = 1;
    const float alpha2 = 0;
    // 设置输入Tensor描述符
    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/batch_size,
                                          /*channels=*/channel_in,
                                          /*image_height=*/height_in,
                                          /*image_width=*/width_in);
    // 设置输出Tensor描述符
    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/channel_out,
                                      /*image_height=*/height_out,
                                      /*image_width=*/width_out);
    // 设置bias描述符
    cudnnTensorDescriptor_t bias_descriptor;
    cudnnCreateTensorDescriptor(&bias_descriptor);
    cudnnSetTensor4dDescriptor(bias_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/1,
                                      /*channels=*/channel_out,
                                      /*image_height=*/1,
                                      /*image_width=*/1);
    // 设置z描述符
    // // y = act ( alpha1 * conv(x) + alpha2 * z + bias ) 这里用不到
    cudnnTensorDescriptor_t z_descriptor;
    cudnnCreateTensorDescriptor(&z_descriptor);
    cudnnSetTensor4dDescriptor(z_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/channel_out,
                                      /*image_height=*/height_out,
                                      /*image_width=*/width_out);
    // 设置conv weight的描述
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnCreateFilterDescriptor(&kernel_descriptor);
    cudnnSetFilter4dDescriptor(kernel_descriptor,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*out_channels=*/channel_out,
                                          /*in_channels=*/channel_in,
                                          /*kernel_height=*/kernel_h,
                                          /*kernel_width=*/kernel_w);
    // 设置卷积相关参数
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnCreateConvolutionDescriptor(&convolution_descriptor);
    cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                              /*pad_height=*/1,
                                              /*pad_width=*/1,
                                              /*vertical_stride=*/1,
                                              /*horizontal_stride=*/1,
                                              /*dilation_height=*/1,
                                              /*dilation_width=*/1,
                                              /*mode=*/CUDNN_CROSS_CORRELATION,
                                              /*computeType=*/CUDNN_DATA_FLOAT);
    // 设置激活层相关参数
    cudnnActivationDescriptor_t activation_descriptor;
    cudnnCreateActivationDescriptor(&activation_descriptor);
    cudnnSetActivationDescriptor(activation_descriptor,
                                            /*mode=*/CUDNN_ACTIVATION_SIGMOID,
                                            /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
                                            /*relu_coef=*/0);
    // 获取卷积计算算法相关参数和workspace
    int cnt = 0;
    cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &cnt);
    //std::cout << "cnt: " << cnt << std::endl;
    cudnnConvolutionFwdAlgoPerf_t convolution_algorithm;
    int ret_cnt = 0;
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                            input_descriptor,
                                            kernel_descriptor,
                                            convolution_descriptor,
                                            output_descriptor,
                                            1,
                                            &ret_cnt,
                                            &convolution_algorithm);
    //std::cout << "ret_cnt: " << ret_cnt << "  " << convolution_algorithm.algo << std::endl;
    size_t workspace_bytes = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                      input_descriptor,
                                                      kernel_descriptor,
                                                      convolution_descriptor,
                                                      output_descriptor,
                                                      convolution_algorithm.algo,
                                                      &workspace_bytes);
    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);
    // 执行卷积运算
    cudnnConvolutionBiasActivationForward(
        cudnn, &alpha1, input_descriptor, qptr_gpu, kernel_descriptor, kernel_gpu,
        convolution_descriptor, convolution_algorithm.algo, d_workspace, workspace_bytes,
        &alpha2, z_descriptor, z_gpu,
        bias_descriptor, bias_gpu, activation_descriptor, output_descriptor, outptr_gpu);
    cudnnConvolutionBiasActivationForward(
        cudnn, &alpha1, input_descriptor, qptr_gpu, kernel_descriptor, kernel_gpu,
        convolution_descriptor, convolution_algorithm.algo, d_workspace, workspace_bytes,
        &alpha2, z_descriptor, z_gpu,
        bias_descriptor, bias_gpu, activation_descriptor, output_descriptor, outptr_gpu);
    auto start_time = std::chrono::high_resolution_clock::now();
    cudnnConvolutionForward(cudnn, &alpha1, input_descriptor, qptr_gpu, kernel_descriptor, kernel_gpu,
        convolution_descriptor, convolution_algorithm.algo, d_workspace, workspace_bytes, &alpha2,
        output_descriptor, outptr_gpu);
    cudaStreamSynchronize(stream);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << duration << std::endl;
    //for (int i = 0; i < 100; ++i)
    //{
        start_time = std::chrono::high_resolution_clock::now();
        cudnnConvolutionForward(cudnn, &alpha1, input_descriptor, qptr_gpu, kernel_descriptor, kernel_gpu,
            convolution_descriptor, convolution_algorithm.algo, d_workspace, workspace_bytes, &alpha2,
            output_descriptor, outptr_gpu);
        cudnnConvolutionForward(cudnn, &alpha1, input_descriptor, qptr_gpu, kernel_descriptor, kernel_gpu,
            convolution_descriptor, convolution_algorithm.algo, d_workspace, workspace_bytes, &alpha2,
            output_descriptor, outptr_gpu);
        cudnnConvolutionForward(cudnn, &alpha1, input_descriptor, qptr_gpu, kernel_descriptor, kernel_gpu,
            convolution_descriptor, convolution_algorithm.algo, d_workspace, workspace_bytes, &alpha2,
            output_descriptor, outptr_gpu);
        cudnnConvolutionForward(cudnn, &alpha1, input_descriptor, qptr_gpu, kernel_descriptor, kernel_gpu,
            convolution_descriptor, convolution_algorithm.algo, d_workspace, workspace_bytes, &alpha2,
            output_descriptor, outptr_gpu);
        cudnnConvolutionForward(cudnn, &alpha1, input_descriptor, qptr_gpu, kernel_descriptor, kernel_gpu,
            convolution_descriptor, convolution_algorithm.algo, d_workspace, workspace_bytes, &alpha2,
            output_descriptor, outptr_gpu);
        cudaStreamSynchronize(stream);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << duration/5.0 << std::endl;
    //}
    /*
    cudaMemcpy(out_tensor, outptr_gpu, batch_size*channel_out*height_out*width_out*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i)
    {
        std::cout << in_tensor[i] << "  " << out_tensor[i] << std::endl;
    }
    */
    // CPU Convolution
    /*
    int kw_low = kernel_w / 2;
    int kh_low = kernel_h / 2;
    int kw_high = width_in + kernel_w / 2;
    int kh_high = height_in + kernel_h / 2;

    int padded_iw = width_in + 2 * kw_low;
    int padded_ih = height_in + 2*kh_low;

    float *pad_temp = new float[channel_in * padded_ih * padded_iw];
    for (int i1 = 0; i1 < channel_in; ++i1)
    {
        for (int i2 = 0; i2 < padded_ih; ++i2)
        {
            for (int i3 = 0; i3 < padded_iw; ++i3)
            {
                pad_temp[(i1*padded_ih*padded_iw) + (i2*padded_iw) + i3] = 
                    (kh_low <= i2 && i2 < kh_high && kw_low <= i3 && i3 < kw_high)
                    ? in_tensor[(i1*height_in*width_in) + ((i2-kh_low)*width_in) + i3 - kw_low]
                    : 0.0f;
            }
        }
    }

    float *result = new float[batch_size*channel_out*height_out*width_out];
    for (int i11 = 0; i11 < channel_out; ++i11)
    {
        for (int i21 = 0; i21 < height_out; ++i21)
        {
            for (int i31 = 0; i31 < width_out; ++i31)
            {
                for (int i4 = 0; i4 < channel_in; ++i4)
                {
                    for (int i5 = 0; i5 < kernel_h; ++i5)
                    {
                        for (int i6 = 0; i6 < kernel_w; ++i6)
                        {
                            int cse_var_1 = i11 * height_out * width_out + i21 * width_out + i31;
                            if (i4 == 0 && i5 == 0 && i6 == 0)
                                result[cse_var_1] = 0.0f;
                            result[cse_var_1] = result[cse_var_1] + pad_temp[(i4*padded_ih*padded_iw) + (i21 + i5)*padded_iw + i31 + i6] * 
                                                                kernel_tensor[(i11*channel_in*kernel_h*kernel_w) + (i4 * kernel_h * kernel_w) + (i5 * kernel_w) + i6];
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < 10; ++i)
    {
        std::cout << result[i] << std::endl;
    }
    */

    // 销毁描述符和句柄
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(z_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyTensorDescriptor(bias_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);
    cudaFree(d_workspace);
    return 0;
}
