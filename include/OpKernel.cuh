#include "cuda_runtime.h"  
#include "device_launch_parameters.h"

void LeakyReluKernel(const float *A, const float alpha, float *C, int numElements);
void TanhKernel(const float *A, float *C, int numElements);
void AbsKernel(const float *A, float *C, int numElements);
void DivKernel(const float *A, const float *B, const float *C, const float div_const, int numElements);
void AddKernel(const float *A, const float *B, const float *C, const float add_const, int numElements);
void ConvKernel(const float *A, float *C, float *weight, int *pads, int *edag, int *outshape, int *kshape, 
                                                        int *strides, float *pad_temp, int *input_shape, int group, const float *bias);
