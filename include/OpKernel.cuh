#include "cuda_runtime.h"  
#include "device_launch_parameters.h"

void LeakyReluKernel(const float *A, const float alpha, float *C, int numElements);
void AbsKernel(const float *A, float *C, int numElements);
void AddKernel(const float *A, const float *B, const float *C, const float add_const, int numElements);
