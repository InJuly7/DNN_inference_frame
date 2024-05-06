#ifndef VERIFY_KERNEL_H
#define VERIFY_KERNEL_H

#include <iostream>
#include <cmath>


void verify_AddKernel_1(float* h_A,float add_const,float* h_C,int n);

void verify_AddKernel_2(float *h_A, float *h_B, float *h_C, int n);

void verify_LeakyReluKernel(float *h_A, float alpha, float *h_C, int n);

void verify_AbsKernel(float *h_A, float *h_C, int n);

void verify_DivKernel(float *h_A, float div_const, float *h_C, int n);

void verify_TanhKernel(const float *A, float *C, int n);

void verify_ConcatKernel(const float *A, const float *B, float *C);

#endif // VERIFY_KERNEL_H