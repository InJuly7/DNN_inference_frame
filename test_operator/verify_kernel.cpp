#include "verify_kernel.h"

void verify_AddKernel_1(float* h_A,float add_const,float* h_C,int n)
{
    for (int i = 0; i < n; ++i)
    {
        float result = h_A[i]+add_const;
        std::cout<<h_A[i]<<" "<<add_const<<" "<<result<<"--->"<<h_C[i]<<std::endl;
        if (fabs(result-h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
}

void verify_AddKernel_2(float* h_A,float* h_B,float* h_C,int n)
{
    for (int i = 0; i < n; ++i)
    {
        float result = h_A[i]+h_B[i];
        std::cout<<h_A[i]<<" "<<h_B[i]<<" "<<result<<"--->"<<h_C[i]<<std::endl;
        if (fabs(result-h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
}

void verify_LeakyReluKernel(float* h_A,float alpha,float* h_C,int n)
{
    for (int i = 0; i < n; ++i)
    {
        float val = h_A[i];
        float result = (val < 0.0f) ? alpha*val: val;
        std::cout<<h_A[i]<<" "<<alpha<<" "<<result<<"--->"<<h_C[i]<<std::endl;
        if (fabs(result-h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
}

void verify_AbsKernel(float* h_A,float* h_C,int n)
{
    for (int i = 0; i < n; ++i)
    {
        float result = fabs(h_A[i]);
        std::cout<<h_A[i]<<" "<<result<<"--->"<<h_C[i]<<std::endl;
        if (fabs(result-h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
}

void verify_DivKernel(float* h_A,float div_const,float* h_C,int n)
{
    for (int i = 0; i < n; ++i)
    {
        float result = h_A[i] / div_const;
        std::cout<<h_A[i]<<" "<<div_const<<" "<<result<<"--->"<<h_C[i]<<std::endl;
        if (fabs(result-h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
}

void verify_TanhKernel(const float* A, float* C, int n)
{
    for (int i = 0; i < n; ++i)
    {   
        float val = A[i];
        float result = (expf(2.0f * val) - 1.0f) / (expf(2.0f * val) + 1.0f);
        std::cout<<A[i]<<" "<<result<<"--->"<<C[i]<<std::endl;
        if (fabs(result-C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
}

void verify_ConcatKernel(const float* A, const float* B, float* C)
{
    // for(int i = 0; i < H; ++i)
    // {   
    //     for(int j = 0; j < w; ++j)
    //     {
    //         int idx = i*H + W;
    //         float val = A[i];
    //         float result = (expf(2.0f * val) - 1.0f) / (expf(2.0f * val) + 1.0f);
    //         std::cout<<A[i]<<" "<<result<<"--->"<<C[i]<<std::endl;
    //         if (fabs(result-C[i]) > 1e-5)
    //         {
    //             fprintf(stderr, "Result verification failed at element %d!\n", i);
    //             exit(EXIT_FAILURE);
    //         }

    //     }
        
    // }
    std::cout<<"4*16*16 Concat 4*16*16 ---> 8*16*16"<<std::endl;
    for(int i = 0; i < 10; i++)
    {
        std::cout<<"A["<<i<<"] = "<<A[i]<<"--->"<<"C["<<i<<"] = "<<C[i]<<std::endl;
    }
    for(int i = 0; i < 10; i++)
    {
        std::cout<<"B["<<i<<"] =  "<<B[i]<<"--->"<<"C["<<i+1024<<"] = "<<C[i+1024]<<std::endl;
    }
    
}
