#ifndef ENGINE_H
#define ENGINE_H
#include <string>
#include <memory>


void BuildCudaOperator();
void CreateCudaOperator(const std::string &operatorType, const std::string &operatorName);
void PrintParaOffsets();
void engine();
void freeHostTensors();
void HostinitializeTensors();
void paraMemcpy(float* cudaTensor_ptr,float* cudaPara_ptr,float* cudaPadTemp_ptr);
void writeArrayToFile(float *d_output_1, int rows, int cols, const char *filename);

#endif // ENGINE_H


