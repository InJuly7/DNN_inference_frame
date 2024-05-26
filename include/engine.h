#ifndef ENGINE_H
#define ENGINE_H
#include <string>

void BuildCudaOperator();

void CreateCudaOperator(const std::string &operatorType, const std::string &operatorName);

void PrintParaOffsets();

#endif // ENGINE_H


