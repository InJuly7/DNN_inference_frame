#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <cstdlib> // For system()

bool checkSignalFile(const std::string& filename)
{
    std::ifstream file(filename);
    return file.good();
}

int main()
{
    std::string signalFilename = "ready.txt";
    // 调用Python脚本
    std::cout << "Running Python script to parse ONNX model..." << std::endl;
    int result = system("python onnx_parse.py");
    if (result != 0)
    {
        std::cerr << "Failed to run Python script." << std::endl;
        return 1;
    }
    // 等待信号文件出现
    while (!checkSignalFile(signalFilename))
    {
        std::cout << "Waiting for signal..." << std::endl;
        // std::this_thread::sleep_for(std::chrono::seconds(1)); // Check every second
        for(int i = 0; i < 1000; i++)
        {
            
        }
    }
    std::cout << "Signal received. Proceeding with parsing model_parameters.txt..." << std::endl;
    
    return 0;
}