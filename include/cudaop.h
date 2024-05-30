#ifndef CUADOP_H
#define CUDAOP_H

#include <string>
#include <vector>
#include <map>
#include <memory>


#define PRINTKERNELPRARA 0
#define PRINTCUDAOP 1
#define PRINTPARAOFFSET 0

namespace cuda
{
    class Node
    {
        public:
            std::string type;
            std::string name;
            std::vector<int> inputs_idx;
            std::vector<int> outputs_idx;
            int para_index;
            
            Node(const std::string& Node_type, const std::string& Node_name);
            void PrintCudaNode();
            virtual void Execute() = 0;
            virtual int SetKernelPara() = 0;
            virtual void printArgInfo() = 0;
    };

    class Conv : public Node
    {
        public:
            // cuda para
            std::vector<int> input_shape;
            std::vector<int> output_shape;
            std::vector<int> strides;
            std::vector<int> pads;
            std::vector<int> edag;
            std::vector<int> kshape;
            std::vector<float> weight;
            std::vector<float> bias;
            
            int group;
            int weight_size = 0;
            int bias_size = 0;
            int pads_temp_size = 0;
            int kernelpara_size = 0;
            

            Conv(std::string Node_type,std::string Node_name) : Node(Node_type, Node_name) {};
            void Execute() override;
            int SetKernelPara();
            void printArgInfo();
    };
    
    class LeakyRelu : public Node
    {
        public:
            float alpha;

            // cuda para
            int kernelpara_size = 8;
            int numElements = 0;

            LeakyRelu(std::string Node_type,std::string Node_name) : Node(Node_type, Node_name) {};;
            void Execute() override;
            int SetKernelPara();
            void printArgInfo();
    };

    class Abs : public Node
    {
        public:

            // cuda para
            int numElements = 0;
            int kernelpara_size = 4;

            Abs(std::string Node_type,std::string Node_name) : Node(Node_type, Node_name) {};
            void Execute() override;
            int SetKernelPara();
            void printArgInfo();
    };
    
    class Tanh : public Node
    {
        public:

            // cuda para
            int numElements = 0;
            int kernelpara_size = 4;

            Tanh(std::string Node_type,std::string Node_name) : Node(Node_type, Node_name) {};
            void Execute() override;
            int SetKernelPara();
            void printArgInfo();
    };

    class Add : public Node
    {
        public:

            // cuda para
            float add_value = 0;
            int kernelpara_size = 4;
            int numElements = 0;

            Add(std::string Node_type,std::string Node_name) : Node(Node_type, Node_name) {};
            void Execute() override;
            int SetKernelPara();
            void printArgInfo();        
    };

    class Div : public Node
    {
        public:

            // cuda para
            float div_value = 1;
            int kernelpara_size = 4;
            int numElements = 0;

            Div(std::string Node_type,std::string Node_name) : Node(Node_type, Node_name) {};
            void Execute() override;
            int SetKernelPara();
            void printArgInfo();
    };
};





#endif // CUDAOP_H

