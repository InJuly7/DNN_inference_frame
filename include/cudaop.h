#ifndef CUADOP_H
#define CUDAOP_H

#include <string>
#include <vector>
#include <map>
#include <memory>



namespace cuda
{
    class Node
    {
        public:
            std::string type;
            std::string name;
            std::vector<int> inputs_idx;
            std::vector<int> outputs_idx;

            Node(const std::string& Node_type, const std::string& Node_name);
            void PrintCudaNode();
            virtual void Execute() = 0;
            virtual int SetKernelPara() = 0;
    };

    class Conv : public Node
    {
        public:
            // cuda para
            std::vector<int> input_shape;
            std::vector<int> output_shape;
            std::vector<int> pad;
            std::vector<int> edag;
            std::vector<int> kshape;
            std::vector<float> weight;
            std::vector<float> bias;
            
            int weight_size = 0;
            int bias_size = 0;
            int pad_temp_size = 0;
            int kernelpara_size = 0;
            int para_index;

            Conv(std::string Node_type,std::string Node_name) : Node(Node_type, Node_name) {};
            void Execute() override;
            int SetKernelPara();
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
    };
};





#endif // CUDAOP_H

