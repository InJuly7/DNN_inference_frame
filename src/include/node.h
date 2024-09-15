#ifndef NODE_H
#define NODE_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <variant>

namespace ONNX
{
    class Node
    {    
        public:
            std::string type;
            std::string name;
            std::string output;
            std::vector<std::string> inputs;
            

            // void PrintInfo();
            virtual void print_node() = 0;
            // virtual void StoreParameter(const std::string& line);
            // virtual void PrintAttributes() = 0;
            // virtual void SetAttributesFromFile(std::string line);
    };

    class InputNode : public Node
    {
        public:
            std::vector<int> shape;
            InputNode(std::string node_name, std::vector<int> node_shape);
            void print_node();
            
    };

    class OutputNode : public Node
    {
        public:
            std::vector<int> shape;
            OutputNode(std::string node_name, std::vector<int> node_shape);
            void print_node(); 
    };

    class Abs : public Node
    {
        public:
            Abs(std::string node_name, std::vector<std::string> node_inputs, std::string node_output);
            void print_node();
    };

    class Add : public Node
    {
        public:
            Add(std::string node_name, std::vector<std::string> node_inputs, std::string node_output);
            void print_node();       
    };
    
    class Concat : public Node
    {
        public:
            int axis;

            Concat(std::string node_name, int axis, std::vector<std::string> node_inputs, std::string node_output);
            void print_node(); 
    };

    template<typename T>
    class Constant : public Node
    {
        public:
            T value;
            Constant(std::string& node_name, const T& node_value, std::string& node_output)
            {
                this->name = node_name;
                this->type = "Constant";
                this->inputs = {};
                this->output = node_output;
                this->value = node_value;
            }
            void print_node()
            {
                std::cout << "Node Name: " << name << std::endl;
                std::cout << "Node Type: " << type << std::endl;
                std::cout << "Value: " << value << std::endl;
                std::cout << "Outputs: " << output << std::endl;
                
            }
    };

    class Conv : public Node
    {
        public:
            std::vector<int> dilations;
            int group;
            std::vector<int> kernel_shape;
            std::vector<int> pads;
            std::vector<int> strides;
            std::vector<float> bias;
            std::vector<float> weights;
            std::vector<int> bias_shape;
            std::vector<int> weight_shape;
    

            ONNX::Conv::Conv(std::string node_name, std::vector<int> dilations, int group, 
                    std::vector<int> kernel_shape, std::vector<int> pads, std::vector<int> strides, std::vector<std::string> node_inputs, 
                    std::string node_output, std::vector<int> weight_shape, std::vector<float> weights, std::vector<int> bias_shape, 
                    std::vector<float> bias);
            void print_node();
    };

    class Div : public Node
    {
        public:
            Div(std::string node_name, std::vector<std::string> node_inputs, std::string node_output);
            void print_node();
    };

    class LeakyRelu : public Node
    {
        public:
            float alpha;
            
            LeakyRelu(std::string node_name, float alpha, std::vector<std::string> node_inputs, std::string node_output);
            void print_node(); 
    };

    class Slice : public Node
    {
        public:
            Slice(std::string node_name, std::vector<std::string> node_inputs, std::string node_output);
            void print_node(); 
    };

    class Tanh : public Node
    {
        public:
            Tanh(std::string node_name, std::vector<std::string> node_inputs, std::string node_output);
            void print_node(); 
    };

    

    

};

#endif // NODE_H
