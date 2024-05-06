#ifndef OPERATORS_H
#define OPERATORS_H

#include <string>
#include <vector>
#include <map>

struct Parameter
{
    std::string name;
    std::vector<int> shape; 
    // 展平数值
    std::vector<float> values;
    void setValues(const std::string& line);
};
namespace op
{
    class Node
    {    
        public:
            std::string type;
            std::string name;
            std::vector<std::string> inputs;
            std::vector<std::string> outputs;

            void PrintInfo();
            virtual void PrintPara();
            virtual void StoreParameter(const std::string& line);
            virtual void Execute() = 0;
            virtual void PrintAttributes() = 0;
            virtual void SetAttributesFromFile(std::string line);
    };

    class Conv : public Node
    {
        public:
            std::vector<int> dilations;
            int group;
            std::vector<int> kernel_shape;
            std::vector<int> pads;
            std::vector<int> strides;
            std::map<std::string, Parameter> parameters;
            
            Conv(std::string Node_type,std::string Node_name);
            void StoreParameter(const std::string& line) override;
            void PrintPara();
            void SetAttributesFromFile(std::string line) override;
            void Execute() override;
            void PrintAttributes() override;
    };
    class LeakyRelu : public Node
    {
        public:
            float alpha;

            LeakyRelu(std::string Node_type,std::string Node_name);
            void SetAttributesFromFile(std::string line) override;
            void Execute() override;
            void PrintAttributes() override;
    };
    class Constant : public Node
    {
        public:
            float constant_value;
            
            Constant(std::string Node_type,std::string Node_name);
            void SetAttributesFromFile(std::string line) override;
            void Execute() override;
            void PrintAttributes() override;
    };
    class Slice : public Node
    {
        public:
            int start_index;
            int end_index;
            int axis;
            int steps;

            Slice(std::string Node_type,std::string Node_name);
            void SetAttributesFromFile();
            void Execute() override;
            void PrintAttributes() override;
    };
    class Concat : public Node
    {
        public:
            int axis;

            Concat(std::string Node_type,std::string Node_name);
            void SetAttributesFromFile(std::string line) override;
            void Execute() override;
            void PrintAttributes() override;
    };
    class Add : public Node
    {
        public:
            float add_value;

            Add(std::string Node_type,std::string Node_name);
            void SetAttributesFromFile(std::string line) override;
            void Execute() override;
            void PrintAttributes() override;
    };
    class Abs : public Node
    {
        public:
            Abs(std::string Node_type,std::string Node_name);
            void SetAttributesFromFile(std::string line) override;
            void Execute() override;
            void PrintAttributes() override;
    };


};

#endif // OPERATORS_H
