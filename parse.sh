#!/bin/bash
g++ main.cpp -c -o main.o
g++ ./src/parse.cpp -c -o parse.o
g++ main.o parse.o -o test
./test
echo "Compilation and linking completed successfully."
