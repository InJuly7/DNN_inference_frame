# dlcc -x cuda  xx.cu/cpp --cuda-gpu-arch=dlgpuc64 -std=c++11 -o xxx
CXX = dlcc -x cuda
CPP = g++

CXX_FLAGS := --cuda-gpu-arch=dlgpuc64 -std=c++14 -fPIC -fpermissive -Wno-attributes
CPP_FLAGS := -std=c++11

INCLUDES := -I/dl/sdk/include -I../include -I/dl/sdk/include/dlnne

ifeq ($(debug),1)
    CXX_FLAGS += -DDEBUG -g
endif

CU_SRCS = ../src/OpKernel.cu
CPP_SRCS := \
	main.cpp \
    ../src/memorypool.cpp \
    ../src/operator.cpp \
    ../src/util.cpp \
	../src/cudaop.cpp \
    ../src/engine.cpp 
    

# 目标文件
CU_OBJS = $(notdir $(CU_SRCS:.cu=.o))
CPP_OBJS = $(notdir $(CPP_SRCS:.cpp=.o))

EXECUTABLE = test

all: $(EXECUTABLE)

$(EXECUTABLE): $(CU_OBJS) $(CPP_OBJS)
	dlcc $^ -o $@

%.o: ../src/%.cu
	$(CXX) $< $(CXX_FLAGS) $(INCLUDES) -c -o $@

%.o: %.cu
	$(CXX) $< $(CXX_FLAGS) $(INCLUDES) -c -o $@

%.o: ../src/%.cpp
	$(CXX) $< $(CXX_FLAGS) $(INCLUDES) -c -o $@

%.o: ../%.cpp
	$(CXX) $< $(CXX_FLAGS) $(INCLUDES) -c -o $@

clean:
	rm -f $(EXECUTABLE) $(CU_OBJS) $(CPP_OBJS)

.PHONY: all clean
