# dlcc -x cuda  xx.cu/cpp --cuda-gpu-arch=dlgpuc64 -std=c++11 -o xxx
CXX = dlcc -x cuda
CPP = g++

CXX_FLAGS := --cuda-gpu-arch=dlgpuc64 -std=c++11
CPP_FLAGS := -std=c++11 

INCLUDES := -I/dl/sdk/include

ifeq ($(debug),1)
    CXX_FLAGS += -DDEBUG -g
endif

CU_SRCS = ../src/OpKernel.cu test_kernel.cu 
CPP_SRCS = verify_kernel.cpp

## 目标文件
CU_OBJS = $(notdir $(CU_SRCS:.cu=.o))
CPP_OBJS = $(notdir $(CPP_SRCS:.cpp=.o))

EXECUTABLE = test_kernel

all: $(EXECUTABLE)

$(EXECUTABLE): $(CU_OBJS) $(CPP_OBJS)
	dlcc $^ -o $@
	rm -f $(CU_OBJS) $(CPP_OBJS)

%.o: ../src/%.cu
	$(CXX) $< $(CXX_FLAGS) $(INCLUDES) -c -o $@

%.o: %.cu
	$(CXX) $< $(CXX_FLAGS) $(INCLUDES) -c -o $@

%.o: %.cpp
	$(CPP) $< $(CPP_FLAGS) $(INCLUDES) -c -o $@

clean:
	rm -f $(EXECUTABLE) $(CU_OBJS) $(CPP_OBJS)

.PHONY: all clean
