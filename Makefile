OBJDIR=objs
CXX=g++ -m64
CXXFLAGS= -Wall
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS= -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -lineinfo

# List of source files
BASE_SRC = cudaBenchMarking.cpp acceleration.cu
STREAM_SRC = stream.cu

# List of header files
HEADERS = acceleration.h

# List of executable names
EXE = base
EXE_CUDA_STREAM = stream
EXECUTABLE = base stream

# Targets
OBJS_BASE=$(OBJDIR)/cudaBenchMarking.o $(OBJDIR)/acceleration.o 
OBJS_STREAM = $(OBJDIR)/stream.o
all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
	rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXE): dirs $(OBJS_BASE)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS_BASE) $(LDFLAGS)

$(EXE_CUDA_STREAM): dirs $(OBJS_STREAM)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS_STREAM) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

