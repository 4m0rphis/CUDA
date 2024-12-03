CXX=g++
CXXFLAGS=-O3 -march=native -I/usr/include/opencv4/
LDLIBS=`pkg-config --libs opencv4`
NVFLAGS=-O3 -I/usr/include/opencv4/ -ccbin g++ -w


all: blurbox blurbox_omp blurbox-cu blurbox_sm


blurbox_sm: blurbox_sm.cu 
	nvcc $(NVFLAGS) -o $@ $< $(LDLIBS)

blurbox-cu: blurbox.cu
	nvcc $(NVFLAGS) -o $@ $< $(LDLIBS)

blurbox: blurbox.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

blurbox_omp: blurbox_omp.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm -f blurbox blurbox_omp blurbox-cu blurbox_sm out.jpg
