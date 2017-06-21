g++ --std=c++11 -o run main.cpp -I/data01/home/xiangliang/3rd/include/eigen3 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64/ -lmkl_rt -O3 -march=native

g++ --std=c++11 -o openblas openblas.cpp -I/data01/home/xiangliang/3rd/include/ -O3 -lopenblas
