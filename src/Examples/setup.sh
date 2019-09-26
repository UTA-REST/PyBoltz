/usr/local/cuda-10.1/bin/nvcc -Xcompiler -fPIC -c -O2 -o MonteTGpu.o MonteTGpu.cu
/usr/local/cuda-10.1/bin/nvcc -Xcompiler -fPIC -shared -o MonteTGpu.so MonteTGpu.o
