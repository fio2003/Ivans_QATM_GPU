#This should bild everything
CC = nvcc
#CFLAGS = -O3 -flto -Wall -fmessage-length=0  -std=gnu11
#LIBS = -lm

all: main lebedev 
	$(CC) $(CFLAGS) --cudart static --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_50,code=sm_50 project_6_v1.o sphere_lebedev_rule.o -link -o quatm_c_cuda.run
lebedev: sphere_lebedev_rule.cu
	$(CC) -G -g -O0 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_50,code=sm_50 -c sphere_lebedev_rule.cu
main: project_6_v1.cu
	$(CC) -G -g -O0 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_50,code=sm_50 -c project_6_v1.cu
clean:
	rm -f *.o *.run