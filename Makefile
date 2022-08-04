all:gemm-test

#CC=gcc
CC=icc

ifeq ($(CC),gcc)
	CFLAGS= -fopenmp -mavx2 -O3 -g
	LDFLAGS=-lgomp
else
	CFLAGS= -qopenmp -O3 -xCORE-AVX512 -g
	LDFLAGS=-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm
endif

gemm-test: gemm-test.o
	$(CC) -o $@ $< $(LDFLAGS)

run:
	taskset -c 0-$$(($(OMP_NUM_THREADS)-1)) ./gemm-test 960 10



%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o gemm-test

