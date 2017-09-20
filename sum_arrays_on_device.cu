#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>

__global__ void
sum_arrays_on_device(float *c, float *a, float *b)
{
        uint32_t i = threadIdx.x + blockIdx.x*blockDim.x;
        c[i] = a[i] + b[i];
}

void initialize_data(float *host_data,
                     float *device_data,
                     const uint32_t num_floats)
{
        cudaError_t status;
        time_t t;
        uint32_t size_bytes = sizeof(float)*num_floats;

        srand((uint32_t)time(&t));

        for (uint32_t i = 0;
             i < num_floats;
             ++i) {
                host_data[i] = (float)(rand() & 0xFF)/10.0f;
        }

        status = cudaMemcpy(device_data,
                            host_data,
                            size_bytes,
                            cudaMemcpyHostToDevice);
        assert(status == cudaSuccess);
}

int main(void)
{
        cudaError_t status;
        float *a;
        float *b;
        float *c;
        constexpr uint32_t num_floats = 32;
        constexpr uint32_t size_bytes = sizeof(float)*num_floats;

        status = cudaMalloc(&a, size_bytes);
        assert(status == cudaSuccess);

        status = cudaMalloc(&b, size_bytes);
        assert(status == cudaSuccess);

        status = cudaMalloc(&c, size_bytes);
        assert(status == cudaSuccess);

        float a_host[num_floats];
        float b_host[num_floats];
        initialize_data(a_host, a, num_floats);
        initialize_data(b_host, b, num_floats);

        dim3 block = num_floats/4;
        dim3 grid = (num_floats + (block.x - 1))/block.x;
        sum_arrays_on_device<<<grid, block>>>(c, a, b);

        status = cudaDeviceSynchronize();
        assert(status == cudaSuccess);

        status = cudaFree(a);
        assert(status == cudaSuccess);

        status = cudaFree(b);
        assert(status == cudaSuccess);

        status = cudaFree(c);
        assert(status == cudaSuccess);

        status = cudaDeviceReset();
        assert(status == cudaSuccess);

        return EXIT_SUCCESS;
}
