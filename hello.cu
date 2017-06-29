#include <stdio.h>

__global__ void hello_from_gpu(void)
{
        printf("Hello world from GPU, thread %d!\n", threadIdx.x);
}

int main(void)
{
        printf("Hello world from CPU!\n");
        hello_from_gpu<<<1, 10>>>();

        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
                printf("Kernel launch failed with error %s\n",
                       cudaGetErrorString(cudaerr));

        cudaDeviceReset();

        return EXIT_SUCCESS;
}
