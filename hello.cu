#include <assert.h>
#include <stdio.h>

__global__ void hello_from_gpu(void)
{
        printf("Hello world from GPU, thread %d!\n", threadIdx.x);
}

int main(void)
{
        printf("Hello world from CPU!\n");
        hello_from_gpu<<<1, 10>>>();

        int32_t runtime_version;
        cudaError_t cudaerr = cudaRuntimeGetVersion(&runtime_version);
        assert(cudaerr == cudaSuccess);

        int32_t driver_version;
        cudaerr = cudaDriverGetVersion(&driver_version);
        assert(cudaerr == cudaSuccess);

        printf("Runtime: %d, Driver: %d\n", runtime_version, driver_version);

        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
                printf("Kernel launch failed with error %s\n",
                       cudaGetErrorString(cudaerr));

        cudaDeviceReset();

        return EXIT_SUCCESS;
}
