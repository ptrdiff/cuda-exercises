#include <iostream>


__global__ void add(int* a, int* b, int* c, int vector_size)
{
    for (int i = 0; i < vector_size; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_wthreads(int* a, int* b, int* c, int vector_size)
{
    for (int i = threadIdx.x; i < vector_size; i+=blockDim.x)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_wblocks(int* a, int* b, int* c, int vector_size)
{
    for (int i = blockIdx.x; i < vector_size; i+=gridDim.x)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_wtb(int* a, int* b, int* c, int vector_size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < vector_size; 
        i+=blockDim.x * gridDim.x)
    {
        c[i] = a[i] + b[i];
    }
}

void print_res(int* a, int* b, int* c, int vector_size, int blocks_count, int threads_count)
{
    std::cout << "Add with " << blocks_count << " blocks and " << threads_count << " threads:\n";
    for (int i = 0; i < vector_size; ++i)
    {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << '\n';
    }
}


int main(void)
{
    int vector_size = 512;
    int threads_count = 4;

    int *a, *b;
    int *res1, *res2, *res3, *res4;

    cudaMallocManaged(&a, vector_size * sizeof(int));
    cudaMallocManaged(&b, vector_size * sizeof(int));

    cudaMallocManaged(&res1, vector_size * sizeof(int));
    cudaMallocManaged(&res2, vector_size * sizeof(int));
    cudaMallocManaged(&res3, vector_size * sizeof(int));
    cudaMallocManaged(&res4, vector_size * sizeof(int));

    for (int i = 0; i < vector_size; ++i)
    {
        a[i] = 2 * i;
        b[i] = -i;
        res1[i] = 0;
        res2[i] = 0;
        res3[i] = 0;
        res4[i] = 0;
    }

    add<<<1, 1>>>(a, b, res1, vector_size);
    add_wthreads<<<1, vector_size>>>(a, b, res2, vector_size);
    add_wblocks<<<vector_size, 1>>>(a, b, res3, vector_size);
    add_wtb<<<(vector_size + (threads_count - 1)) / threads_count, threads_count>>>(a, b, res4, vector_size);
    cudaDeviceSynchronize();
    
    print_res(a, b, res1, 10, 1, 1);
    print_res(a, b, res2, 10, 1, vector_size);
    print_res(a, b, res3, 10, vector_size, 1);
    print_res(a, b, res4, 10, (vector_size + (threads_count - 1)) / threads_count, threads_count);


    cudaFree(a);
    cudaFree(b);
    cudaFree(res1);
    cudaFree(res2);
    cudaFree(res3);
    cudaFree(res4);

    return 0;
}
