#include <iostream>


__global__ void add(int* a, int* b, int* c, int rows, int columns)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += blockDim.x * gridDim.x)
    {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < columns; j += blockDim.y * gridDim.y)
        {
            c[i * columns + j] = a[i * columns + j] + b[i * columns + j];   
        }
    }
}

void print_matrix(int *matrix, int row, int columns)
{
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            std::cout << matrix[i * columns + j] << '\t';
        }
        std::cout << '\n';
    }
}


int main(void)
{
    int rows = 10;
    int columns = 10;
    int rows_threads = 8;
    int columns_threads = 8;
    int rows_blocks = rows / rows_threads;
    int columns_blocks = columns / columns_threads;

    dim3 blocks(rows_blocks, columns_blocks);
    dim3 threads(rows_threads, columns_threads);

    int *a, *b, *res;

    cudaMallocManaged(&a, rows * columns * sizeof(int));
    cudaMallocManaged(&b, rows * columns * sizeof(int));
    cudaMallocManaged(&res, rows * columns * sizeof(int));

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            a[i * columns + j] = i;
            b[i * columns + j] = j;
            res[i * columns + j] = 0;
        }
    }
    
    add<<<blocks, threads>>>(a, b, res, rows, columns);
    cudaDeviceSynchronize();
    
    std::cout << "First matrix:\n";
    print_matrix(a, rows, columns);

    std::cout << "Second matrix:\n";
    print_matrix(b, rows, columns);

    std::cout << "Result matrix:\n";
    print_matrix(res, rows, columns);

    cudaFree(a);
    cudaFree(b);
    cudaFree(res);

    return 0;
}
