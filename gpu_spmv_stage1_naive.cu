#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    }

// STAGE 1 KERNEL: Naive One-Thread-Per-Row
__global__ void spmv_naive_kernel(int num_rows, const int* row_ptr, const int* col_idx, 
                                  const double* values, const double* x, double* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        double sum = 0.0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        for (int j = row_start; j < row_end; j++) {
            // Irregular access: x[col_idx[j]]
            sum += values[j] * x[col_idx[j]];
        }
        y[row] = sum;
    }
}

int main() {
    int N = 10000;
    double density = 0.01;
    
    // 1. Host Matrix Generation (CSR)
    std::vector<int> h_row_ptr;
    std::vector<int> h_col_idx;
    std::vector<double> h_values;
    std::vector<double> h_x(N, 1.0);
    std::vector<double> h_y(N, 0.0);

    h_row_ptr.push_back(0);
    std::default_random_engine gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (dist(gen) < density) {
                h_values.push_back(dist(gen));
                h_col_idx.push_back(j);
            }
        }
        h_row_ptr.push_back(h_values.size());
    }
    int total_nnz = h_values.size();

    // 2. Device Allocation
    int *d_row_ptr, *d_col_idx;
    double *d_values, *d_x, *d_y;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, total_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, total_nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(double)));

    // 3. Host to Device Copy
    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx.data(), total_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), total_nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // 4. Kernel Launch & Timing
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    spmv_naive_kernel<<<gridSize, blockSize>>>(N, d_row_ptr, d_col_idx, d_values, d_x, d_y);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // 5. Cleanup
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, N * sizeof(double), cudaMemcpyDeviceToHost));
    std::cout << "Stage 1 (Naive) Time: " << ms << " ms" << std::endl;

    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_values); 
    cudaFree(d_x); cudaFree(d_y);
    return 0;
}