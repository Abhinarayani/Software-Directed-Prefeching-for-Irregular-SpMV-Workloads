#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

#define WARP_SIZE 32

#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    }

// STAGE 2 KERNEL: Warp-Level CSR
__global__ void spmv_warp_kernel(int num_rows, const int* row_ptr, const int* col_idx, 
                                 const double* values, const double* x, double* y) {
    // Each warp handles one row
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; 

    if (warp_id < num_rows) {
        int row_start = row_ptr[warp_id];
        int row_end = row_ptr[warp_id + 1];
        double sum = 0.0;

        // All 32 threads iterate through the row elements with a stride of 32
        // This ensures COALESCED access to values and col_idx
        for (int j = row_start + lane_id; j < row_end; j += WARP_SIZE) {
            sum += values[j] * x[col_idx[j]];
        }

        // Parallel reduction within the warp using shuffle instructions
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Only the first thread in the warp writes the result
        if (lane_id == 0) {
            y[warp_id] = sum;
        }
    }
}

int main() {
    int N = 10000;
    double density = 0.01;
    
    // [Matrix generation logic is identical to Stage 1]
    // ... (Generate h_row_ptr, h_col_idx, h_values, h_x) ...
    // [Device Allocation and Copy logic is identical to Stage 1]
    // ... (cudaMalloc and cudaMemcpy to Device) ...

    // 4. Kernel Launch - Adjusted for Warps
    int threadsPerBlock = 256;
    int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    int gridSize = (N + warpsPerBlock - 1) / warpsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    spmv_warp_kernel<<<gridSize, threadsPerBlock>>>(N, d_row_ptr, d_col_idx, d_values, d_x, d_y);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Stage 2 (Warp-Level) Time: " << ms << " ms" << std::endl;

    // [Cleanup identical to Stage 1]
    return 0;
}