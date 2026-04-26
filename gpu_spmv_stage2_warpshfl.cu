#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#define WARP_SIZE 32

// Error checking macro
#define CUDA_CHECK(call)                                                       \
  if ((call) != cudaSuccess) {                                                 \
    cudaError_t err = cudaGetLastError();                                      \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line "      \
              << __LINE__ << std::endl;                                        \
    exit(1);                                                                   \
  }

// STAGE 2 KERNEL: Warp-Level CSR with Shuffle
__global__ void spmv_warp_kernel(int num_rows, const int *row_ptr,
                                 const int *col_idx, const double *values,
                                 const double *x, double *y) {
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  if (warp_id < num_rows) {
    int row_start = row_ptr[warp_id];
    int row_end = row_ptr[warp_id + 1];
    double sum = 0.0;

    // COALESCED ACCESS: All threads in warp fetch adjacent column
    // indices/values
    for (int j = row_start + lane_id; j < row_end; j += WARP_SIZE) {
      sum += values[j] * x[col_idx[j]];
    }

    // Parallel reduction within the warp using shuffle
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Only thread 0 of the warp writes to the output
    if (lane_id == 0) {
      y[warp_id] = sum;
    }
  }
}

int main() {
  int N = 10000;
  double density = 0.01;

  // 1. Host Matrix Generation
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

  // 2. Device Allocation (The fix for your "undefined" error)
  int *d_row_ptr, *d_col_idx;
  double *d_values, *d_x, *d_y;

  CUDA_CHECK(cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_col_idx, total_nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_values, total_nnz * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(double)));

  // 3. Host to Device Copy
  CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (N + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx.data(), total_nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), total_nnz * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice));

  // 4. Kernel Launch Parameters
  int threadsPerBlock = 256;
  int warpsPerBlock = threadsPerBlock / WARP_SIZE;
  int gridSize = (N + warpsPerBlock - 1) / warpsPerBlock;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  spmv_warp_kernel<<<gridSize, threadsPerBlock>>>(N, d_row_ptr, d_col_idx,
                                                  d_values, d_x, d_y);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);

  std::cout << "Stage 2 (Warp-Level) Time: " << ms << " ms" << std::endl;

  // 5. Cleanup
  CUDA_CHECK(
      cudaMemcpy(h_y.data(), d_y, N * sizeof(double), cudaMemcpyDeviceToHost));
  cudaFree(d_row_ptr);
  cudaFree(d_col_idx);
  cudaFree(d_values);
  cudaFree(d_x);
  cudaFree(d_y);

  return 0;
}