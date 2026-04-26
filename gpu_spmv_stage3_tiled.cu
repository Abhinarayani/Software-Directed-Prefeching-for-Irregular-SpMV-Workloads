#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#define WARP_SIZE 32
#define TILE_SIZE 1024 // Adjust this based on N and Shared Memory limits

// Error checking macro
#define CUDA_CHECK(call)                                                       \
  if ((call) != cudaSuccess) {                                                 \
    cudaError_t err = cudaGetLastError();                                      \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line "      \
              << __LINE__ << std::endl;                                        \
    exit(1);                                                                   \
  }

// STAGE 3 KERNEL: Shared Memory Tiled SpMV
__global__ void spmv_tiled_kernel(int num_rows, const int *row_ptr,
                                  const int *col_idx, const double *values,
                                  const double *x, double *y) {

  // Shared memory acts as a high-speed "scratchpad" for the vector x
  __shared__ double shared_x[TILE_SIZE];

  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  if (warp_id < num_rows) {
    int row_start = row_ptr[warp_id];
    int row_end = row_ptr[warp_id + 1];
    double sum = 0.0;

    // Iterate through the vector x in tiles
    for (int tile_base = 0; tile_base < num_rows; tile_base += TILE_SIZE) {

      // 1. Cooperative Load: The warp loads a chunk of x into Shared Memory
      for (int i = lane_id; i < TILE_SIZE; i += WARP_SIZE) {
        if (tile_base + i < num_rows) {
          shared_x[i] = x[tile_base + i];
        } else {
          shared_x[i] = 0.0;
        }
      }
      __syncthreads(); // Barrier: Ensure all threads finished loading the tile

      // 2. Compute: Only process elements that fall within the current tile
      // range
      for (int j = row_start + lane_id; j < row_end; j += WARP_SIZE) {
        int col = col_idx[j];
        if (col >= tile_base && col < tile_base + TILE_SIZE) {
          sum += values[j] * shared_x[col - tile_base];
        }
      }
      __syncthreads(); // Barrier: Wait before loading the next tile of x
    }

    // 3. Reduction: Sum partial results within the warp
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane_id == 0) {
      y[warp_id] = sum;
    }
  }
}

int main() {
  int N = 10000;
  double density = 0.01;

  // 1. Host Setup
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

  // 3. Data Transfer
  CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (N + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx.data(), total_nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), total_nnz * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice));

  // 4. Launch
  int threadsPerBlock = 256;
  int warpsPerBlock = threadsPerBlock / WARP_SIZE;
  int gridSize = (N + warpsPerBlock - 1) / warpsPerBlock;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  spmv_tiled_kernel<<<gridSize, threadsPerBlock>>>(N, d_row_ptr, d_col_idx,
                                                   d_values, d_x, d_y);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);

  std::cout << "Stage 3 (Tiled Shared Memory) Time: " << ms << " ms"
            << std::endl;

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