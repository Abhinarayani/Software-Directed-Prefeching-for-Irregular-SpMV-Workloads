#include <cmath>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <iomanip>
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

// --- KERNEL ---
template <int K_TILE>
__global__ void
spmv_hybrid_async_kernel(int num_rows, const int *row_ptr, const int *col_idx,
                         const double *values, const double *x, double *y) {

  __shared__ double shared_x[2][K_TILE];

  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  if (warp_id < num_rows) {
    int row_start = row_ptr[warp_id];
    int row_end = row_ptr[warp_id + 1];
    if (row_start == row_end)
      return;

    int first_col = col_idx[row_start];
    int last_col = col_idx[row_end - 1];
    int start_tile = (first_col / K_TILE) * K_TILE;

    double sum = 0.0;
    int buf_idx = 0;

    // Prefetch first tile
    for (int i = lane_id; i < K_TILE; i += WARP_SIZE) {
      int g_idx = start_tile + i;
      double val = (g_idx < num_rows) ? x[g_idx] : 0.0;
      __pipeline_memcpy_async(&shared_x[buf_idx][i], &val, sizeof(double));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    for (int tile_base = start_tile; tile_base <= last_col;
         tile_base += K_TILE) {
      int next_tile = tile_base + K_TILE;
      int next_buf = 1 - buf_idx;

      if (next_tile <= last_col) {
        for (int i = lane_id; i < K_TILE; i += WARP_SIZE) {
          int g_idx = next_tile + i;
          double val = (g_idx < num_rows) ? x[g_idx] : 0.0;
          __pipeline_memcpy_async(&shared_x[next_buf][i], &val, sizeof(double));
        }
        __pipeline_commit();
      }

      for (int j = row_start + lane_id; j < row_end; j += WARP_SIZE) {
        int col = col_idx[j];
        if (col >= tile_base && col < tile_base + K_TILE) {
          sum += values[j] * shared_x[buf_idx][col - tile_base];
        }
      }

      if (next_tile <= last_col) {
        __pipeline_wait_prior(0);
        __syncthreads();
        buf_idx = next_buf;
      }
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (lane_id == 0)
      y[warp_id] = sum;
  }
}

// --- HOST EXPERIMENT RUNNER ---
void run_experiment(int K, int N, int *d_rp, int *d_ci, double *d_v,
                    double *d_x, double *d_y,
                    const std::vector<double> &golden_y) {
  int threadsPerBlock = 256;
  int gridSize =
      (N + (threadsPerBlock / WARP_SIZE) - 1) / (threadsPerBlock / WARP_SIZE);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto launch = [&](int k_val) {
    if (k_val == 256)
      spmv_hybrid_async_kernel<256>
          <<<gridSize, threadsPerBlock>>>(N, d_rp, d_ci, d_v, d_x, d_y);
    else if (k_val == 512)
      spmv_hybrid_async_kernel<512>
          <<<gridSize, threadsPerBlock>>>(N, d_rp, d_ci, d_v, d_x, d_y);
    else if (k_val == 1024)
      spmv_hybrid_async_kernel<1024>
          <<<gridSize, threadsPerBlock>>>(N, d_rp, d_ci, d_v, d_x, d_y);
    else if (k_val == 2048)
      spmv_hybrid_async_kernel<2048>
          <<<gridSize, threadsPerBlock>>>(N, d_rp, d_ci, d_v, d_x, d_y);
  };

  // Warm-up
  launch(K);
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  launch(K);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);

  // Verification
  std::vector<double> h_y(N);
  CUDA_CHECK(
      cudaMemcpy(h_y.data(), d_y, N * sizeof(double), cudaMemcpyDeviceToHost));

  bool correct = true;
  if (std::abs(h_y[0] - golden_y[0]) > 1e-6)
    correct = false;

  std::cout << std::left << std::setw(10) << K << " | " << std::setw(12) << ms
            << " ms"
            << " | " << (correct ? "PASS" : "FAIL") << std::endl;
}

int main() {
  int N = 10000;
  double density = 0.01;

  // 1. Matrix Generation
  std::vector<int> h_row_ptr;
  std::vector<int> h_col_idx;
  std::vector<double> h_values;
  std::vector<double> h_x(N, 1.0);
  std::vector<double> golden_y(N, 0.0);

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

  // 2. Compute Golden Reference on CPU
  for (int i = 0; i < N; i++) {
    double sum = 0;
    for (int j = h_row_ptr[i]; j < h_row_ptr[i + 1]; j++) {
      sum += h_values[j] * h_x[h_col_idx[j]];
    }
    golden_y[i] = sum;
  }

  // 3. Device Allocation
  int *d_rp, *d_ci;
  double *d_v, *d_x, *d_y;
  CUDA_CHECK(cudaMalloc(&d_rp, (N + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_ci, h_values.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_v, h_values.size() * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_rp, h_row_ptr.data(), (N + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ci, h_col_idx.data(), h_values.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_values.data(), h_values.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice));

  // 4. Run Sweep
  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << std::left << std::setw(10) << "K (Tile)" << " | "
            << std::setw(12) << "Time" << " | " << "Status" << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;

  run_experiment(256, N, d_rp, d_ci, d_v, d_x, d_y, golden_y);
  run_experiment(512, N, d_rp, d_ci, d_v, d_x, d_y, golden_y);
  run_experiment(1024, N, d_rp, d_ci, d_v, d_x, d_y, golden_y);
  run_experiment(2048, N, d_rp, d_ci, d_v, d_x, d_y, golden_y);

  std::cout << "-----------------------------------------------" << std::endl;

  cudaFree(d_rp);
  cudaFree(d_ci);
  cudaFree(d_v);
  cudaFree(d_x);
  cudaFree(d_y);
  return 0;
}