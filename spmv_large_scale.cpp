#include <omp.h>

#include <iostream>
#include <random>
#include <vector>

struct CSRMatrix {
  int num_rows;
  int num_cols;
  std::vector<double> values;
  std::vector<int> column_indices;
  std::vector<int> row_pointers;
};

void generate_random_csr(CSRMatrix& A, int n, double density) {
  A.num_rows = n;
  A.num_cols = n;
  A.row_pointers.push_back(0);
  std::default_random_engine gen(42);  // Fixed seed for reproducibility
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (dist(gen) < density) {
        A.values.push_back(dist(gen));
        A.column_indices.push_back(j);
      }
    }
    A.row_pointers.push_back(A.values.size());
  }
}

void spmv_prefetch(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y, int k) {
#pragma omp parallel for
  for (int i = 0; i < A.num_rows; i++) {
    double sum = 0.0;
    int row_start = A.row_pointers[i];
    int row_end = A.row_pointers[i + 1];

    for (int j = row_start; j < row_end; j++) {
      if (k > 0 && (j + k < A.column_indices.size())) {
        __builtin_prefetch(&x[A.column_indices[j + k]], 0, 3);
      }
      sum += A.values[j] * x[A.column_indices[j]];
    }
    y[i] = sum;
  }
}

int main(int argc, char** argv) {
  int N = 10000;
  double density = 0.01;
  int k = (argc > 1) ? std::atoi(argv[1]) : 0;

  CSRMatrix A;
  generate_random_csr(A, N, density);
  std::vector<double> x(N, 1.0);
  std::vector<double> y(N, 0.0);

  // Profile the actual computation
  spmv_prefetch(A, x, y, k);

  return 0;
}