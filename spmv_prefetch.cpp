#include <omp.h>

#include <iostream>
#include <vector>

struct CSRMatrix {
  int num_rows;
  int num_cols;
  std::vector<double> values;
  std::vector<int> column_indices;
  std::vector<int> row_pointers;
};

void spmv_prefetch(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y, int k) {
#pragma omp parallel for
  for (int i = 0; i < A.num_rows; i++) {
    double sum = 0.0;
    int row_start = A.row_pointers[i];
    int row_end = A.row_pointers[i + 1];

    for (int j = row_start; j < row_end; j++) {
      // PREFETCH LOGIC: Look 'k' steps ahead in the column_indices array
      if (j + k < A.column_indices.size()) {
        int future_col = A.column_indices[j + k];
        __builtin_prefetch(&x[future_col], 0, 3);
      }

      int col = A.column_indices[j];
      sum += A.values[j] * x[col];
    }
    y[i] = sum;
  }
}

int main(int argc, char** argv) {
  // For now, use the same 3x3 test matrix as the baseline to ensure accuracy
  CSRMatrix A;
  A.num_rows = 3;
  A.num_cols = 3;
  A.values = {5.0, 8.0, 3.0, 6.0};
  A.column_indices = {0, 1, 0, 2};
  A.row_pointers = {0, 1, 2, 4};

  std::vector<double> x = {1.0, 1.0, 1.0};
  std::vector<double> y(3, 0.0);

  // Set prefetch distance k from command line or default to 1
  int k = (argc > 1) ? std::atoi(argv[1]) : 1;

  spmv_prefetch(A, x, y, k);

  std::cout << "Prefetch Distance (k): " << k << std::endl;
  std::cout << "Result y: [" << y[0] << ", " << y[1] << ", " << y[2] << "]" << std::endl;

  return 0;
}