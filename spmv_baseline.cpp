#include <omp.h>

#include <iostream>
#include <vector>

// 1. Define the CSR Structure
struct CSRMatrix {
  int num_rows;
  int num_cols;
  std::vector<double> values;       // Non-zero values
  std::vector<int> column_indices;  // Column index for each value
  std::vector<int> row_pointers;    // Where each row starts
};

// 2. The SpMV Function
void spmv_baseline(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y) {
// Parallelize the outer loop: each thread handles different rows [cite: 3, 25]
#pragma omp parallel for
  for (int i = 0; i < A.num_rows; i++) {
    double sum = 0.0;
    // row_pointers[i] to row_pointers[i+1] gives the range of non-zeros for this row [cite: 18, 21]
    for (int j = A.row_pointers[i]; j < A.row_pointers[i + 1]; j++) {
      int col = A.column_indices[j];
      sum += A.values[j] * x[col];
    }
    y[i] = sum;
  }
}

int main() {
  // 3. Create a simple 3x3 test matrix
  CSRMatrix A;
  A.num_rows = 3;
  A.num_cols = 3;
  // A.values = {5.0, 8.0, 3.0, 6.0};  // Non-zeros
  A.values = {4.0, 7.0, 2.0, 5.0};  // Non-zeros
  A.column_indices = {0, 1, 0, 2};  // Columns
  A.row_pointers = {0, 1, 2, 4};    // Row start indices

  std::vector<double> x = {1.0, 1.0, 1.0};  // Input vector
  std::vector<double> y(3, 0.0);            // Output vector

  spmv_baseline(A, x, y);

  // 4. Verify Accuracy
  std::cout << "Result y: [" << y[0] << ", " << y[1] << ", " << y[2] << "]" << std::endl;
  if (y[0] == 5.0 && y[1] == 8.0 && y[2] == 9.0) {
    std::cout << "Verification Successful!" << std::endl;
  } else {
    std::cout << "Verification Failed." << std::endl;
  }

  return 0;
}