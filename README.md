# Software-Directed-Prefeching-for-Irregular-SpMV-Workloads

# Core Concepts

## Sparse Matrix-Vector Multiplication (SpMV)
SpMV is the mathematical operation $y = A \times x$.
In high-performance computing, when a matrix $A$ is **sparse** (meaning most of its elements are zeros), we store and process only the non-zero values to significantly reduce memory usage and execution time.

## Compressed Sparse Row (CSR) Format
The CSR format is a storage method that "compresses" a sparse matrix into three distinct linear arrays. 

### Example Visualization
For a $3 \times 3$ sparse matrix:

$$
\begin{bmatrix}
5 & 0 & 0 \\
0 & 8 & 0 \\
3 & 0 & 6
\end{bmatrix}
$$

### The Three CSR Arrays
1. **`values`**: Stores only the non-zero numbers in order (row by row).
   - **Example:** `[5, 8, 3, 6]`
2. **`column_indices`**: Stores the column position (index) for each element in the `values` array.
   - **Example:** `[0, 1, 0, 2]` *(e.g., '5' is at column 0, '8' is at column 1)*
3. **`row_pointers`**: Acts as a "map" indicating where each row begins within the `values` array. It always contains $NumOfRows + 1$ elements.
   - **Example:** `[0, 1, 2, 4]`
   - **Row 0** starts at index `0`.
   - **Row 1** starts at index `1`.
   - **Row 2** starts at index `2`.
   - The final value (`4`) represents the total number of non-zero elements.
