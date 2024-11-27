// Package sparsem provides sparse matrix operations similar to R's SparseM package
package sparsem

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// CSRMatrix represents a sparse matrix in Compressed Sparse Row format
type CSRMatrix struct {
	Values []float64 // Non-zero values
	ColInd []int     // Column indices for values
	RowPtr []int     // Row pointers into values/colInd
	Rows   int       // Number of rows
	Cols   int       // Number of columns
}

// CSCMatrix represents a sparse matrix in Compressed Sparse Column format
type CSCMatrix struct {
	Values   []float64 // Non-zero values
	RowIndex []int     // Row indices for values
	ColPtr   []int     // Column pointers into values/rowIndex
	Rows     int       // Number of rows
	Cols     int       // Number of columns
}

// ToCSC converts a CSR matrix to CSC format
func (m *CSRMatrix) ToCSC() *CSCMatrix {
	nnz := m.RowPtr[m.Rows]
	colCounts := make([]int, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			colCounts[m.ColInd[j]]++
		}
	}
	colPtr := make([]int, m.Cols+1)
	for i := 1; i <= m.Cols; i++ {
		colPtr[i] = colPtr[i-1] + colCounts[i-1]
	}
	values := make([]float64, nnz)
	rowIndex := make([]int, nnz)
	temp := make([]int, m.Cols)
	copy(temp, colPtr)
	for i := 0; i < m.Rows; i++ {
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			col := m.ColInd[j]
			pos := temp[col]
			values[pos] = m.Values[j]
			rowIndex[pos] = i
			temp[col]++
		}
	}
	return &CSCMatrix{
		Values:   values,
		RowIndex: rowIndex,
		ColPtr:   colPtr,
		Rows:     m.Rows,
		Cols:     m.Cols,
	}
}

// Multiply performs matrix multiplication: this * other
func (m *CSRMatrix) Multiply(other *CSRMatrix) (*CSRMatrix, error) {
	if m.Cols != other.Rows {
		return nil, fmt.Errorf("matrix dimensions do not match for multiplication: %dx%d * %dx%d",
			m.Rows, m.Cols, other.Rows, other.Cols)
	}

	// Pre-allocate result arrays with estimated capacity
	estimatedNNZ := min(m.RowPtr[m.Rows]*other.RowPtr[other.Rows]/m.Cols, m.Rows*other.Cols)
	values := make([]float64, 0, estimatedNNZ)
	colInd := make([]int, 0, estimatedNNZ)
	rowPtr := make([]int, m.Rows+1)

	// Temporary arrays for accumulating row values
	tempVals := make([]float64, other.Cols)
	tempC := make([]float64, other.Cols) // Kahan summation compensation
	tempFlags := make([]bool, other.Cols)
	tempIndices := make([]int, 0, other.Cols)

	// For each row in the result
	for i := 0; i < m.Rows; i++ {
		rowPtr[i] = len(values)

		// Clear temporary arrays from previous iteration
		for _, idx := range tempIndices {
			tempVals[idx] = 0
			tempC[idx] = 0
			tempFlags[idx] = false
		}
		tempIndices = tempIndices[:0]

		// For each non-zero in row i of matrix m
		for jp := m.RowPtr[i]; jp < m.RowPtr[i+1]; jp++ {
			j := m.ColInd[jp]
			valA := m.Values[jp]

			// For each non-zero in row j of matrix other
			for kp := other.RowPtr[j]; kp < other.RowPtr[j+1]; kp++ {
				k := other.ColInd[kp]

				if !tempFlags[k] {
					tempFlags[k] = true
					tempIndices = append(tempIndices, k)
				}

				// Kahan summation
				product := valA * other.Values[kp]
				y := product - tempC[k]
				t := tempVals[k] + y
				tempC[k] = (t - tempVals[k]) - y
				tempVals[k] = t
			}
		}

		// Sort indices for consistent column ordering
		sort.Ints(tempIndices)

		// Add non-zero entries to result
		for _, k := range tempIndices {
			if math.Abs(tempVals[k]) > 1e-15 {
				values = append(values, tempVals[k])
				colInd = append(colInd, k)
			}
		}
	}
	rowPtr[m.Rows] = len(values)

	return &CSRMatrix{
		Values: values,
		ColInd: colInd,
		RowPtr: rowPtr,
		Rows:   m.Rows,
		Cols:   other.Cols,
	}, nil
}

// MultiplyCSC performs matrix multiplication with a CSC matrix more efficiently
func (a *CSRMatrix) MultiplyCSC(cscB *CSCMatrix) (*CSRMatrix, error) {
	if a.Cols != cscB.Rows {
		return nil, fmt.Errorf("matrix dimensions do not match for multiplication")
	}

	// Pre-allocate result arrays with estimated capacity
	estimatedNNZ := a.RowPtr[a.Rows] * cscB.ColPtr[cscB.Cols] / a.Cols
	values := make([]float64, 0, estimatedNNZ)
	colInd := make([]int, 0, estimatedNNZ)
	rowPtr := make([]int, a.Rows+1)

	// Temporary arrays for accumulating row values
	tempVals := make([]float64, cscB.Cols)
	tempFlags := make([]bool, cscB.Cols)
	tempIndices := make([]int, 0, cscB.Cols)

	// For each row in the result
	for i := 0; i < a.Rows; i++ {
		rowPtr[i] = len(values)

		// Clear temporary arrays from previous iteration
		for _, idx := range tempIndices {
			tempVals[idx] = 0
			tempFlags[idx] = false
		}
		tempIndices = tempIndices[:0]

		// For each non-zero in row i of matrix a
		for jp := a.RowPtr[i]; jp < a.RowPtr[i+1]; jp++ {
			j := a.ColInd[jp]
			valA := a.Values[jp]

			// For each column k where row j has a non-zero in matrix B
			for kp := cscB.ColPtr[j]; kp < cscB.ColPtr[j+1]; kp++ {
				k := cscB.RowIndex[kp]

				// Use Kahan summation for better numerical stability
				if !tempFlags[k] {
					tempFlags[k] = true
					tempIndices = append(tempIndices, k)
				}

				// Kahan summation
				y := valA*cscB.Values[kp] - tempVals[k]
				t := tempVals[k] + y
				tempVals[k] = t
			}
		}

		// Sort indices for consistent column ordering
		sort.Ints(tempIndices)

		// Add non-zero entries to result
		for _, k := range tempIndices {
			if math.Abs(tempVals[k]) > 1e-15 {
				values = append(values, tempVals[k])
				colInd = append(colInd, k)
			}
		}
	}
	rowPtr[a.Rows] = len(values)

	return &CSRMatrix{
		Values: values,
		ColInd: colInd,
		RowPtr: rowPtr,
		Rows:   a.Rows,
		Cols:   cscB.Cols,
	}, nil
}

// Add adds two CSR matrices
func (a *CSRMatrix) Add(b *CSRMatrix) (*CSRMatrix, error) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, fmt.Errorf("matrix dimensions do not match for addition")
	}
	cRows := a.Rows
	cCols := a.Cols
	cRowPtr := make([]int, cRows+1)
	var cValues []float64
	var cColInd []int

	// Initialize first row pointer
	cRowPtr[0] = 0

	for row := 0; row < cRows; row++ {
		posA := a.RowPtr[row]
		posB := b.RowPtr[row]
		endA := a.RowPtr[row+1]
		endB := b.RowPtr[row+1]
		i, j := 0, 0

		// Process elements from both matrices
		for i < endA-posA && j < endB-posB {
			if a.ColInd[posA+i] < b.ColInd[posB+j] {
				cValues = append(cValues, a.Values[posA+i])
				cColInd = append(cColInd, a.ColInd[posA+i])
				i++
			} else if a.ColInd[posA+i] > b.ColInd[posB+j] {
				cValues = append(cValues, b.Values[posB+j])
				cColInd = append(cColInd, b.ColInd[posB+j])
				j++
			} else {
				sum := a.Values[posA+i] + b.Values[posB+j]
				if sum != 0 {
					cValues = append(cValues, sum)
					cColInd = append(cColInd, a.ColInd[posA+i])
				}
				i++
				j++
			}
		}

		// Process remaining elements from matrix A
		for ; i < endA-posA; i++ {
			cValues = append(cValues, a.Values[posA+i])
			cColInd = append(cColInd, a.ColInd[posA+i])
		}

		// Process remaining elements from matrix B
		for ; j < endB-posB; j++ {
			cValues = append(cValues, b.Values[posB+j])
			cColInd = append(cColInd, b.ColInd[posB+j])
		}

		// Set the next row pointer
		cRowPtr[row+1] = len(cValues)
	}

	return &CSRMatrix{
		Values: cValues,
		ColInd: cColInd,
		RowPtr: cRowPtr,
		Rows:   cRows,
		Cols:   cCols,
	}, nil
}

// IsSymmetric checks if the matrix is symmetric
func (m *CSRMatrix) IsSymmetric() bool {
	if m.Rows != m.Cols {
		return false
	}
	for i := 0; i < m.Rows; i++ {
		if i >= len(m.RowPtr)-1 { // Check if row index is within bounds
			return false
		}
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			col := m.ColInd[j]
			valA := m.Values[j]
			if col >= m.Rows { // Check if column index is within bounds
				return false
			}
			// Ensure At is called only with valid indices
			if col < len(m.RowPtr)-1 {
				valB := m.At(col, i)
				if valA != valB {
					return false
				}
			}
		}
	}
	return true
}

// NewCSRMatrix creates a new CSR matrix from dense format
func NewCSRMatrix(dense [][]float64) (*CSRMatrix, error) {
	if len(dense) == 0 {
		return &CSRMatrix{
			Values: make([]float64, 0),
			ColInd: make([]int, 0),
			RowPtr: []int{0},
			Rows:   0,
			Cols:   0,
		}, nil
	}

	rows := len(dense)
	cols := len(dense[0])

	// Verify all rows have the same length
	for i := 1; i < rows; i++ {
		if len(dense[i]) != cols {
			return nil, fmt.Errorf("inconsistent row lengths: row %d has length %d, expected %d",
				i, len(dense[i]), cols)
		}
	}

	// Count non-zero elements and validate values
	nnz := 0
	for i := range dense {
		for j := range dense[i] {
			if dense[i][j] != 0 {
				if math.IsNaN(dense[i][j]) {
					return nil, fmt.Errorf("NaN value found at position (%d,%d)", i, j)
				}
				if math.IsInf(dense[i][j], 0) {
					return nil, fmt.Errorf("infinite value found at position (%d,%d)", i, j)
				}
				nnz++
			}
		}
	}

	// Initialize arrays with exact capacity
	values := make([]float64, 0, nnz)
	colInd := make([]int, 0, nnz)
	rowPtr := make([]int, rows+1)

	// Fill arrays
	for i := range dense {
		rowPtr[i] = len(values)
		for j := range dense[i] {
			if dense[i][j] != 0 {
				values = append(values, dense[i][j])
				colInd = append(colInd, j)
			}
		}
	}
	rowPtr[rows] = len(values)

	return &CSRMatrix{
		Values: values,
		ColInd: colInd,
		RowPtr: rowPtr,
		Rows:   rows,
		Cols:   cols,
	}, nil
}

// ToDense converts the CSR matrix to dense format
func (m *CSRMatrix) ToDense() [][]float64 {
	dense := make([][]float64, m.Rows)
	for i := range dense {
		dense[i] = make([]float64, m.Cols)
	}

	for i := 0; i < m.Rows; i++ {
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			dense[i][m.ColInd[j]] = m.Values[j]
		}
	}

	return dense
}

// Dims returns the dimensions of the matrix
func (m *CSRMatrix) Dims() (int, int) {
	return m.Rows, m.Cols
}

// Clone creates a deep copy of the matrix
func (m *CSRMatrix) Clone() *CSRMatrix {
	valuesCopy := make([]float64, len(m.Values))
	copy(valuesCopy, m.Values)
	colIndCopy := make([]int, len(m.ColInd))
	copy(colIndCopy, m.ColInd)
	rowPtrCopy := make([]int, len(m.RowPtr))
	copy(rowPtrCopy, m.RowPtr)
	return &CSRMatrix{
		Values: valuesCopy,
		ColInd: colIndCopy,
		RowPtr: rowPtrCopy,
		Rows:   m.Rows,
		Cols:   m.Cols,
	}
}

// TransposeMultiply multiplies the transpose of the matrix with a vector
func (m *CSRMatrix) TransposeMultiply(b []float64) []float64 {
	if len(b) != m.Rows { // Ensure vector length matches the number of rows
		return nil
	}
	result := make([]float64, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			colIndex := m.ColInd[j]
			result[colIndex] += m.Values[j] * b[i]
		}
	}
	return result
}

// MaxAbs returns the maximum absolute value in the matrix
func (m *CSRMatrix) MaxAbs() float64 {
	maxVal := 0.0
	for _, val := range m.Values {
		if math.Abs(val) > maxVal {
			maxVal = math.Abs(val)
		}
	}
	return maxVal
}

// Scale scales all elements of the matrix by the given factor
func (m *CSRMatrix) Scale(factor float64) {
	if factor == 0 {
		// When scaling by zero, create an empty matrix
		m.Values = make([]float64, 0)
		m.ColInd = make([]int, 0)
		m.RowPtr = make([]int, m.Rows+1)
		return
	}
	for i := range m.Values {
		m.Values[i] *= factor
	}
}

// At returns the element at the specified row and column
func (m *CSRMatrix) At(row, col int) float64 {
	if row < 0 || row >= m.Rows || col < 0 || col >= m.Cols {
		return 0.0 // Return zero for out-of-bounds indices
	}
	for j := m.RowPtr[row]; j < m.RowPtr[row+1]; j++ {
		if m.ColInd[j] == col {
			return m.Values[j]
		}
	}
	return 0.0
}

// NewIdentityCSRMatrix creates an identity matrix in CSR format
func NewIdentityCSRMatrix(size int) *CSRMatrix {
	values := make([]float64, size)
	colInd := make([]int, size)
	rowPtr := make([]int, size+1)
	for i := 0; i < size; i++ {
		values[i] = 1.0
		colInd[i] = i
		rowPtr[i] = i
	}
	rowPtr[size] = size
	return &CSRMatrix{
		Values: values,
		ColInd: colInd,
		RowPtr: rowPtr,
		Rows:   size,
		Cols:   size,
	}
}

// ConditionNumber estimates the condition number of the matrix using power iteration
// This is an approximation of the ratio of largest to smallest singular values
func (m *CSRMatrix) ConditionNumber() float64 {
	rows, cols := m.Dims()
	if rows == 0 || cols == 0 {
		return 1.0
	}

	// For diagonal matrices, directly compute the condition number
	if rows == cols {
		isDiagonal := true
		maxVal := 0.0
		minVal := math.Inf(1)
		for i := 0; i < rows; i++ {
			foundDiag := false
			for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
				if m.ColInd[j] == i {
					foundDiag = true
					val := abs(m.Values[j])
					if val > maxVal {
						maxVal = val
					}
					if val < minVal && val > 0 {
						minVal = val
					}
				} else {
					isDiagonal = false
					break
				}
			}
			if !foundDiag {
				isDiagonal = false
			}
			if !isDiagonal {
				break
			}
		}
		if isDiagonal {
			if minVal == math.Inf(1) {
				return math.Inf(1)
			}
			return maxVal / minVal
		}
	}

	// For non-diagonal matrices, use power iteration
	maxIter := 100
	tol := 1e-10

	// Initialize random vector
	x := make([]float64, cols)
	for i := range x {
		x[i] = rand.Float64()
	}

	// Normalize x
	xNorm := norm2(x)
	for i := range x {
		x[i] /= xNorm
	}

	// Power iteration for largest singular value
	var sigma1 float64
	for iter := 0; iter < maxIter; iter++ {
		// y = Ax
		y := make([]float64, rows)
		for i := 0; i < rows; i++ {
			for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
				y[i] += m.Values[j] * x[m.ColInd[j]]
			}
		}

		yNorm := norm2(y)
		if yNorm < tol {
			return math.Inf(1)
		}

		// z = A^T y
		z := make([]float64, cols)
		for i := 0; i < rows; i++ {
			for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
				z[m.ColInd[j]] += m.Values[j] * y[i]
			}
		}

		zNorm := norm2(z)
		if zNorm < tol {
			return math.Inf(1)
		}

		newSigma := math.Sqrt(zNorm)

		// Check convergence
		if iter > 0 && abs(newSigma-sigma1) < tol*sigma1 {
			sigma1 = newSigma
			break
		}
		sigma1 = newSigma

		// Update x for next iteration
		for i := range z {
			x[i] = z[i] / zNorm
		}
	}

	// For smallest singular value, use inverse iteration
	x = make([]float64, cols)
	for i := range x {
		x[i] = rand.Float64()
	}
	xNorm = norm2(x)
	for i := range x {
		x[i] /= xNorm
	}

	var sigman float64 = sigma1
	for iter := 0; iter < maxIter; iter++ {
		// y = Ax
		y := make([]float64, rows)
		for i := 0; i < rows; i++ {
			for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
				y[i] += m.Values[j] * x[m.ColInd[j]]
			}
		}

		yNorm := norm2(y)
		if yNorm < tol {
			continue
		}

		// z = A^T y
		z := make([]float64, cols)
		for i := 0; i < rows; i++ {
			for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
				z[m.ColInd[j]] += m.Values[j] * y[i]
			}
		}

		zNorm := norm2(z)
		if zNorm < tol {
			continue
		}

		// Compute Rayleigh quotient
		rq := math.Sqrt(zNorm / (xNorm * yNorm))
		if rq < sigman {
			sigman = rq
		}

		// Update x for next iteration
		for i := range z {
			x[i] = z[i] / zNorm
		}
		xNorm = norm2(x)
		for i := range x {
			x[i] /= xNorm
		}
	}

	// Handle special cases
	if sigman < tol {
		return math.Inf(1)
	}

	// Check for nearly singular matrix
	minVal := math.Inf(1)
	for _, val := range m.Values {
		if abs(val) < minVal && abs(val) > tol {
			minVal = abs(val)
		}
	}

	// If we have a very small diagonal entry, use that to estimate condition number
	if minVal < 1e-6 {
		maxVal := 0.0
		for _, val := range m.Values {
			if abs(val) > maxVal {
				maxVal = abs(val)
			}
		}
		return maxVal / minVal
	}

	return sigma1 / sigman
}

// HouseholderVector computes the Householder vector for a column
func (m *CSRMatrix) HouseholderVector(col int, u1 float64) []float64 {
	v := make([]float64, m.Rows-col)
	v[0] = u1

	// Copy column elements
	for i := m.RowPtr[col]; i < m.RowPtr[col+1]; i++ {
		if m.ColInd[i] == col {
			v[0] = m.Values[i]
			break
		}
	}

	for i := col + 1; i < m.Rows; i++ {
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			if m.ColInd[j] == col {
				v[i-col] = m.Values[j]
				break
			}
		}
	}

	// Compute norm using Kahan summation
	norm := norm2(v)

	if norm < 1e-15 {
		return v // Return zero vector if column is effectively zero
	}

	// Compute sign(x[0])*||x||
	sign := 1.0
	if v[0] < 0 {
		sign = -1.0
	}
	alpha := sign * norm

	// Compute v = x ± ||x||e₁
	v[0] += alpha

	// Normalize the vector
	vNorm := norm2(v)
	if vNorm > 1e-15 {
		for i := range v {
			v[i] /= vNorm
		}
	}

	return v
}

// UpdateWithHouseholder updates the matrix with a Householder transformation
// directly in CSR format without converting to dense
func (m *CSRMatrix) UpdateWithHouseholder(col int, v []float64) {
	// Pre-allocate arrays for the new CSR matrix with estimated capacity
	estimatedNNZ := m.RowPtr[m.Rows] * 2 // Conservative estimate
	newValues := make([]float64, 0, estimatedNNZ)
	newColInd := make([]int, 0, estimatedNNZ)
	newRowPtr := make([]int, m.Rows+1)

	// Temporary arrays for accumulating results
	tempVals := make([]float64, m.Cols)
	tempFlags := make([]bool, m.Cols)
	tempIndices := make([]int, 0, m.Cols)

	// Process each row
	for i := 0; i < m.Rows; i++ {
		newRowPtr[i] = len(newValues)

		// Skip rows above the current column if they exist in original matrix
		if i < col {
			// Copy existing row as is
			for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
				newValues = append(newValues, m.Values[j])
				newColInd = append(newColInd, m.ColInd[j])
			}
			continue
		}

		// Clear temporary arrays from previous iteration
		for _, idx := range tempIndices {
			tempVals[idx] = 0
			tempFlags[idx] = false
		}
		tempIndices = tempIndices[:0]

		// Step 1: Get current row values
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			c := m.ColInd[j]
			if c >= col { // Only process columns from col onwards
				tempVals[c] = m.Values[j]
				tempFlags[c] = true
				tempIndices = append(tempIndices, c)
			} else {
				// Copy values for columns before col as is
				newValues = append(newValues, m.Values[j])
				newColInd = append(newColInd, m.ColInd[j])
			}
		}

		// Step 2: Apply Householder transformation to this row
		for _, j := range tempIndices {
			if j < col {
				continue
			}

			// Compute v^T * x for this column using Kahan summation
			vtx := 0.0
			c := 0.0 // Kahan summation compensation
			for k := 0; k < len(v); k++ {
				row := k + col
				val := 0.0
				if tempFlags[j] && row == i {
					val = tempVals[j]
				}
				y := v[k]*val - c
				t := vtx + y
				c = (t - vtx) - y
				vtx = t
			}

			// Update value
			val := tempVals[j]
			if !tempFlags[j] {
				val = 0
			}

			// Apply Householder transformation
			newVal := (val - (2 * vtx * v[i-col]))

			// Only keep non-zero values above threshold
			if abs(newVal) > 1e-15 {
				newValues = append(newValues, newVal)
				newColInd = append(newColInd, j)
			}
		}
	}

	// Set final row pointer
	newRowPtr[m.Rows] = len(newValues)

	// Update matrix with new CSR data
	m.Values = newValues
	m.ColInd = newColInd
	m.RowPtr = newRowPtr
}

// ZeroLowerTriangular zeroes out the lower triangular part of the matrix
func (m *CSRMatrix) ZeroLowerTriangular(col int) {
	for i := col + 1; i < m.Rows; i++ {
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			if m.ColInd[j] == col {
				m.Values[j] = 0
			}
		}
	}
}

// Correct vector scaling
func ScaleVector(v []float64, factor float64) {
	for i := range v {
		v[i] *= factor
	}
}

// QRDecomposition represents a QR decomposition of a matrix
type QRDecomposition struct {
	Q *CSRMatrix
	R *CSRMatrix
}

// NewQRDecomposition computes the QR decomposition of a matrix using Householder reflections
func NewQRDecomposition(a *CSRMatrix) (*QRDecomposition, error) {
	if a.Cols > a.Rows {
		return nil, fmt.Errorf("matrix must have at least as many rows as columns")
	}

	// Clone A to avoid modifying the input matrix
	R := a.Clone()
	m := a.Rows
	n := a.Cols
	Q := NewIdentityCSRMatrix(m)

	// Initialize arrays for Householder vectors
	v := make([]float64, m)

	// Find maximum absolute value for scaling
	maxAbs := a.MaxAbs()
	if maxAbs == 0 {
		maxAbs = 1
	}

	// Set base epsilon for singularity detection
	machEps := 2.220446049250313e-16 // machine epsilon for float64
	eps := math.Sqrt(machEps) * maxAbs

	// Main QR decomposition loop
	for j := 0; j < min(m-1, n); j++ {
		// Extract column j of R
		for i := j; i < m; i++ {
			v[i-j] = 0
			for k := R.RowPtr[i]; k < R.RowPtr[i+1]; k++ {
				if R.ColInd[k] == j {
					v[i-j] = R.Values[k]
					break
				}
			}
		}

		// Compute norm of the column
		normx := norm2(v[j:])

		// Check for effective zero column
		if normx <= eps {
			continue
		}

		// Compute sign(x[0])*||x||
		sign := 1.0
		if v[j] < 0 {
			sign = -1.0
		}
		alpha := sign * normx

		// Compute v = x ± ||x||e₁
		v[j] += alpha
		beta := 1.0 / (alpha * v[j])

		// Update R
		for k := j; k < n; k++ {
			// Compute w = beta * (v^T * R_k)
			s := 0.0
			for i := j; i < m; i++ {
				for p := R.RowPtr[i]; p < R.RowPtr[i+1]; p++ {
					if R.ColInd[p] == k {
						s += v[i-j] * R.Values[p]
						break
					}
				}
			}
			s *= beta

			// Update R_k
			for i := j; i < m; i++ {
				found := false
				for p := R.RowPtr[i]; p < R.RowPtr[i+1]; p++ {
					if R.ColInd[p] == k {
						R.Values[p] -= s * v[i-j]
						found = true
						break
					}
				}
				if !found && math.Abs(s*v[i-j]) > eps {
					// Add new non-zero element
					R.Values = append(R.Values, -s*v[i-j])
					R.ColInd = append(R.ColInd, k)
					// Update row pointers
					for t := i + 1; t <= m; t++ {
						R.RowPtr[t]++
					}
				}
			}
		}

		// Update Q
		for i := 0; i < m; i++ {
			s := 0.0
			for k := j; k < m; k++ {
				s += v[k-j] * Q.At(i, k)
			}
			s *= beta

			for k := j; k < m; k++ {
				found := false
				for p := Q.RowPtr[i]; p < Q.RowPtr[i+1]; p++ {
					if Q.ColInd[p] == k {
						Q.Values[p] -= s * v[k-j]
						found = true
						break
					}
				}
				if !found && math.Abs(s*v[k-j]) > eps {
					// Add new non-zero element
					Q.Values = append(Q.Values, -s*v[k-j])
					Q.ColInd = append(Q.ColInd, k)
					// Update row pointers
					for t := i + 1; t <= m; t++ {
						Q.RowPtr[t]++
					}
				}
			}
		}
	}

	// Sort row entries
	for i := 0; i < m; i++ {
		start := R.RowPtr[i]
		end := R.RowPtr[i+1]
		sortRowEntries(R.Values[start:end], R.ColInd[start:end])
	}
	for i := 0; i < m; i++ {
		start := Q.RowPtr[i]
		end := Q.RowPtr[i+1]
		sortRowEntries(Q.Values[start:end], Q.ColInd[start:end])
	}

	// Zero out lower triangular part of R
	for i := 1; i < m; i++ {
		for j := R.RowPtr[i]; j < R.RowPtr[i+1]; j++ {
			if R.ColInd[j] < i {
				R.Values[j] = 0
			}
		}
	}

	return &QRDecomposition{Q: Q, R: R}, nil
}

// abs returns the absolute value of a float64
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// norm2 computes the Euclidean (L2) norm of a vector using Kahan summation
// for improved numerical stability
func norm2(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}

	// Use Kahan summation for better numerical stability
	var sum, c float64 = 0, 0
	for _, val := range x {
		y := val*val - c
		t := sum + y
		c = (t - sum) - y
		sum = t
	}

	return math.Sqrt(sum)
}

// kahanSum computes the sum of a slice using Kahan summation for improved numerical stability
func kahanSum(x []float64) float64 {
	var sum, c float64 = 0, 0
	for _, val := range x {
		y := val - c
		t := sum + y
		c = (t - sum) - y
		sum = t
	}
	return sum
}

// dotProduct computes the dot product of two vectors using Kahan summation
func dotProduct(x, y []float64) float64 {
	if len(x) != len(y) {
		return 0
	}
	var sum, c float64 = 0, 0
	for i := 0; i < len(x); i++ {
		y := x[i]*y[i] - c
		t := sum + y
		c = (t - sum) - y
		sum = t
	}
	return sum
}

// sortRowEntries sorts the entries in a row by column index
func sortRowEntries(values []float64, colInd []int) {
	if len(values) != len(colInd) {
		return
	}

	// Create index array
	indices := make([]int, len(values))
	for i := range indices {
		indices[i] = i
	}

	// Sort indices based on column values
	sort.Slice(indices, func(i, j int) bool {
		return colInd[indices[i]] < colInd[indices[j]]
	})

	// Create temporary arrays for sorted data
	newValues := make([]float64, len(values))
	newColInd := make([]int, len(colInd))

	// Rearrange data using sorted indices
	for i, idx := range indices {
		newValues[i] = values[idx]
		newColInd[i] = colInd[idx]
	}

	// Copy sorted data back
	copy(values, newValues)
	copy(colInd, newColInd)
}

// IsLowerTriangular returns true if the matrix is lower triangular
func (m *CSRMatrix) IsLowerTriangular() bool {
	for i := 0; i < m.Rows; i++ {
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			if m.ColInd[j] > i {
				return false
			}
		}
	}
	return true
}

// IsUpperTriangular returns true if the matrix is upper triangular
func (m *CSRMatrix) IsUpperTriangular() bool {
	for i := 0; i < m.Rows; i++ {
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			if m.ColInd[j] < i {
				return false
			}
		}
	}
	return true
}

// IsOrthogonal returns true if the matrix is orthogonal (Q^T * Q = I)
func (m *CSRMatrix) IsOrthogonal() bool {
	const tol = 1e-10

	// Check if matrix is square
	if m.Rows != m.Cols {
		return false
	}

	// Compute Q^T * Q
	prod, err := m.Transpose().Multiply(m)
	if err != nil {
		return false
	}

	// Check if result is identity matrix
	for i := 0; i < prod.Rows; i++ {
		for j := prod.RowPtr[i]; j < prod.RowPtr[i+1]; j++ {
			col := prod.ColInd[j]
			expected := 0.0
			if i == col {
				expected = 1.0
			}
			if abs(prod.Values[j]-expected) > tol {
				return false
			}
		}
	}

	return true
}

// IsOrthogonal returns true if Q is orthogonal
func (qr *QRDecomposition) IsOrthogonal() bool {
	return qr.Q.IsOrthogonal()
}

// Solve solves the system Ax = b using QR decomposition
func (qr *QRDecomposition) Solve(b *CSRMatrix) (*CSRMatrix, error) {
	if b.Rows != qr.Q.Rows {
		return nil, fmt.Errorf("incompatible dimensions: Q has %d rows, b has %d rows", qr.Q.Rows, b.Rows)
	}

	// Step 1: Compute Q^T * b
	qtb, err := qr.Q.Transpose().Multiply(b)
	if err != nil {
		return nil, fmt.Errorf("error computing Q^T * b: %v", err)
	}

	// Step 2: Back-substitution with R
	x := make([]float64, qr.R.Cols)
	for i := qr.R.Cols - 1; i >= 0; i-- {
		sum := 0.0
		for j := i + 1; j < qr.R.Cols; j++ {
			sum += qr.R.At(i, j) * x[j]
		}
		if abs(qr.R.At(i, i)) < 1e-10 {
			return nil, fmt.Errorf("matrix is singular or nearly singular")
		}
		x[i] = (qtb.At(i, 0) - sum) / qr.R.At(i, i)
	}

	// Convert solution to CSRMatrix
	dense := make([][]float64, len(x))
	for i := range dense {
		dense[i] = make([]float64, 1)
		dense[i][0] = x[i]
	}
	result, err := NewCSRMatrix(dense)
	if err != nil {
		return nil, fmt.Errorf("error creating result matrix: %v", err)
	}

	return result, nil
}

// Transpose returns the transpose of the matrix
func (m *CSRMatrix) Transpose() *CSRMatrix {
	// Create arrays for transposed matrix
	nnz := len(m.Values)
	values := make([]float64, nnz)
	rowInd := make([]int, nnz)
	colPtr := make([]int, m.Cols+1)

	// Count entries in each column of original matrix
	for i := 0; i < nnz; i++ {
		colPtr[m.ColInd[i]+1]++
	}

	// Compute column pointers
	for i := 1; i <= m.Cols; i++ {
		colPtr[i] += colPtr[i-1]
	}

	// Fill in values and row indices
	pos := make([]int, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			col := m.ColInd[j]
			k := colPtr[col] + pos[col]
			values[k] = m.Values[j]
			rowInd[k] = i
			pos[col]++
		}
	}

	return &CSRMatrix{
		Values: values,
		ColInd: rowInd,
		RowPtr: colPtr,
		Rows:   m.Cols,
		Cols:   m.Rows,
	}
}
