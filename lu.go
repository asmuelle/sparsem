package sparsem

import (
	"fmt"
	"math"
)

// LUDecomposition represents an LU decomposition of a matrix
type LUDecomposition struct {
	L *CSRMatrix // Lower triangular matrix
	U *CSRMatrix // Upper triangular matrix
	P []int      // Permutation vector for partial pivoting
}

// LU performs LU decomposition with partial pivoting
// PA = LU where P is a permutation matrix, L is lower triangular, and U is upper triangular
func (m *CSRMatrix) LU() (*LUDecomposition, error) {
	n := m.Rows
	if n != m.Cols {
		return nil, fmt.Errorf("matrix must be square for LU decomposition")
	}

	// Convert to dense format for simpler implementation
	a := m.ToDense()
	l := make([][]float64, n)
	u := make([][]float64, n)
	p := make([]int, n)

	// Initialize L, U, and P
	for i := 0; i < n; i++ {
		l[i] = make([]float64, n)
		u[i] = make([]float64, n)
		p[i] = i
		l[i][i] = 1 // Diagonal of L is 1
	}

	// Copy matrix A to U
	for i := 0; i < n; i++ {
		copy(u[i], a[i])
	}

	// Check for near-singularity
	for i := 0; i < n; i++ {
		if math.Abs(a[i][i]) < 1e-10 {
			return nil, fmt.Errorf("matrix is singular or nearly singular")
		}
		// Check row scaling
		rowSum := 0.0
		for j := 0; j < n; j++ {
			rowSum += math.Abs(a[i][j])
		}
		if rowSum < 1e-10 {
			return nil, fmt.Errorf("matrix is singular or nearly singular")
		}
	}

	// Check if matrix is singular before starting decomposition
	maxVal := 0.0
	minVal := math.MaxFloat64
	for i := 0; i < n; i++ {
		rowSum := 0.0
		colSum := 0.0
		for j := 0; j < n; j++ {
			rowSum += math.Abs(u[i][j])
			colSum += math.Abs(u[j][i])
		}
		if rowSum > maxVal {
			maxVal = rowSum
		}
		if rowSum < minVal && rowSum > 0 {
			minVal = rowSum
		}
		if colSum > maxVal {
			maxVal = colSum
		}
		if colSum < minVal && colSum > 0 {
			minVal = colSum
		}
		// Check if row or column is all zeros
		if rowSum < 1e-10 || colSum < 1e-10 {
			return nil, fmt.Errorf("matrix is singular or nearly singular")
		}
	}

	// Check condition number estimate
	if minVal < 1e-10 || maxVal/minVal > 1e12 {
		return nil, fmt.Errorf("matrix is singular or nearly singular")
	}

	// Perform LU decomposition with partial pivoting
	for k := 0; k < n-1; k++ {
		// Find pivot
		pivot := k
		maxVal := math.Abs(u[k][k])
		for i := k + 1; i < n; i++ {
			if math.Abs(u[i][k]) > maxVal {
				maxVal = math.Abs(u[i][k])
				pivot = i
			}
		}

		if maxVal < 1e-10 {
			return nil, fmt.Errorf("matrix is singular or nearly singular")
		}

		// Swap rows if necessary
		if pivot != k {
			u[k], u[pivot] = u[pivot], u[k]
			for j := 0; j < k; j++ {
				l[k][j], l[pivot][j] = l[pivot][j], l[k][j]
			}
			p[k], p[pivot] = p[pivot], p[k]
		}

		// Compute multipliers and eliminate
		for i := k + 1; i < n; i++ {
			if math.Abs(u[k][k]) < 1e-10 {
				return nil, fmt.Errorf("matrix is singular or nearly singular")
			}
			l[i][k] = u[i][k] / u[k][k]
			for j := k; j < n; j++ {
				u[i][j] -= l[i][k] * u[k][j]
			}
			// Zero out the eliminated element for numerical stability
			u[i][k] = 0
		}
	}

	// Check if the last diagonal element is too small
	if math.Abs(u[n-1][n-1]) < 1e-10 {
		return nil, fmt.Errorf("matrix is singular or nearly singular")
	}

	// Convert L and U to CSR format
	L, err := NewCSRMatrix(l)
	if err != nil {
		return nil, fmt.Errorf("failed to create L matrix: %v", err)
	}
	U, err := NewCSRMatrix(u)
	if err != nil {
		return nil, fmt.Errorf("failed to create U matrix: %v", err)
	}

	return &LUDecomposition{
		L: L,
		U: U,
		P: p,
	}, nil
}

// Solve solves the system Ax = b using LU decomposition
// First solves Ly = Pb, then Ux = y
func (lu *LUDecomposition) Solve(b []float64) ([]float64, error) {
	n := lu.L.Rows
	if len(b) != n {
		return nil, fmt.Errorf("dimension mismatch: matrix is %dx%d but b has length %d",
			n, n, len(b))
	}

	// Apply permutation to b
	pb := make([]float64, n)
	for i := 0; i < n; i++ {
		pb[i] = b[lu.P[i]]
	}

	// Solve Ly = Pb using forward substitution
	y := make([]float64, n)
	lDense := lu.L.ToDense()
	for i := 0; i < n; i++ {
		sum := pb[i]
		for j := 0; j < i; j++ {
			sum -= lDense[i][j] * y[j]
		}
		y[i] = sum
	}

	// Solve Ux = y using back substitution
	x := make([]float64, n)
	uDense := lu.U.ToDense()
	for i := n - 1; i >= 0; i-- {
		sum := y[i]
		for j := i + 1; j < n; j++ {
			sum -= uDense[i][j] * x[j]
		}
		if math.Abs(uDense[i][i]) < 1e-10 {
			return nil, fmt.Errorf("matrix is singular")
		}
		x[i] = sum / uDense[i][i]
	}

	return x, nil
}

// Det computes the determinant of the matrix using LU decomposition
func (lu *LUDecomposition) Det() float64 {
	det := 1.0
	uDense := lu.U.ToDense()
	
	// Count number of row exchanges (swaps)
	swaps := 0
	visited := make([]bool, len(lu.P))
	for i := 0; i < len(lu.P); i++ {
		if !visited[i] {
			cycleLen := 0
			for j := i; !visited[j]; j = lu.P[j] {
				visited[j] = true
				cycleLen++
			}
			if cycleLen > 1 {
				swaps += cycleLen - 1
			}
		}
	}
	
	// Determinant is product of diagonal elements of U times (-1)^swaps
	for i := 0; i < len(uDense); i++ {
		det *= uDense[i][i]
	}
	if swaps%2 == 1 {
		det = -det
	}
	
	return det
}

// IsLowerTriangular checks if L is lower triangular with ones on diagonal
func (lu *LUDecomposition) IsLowerTriangular() bool {
	l := lu.L.ToDense()
	n := len(l)
	tolerance := 1e-10

	for i := 0; i < n; i++ {
		// Check diagonal is 1
		if math.Abs(l[i][i]-1) > tolerance {
			return false
		}
		// Check upper triangle is 0
		for j := i + 1; j < n; j++ {
			if math.Abs(l[i][j]) > tolerance {
				return false
			}
		}
	}
	return true
}

// IsUpperTriangular checks if U is upper triangular
func (lu *LUDecomposition) IsUpperTriangular() bool {
	u := lu.U.ToDense()
	n := len(u)
	tolerance := 1e-10

	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			if math.Abs(u[i][j]) > tolerance {
				return false
			}
		}
	}
	return true
}
