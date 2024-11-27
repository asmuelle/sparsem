package sparsem

import (
	"fmt"
	"math"
)

// CholeskyDecomposition represents a Cholesky decomposition of a matrix
type CholeskyDecomposition struct {
	L *CSRMatrix // Lower triangular matrix
}

// Cholesky performs Cholesky decomposition on a symmetric positive definite matrix
func (m *CSRMatrix) Cholesky() (*CholeskyDecomposition, error) {
	if !m.IsSymmetric() {
		return nil, fmt.Errorf("matrix must be symmetric for Cholesky decomposition")
	}

	n := m.Rows
	dense := m.ToDense() // For simplicity, we'll work with dense format first
	l := make([][]float64, n)
	for i := range l {
		l[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sum := dense[i][j]
			for k := 0; k < j; k++ {
				sum -= l[i][k] * l[j][k]
			}
			if i == j {
				if sum <= 0 {
					return nil, fmt.Errorf("matrix is not positive definite")
				}
				l[i][j] = math.Sqrt(sum)
			} else {
				l[i][j] = sum / l[j][j]
			}
		}
	}

	return &CholeskyDecomposition{L: NewCSRMatrix(l)}, nil
}

// Solve solves the system Ax = b where A is the original matrix
func (c *CholeskyDecomposition) Solve(b []float64) ([]float64, error) {
	n := c.L.Rows
	if len(b) != n {
		return nil, fmt.Errorf("dimension mismatch: matrix is %dx%d but b has length %d",
			n, n, len(b))
	}

	// Forward substitution to solve Ly = b
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := b[i]
		for j := 0; j < i; j++ {
			sum -= c.L.ToDense()[i][j] * y[j]
		}
		y[i] = sum / c.L.ToDense()[i][i]
	}

	// Backward substitution to solve L^T x = y
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		sum := y[i]
		for j := i + 1; j < n; j++ {
			sum -= c.L.ToDense()[j][i] * x[j]
		}
		x[i] = sum / c.L.ToDense()[i][i]
	}

	return x, nil
}