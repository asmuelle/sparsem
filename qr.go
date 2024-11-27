package sparsem

import (
	"fmt"
	"math"
)

// QRDecomposition represents a QR decomposition of a matrix
type QRDecomposition struct {
	Q *CSRMatrix // Orthogonal matrix Q
	R *CSRMatrix // Upper triangular matrix R
}

// QR performs QR decomposition using the Modified Gram-Schmidt method
// A = QR where Q is orthogonal and R is upper triangular
func (m *CSRMatrix) QR() (*QRDecomposition, error) {
	if m.Rows < m.Cols {
		return nil, fmt.Errorf("matrix must have at least as many rows as columns")
	}

	n := m.Rows
	p := m.Cols
	a := m.ToDense() // Work with a copy of the input matrix

	// Check for near-singularity using column norms and condition number estimate
	maxNorm := 0.0
	minNorm := math.MaxFloat64
	colNorms := make([]float64, p)
	
	for j := 0; j < p; j++ {
		var norm float64
		for i := 0; i < n; i++ {
			norm += a[i][j] * a[i][j]
		}
		norm = math.Sqrt(norm)
		colNorms[j] = norm
		
		if norm > maxNorm {
			maxNorm = norm
		}
		if norm < minNorm && norm > 0 {
			minNorm = norm
		}
	}

	// Check for linear dependence and condition number
	tolerance := 1e-10 * maxNorm
	if minNorm < tolerance {
		return nil, fmt.Errorf("matrix is singular or nearly singular")
	}

	// For nearly singular matrices, also check for linear dependence
	if maxNorm/minNorm > 1e6 {
		// Check if any column is a linear combination of previous columns
		for j := 1; j < p; j++ {
			for i := 0; i < j; i++ {
				dot := 0.0
				for k := 0; k < n; k++ {
					dot += a[k][i] * a[k][j]
				}
				// If columns are nearly parallel
				if math.Abs(dot/(colNorms[i]*colNorms[j])) > 0.99999 {
					return nil, fmt.Errorf("matrix is singular or nearly singular")
				}
			}
		}
	}

	// Initialize Q and R
	q := make([][]float64, n)
	r := make([][]float64, p)
	for i := range q {
		q[i] = make([]float64, p)
		if i < p {
			r[i] = make([]float64, p)
		}
	}

	// Copy A to Q for initial vectors
	for j := 0; j < p; j++ {
		for i := 0; i < n; i++ {
			q[i][j] = a[i][j]
		}
	}

	// Modified Gram-Schmidt process
	for j := 0; j < p; j++ {
		// Compute norm of current column
		var norm float64
		for i := 0; i < n; i++ {
			norm += q[i][j] * q[i][j]
		}
		norm = math.Sqrt(norm)

		// Check for near-singularity
		if norm < 1e-10 {
			return nil, fmt.Errorf("matrix is singular or nearly singular")
		}

		// Normalize current column
		r[j][j] = norm
		invNorm := 1.0 / norm
		for i := 0; i < n; i++ {
			q[i][j] *= invNorm
		}

		// Orthogonalize remaining columns
		for k := j + 1; k < p; k++ {
			// Compute dot product
			var dot float64
			for i := 0; i < n; i++ {
				dot += q[i][j] * q[i][k]
			}

			// Set R entry and subtract projection
			r[j][k] = dot
			for i := 0; i < n; i++ {
				q[i][k] -= dot * q[i][j]
			}

			// Check for near linear dependence
			var colNorm float64
			for i := 0; i < n; i++ {
				colNorm += q[i][k] * q[i][k]
			}
			if math.Sqrt(colNorm) < 1e-10 {
				return nil, fmt.Errorf("matrix is singular or nearly singular")
			}
		}
	}

	// Convert Q and R to CSR format
	qMatrix := NewCSRMatrix(q)
	rMatrix := NewCSRMatrix(r)

	return &QRDecomposition{
		Q: qMatrix,
		R: rMatrix,
	}, nil
}

// IsOrthogonal checks if Q is orthogonal by verifying Q^T Q ≈ I
func (qr *QRDecomposition) IsOrthogonal() bool {
	q := qr.Q.ToDense()
	n := len(q)
	p := len(q[0])
	tolerance := 1e-10

	for i := 0; i < p; i++ {
		for j := 0; j < p; j++ {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += q[k][i] * q[k][j]
			}
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if math.Abs(sum-expected) > tolerance {
				return false
			}
		}
	}
	return true
}

// Solve solves the system Ax = b using QR decomposition
// First solves Qy = b, then Rx = y
func (qr *QRDecomposition) Solve(b []float64) ([]float64, error) {
	n := qr.Q.Rows
	p := qr.R.Cols

	if len(b) != n {
		return nil, fmt.Errorf("dimension mismatch: matrix is %dx%d but b has length %d",
			n, p, len(b))
	}

	// First step: compute y = Q^T b
	y := make([]float64, p)
	qDense := qr.Q.ToDense()
	for i := 0; i < p; i++ {
		for j := 0; j < n; j++ {
			y[i] += qDense[j][i] * b[j]
		}
	}

	// Second step: solve Rx = y by back substitution
	x := make([]float64, p)
	rDense := qr.R.ToDense()

	// Check diagonal elements of R
	for i := 0; i < p; i++ {
		if math.Abs(rDense[i][i]) < 1e-10 {
			return nil, fmt.Errorf("matrix is singular or nearly singular")
		}
	}

	// Back substitution with scaling
	for i := p - 1; i >= 0; i-- {
		sum := y[i]
		for j := i + 1; j < p; j++ {
			sum -= rDense[i][j] * x[j]
		}
		
		// Scale to avoid overflow
		if math.Abs(rDense[i][i]) > 1e-10 {
			x[i] = sum / rDense[i][i]
		} else {
			return nil, fmt.Errorf("matrix is singular or nearly singular")
		}

		// Check for numerical instability
		if math.IsInf(x[i], 0) || math.IsNaN(x[i]) || math.Abs(x[i]) > 1e12 {
			return nil, fmt.Errorf("matrix is singular or nearly singular")
		}
	}

	return x, nil
}
