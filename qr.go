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

// QR performs QR decomposition using the Householder method
// A = QR where Q is orthogonal and R is upper triangular
func (m *CSRMatrix) QR() (*QRDecomposition, error) {
	if m.Rows < m.Cols {
		return nil, fmt.Errorf("matrix must have at least as many rows as columns")
	}

	n := m.Rows
	p := m.Cols
	dense := m.ToDense() // Start with dense format for simplicity

	// Initialize Q as identity matrix
	q := make([][]float64, n)
	for i := range q {
		q[i] = make([]float64, n)
		q[i][i] = 1
	}

	// Initialize R as a copy of the input matrix
	r := make([][]float64, n)
	for i := range r {
		r[i] = make([]float64, p)
		copy(r[i], dense[i])
	}

	// Perform Householder reflections
	for j := 0; j < p; j++ {
		// Compute Householder vector
		var norm float64
		for i := j; i < n; i++ {
			norm += r[i][j] * r[i][j]
		}
		norm = math.Sqrt(norm)

		if norm == 0 {
			continue
		}

		if r[j][j] > 0 {
			norm = -norm
		}

		// First component of v
		v0 := r[j][j] - norm
		beta := -v0 * norm

		if beta == 0 {
			continue
		}

		// Apply Householder reflection to R
		r[j][j] = norm
		for k := j + 1; k < p; k++ {
			sum := v0 * r[j][k]
			for i := j + 1; i < n; i++ {
				sum += r[i][j] * r[i][k]
			}
			sum /= beta

			r[j][k] += sum * v0
			for i := j + 1; i < n; i++ {
				r[i][k] += sum * r[i][j]
			}
		}

		// Apply Householder reflection to Q
		for k := 0; k < n; k++ {
			sum := v0 * q[k][j]
			for i := j + 1; i < n; i++ {
				sum += r[i][j] * q[k][i]
			}
			sum /= beta

			q[k][j] += sum * v0
			for i := j + 1; i < n; i++ {
				q[k][i] += sum * r[i][j]
			}
		}

		// Zero out the below-diagonal entries
		for i := j + 1; i < n; i++ {
			r[i][j] = 0
		}
	}

	// Transpose Q to get the final orthogonal matrix
	qTranspose := make([][]float64, n)
	for i := range qTranspose {
		qTranspose[i] = make([]float64, n)
		for j := range qTranspose[i] {
			qTranspose[i][j] = q[j][i]
		}
	}

	return &QRDecomposition{
		Q: NewCSRMatrix(qTranspose),
		R: NewCSRMatrix(r),
	}, nil
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

	// Solve Qy = b (multiply by Q^T)
	y := make([]float64, n)
	qDense := qr.Q.ToDense()
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < n; j++ {
			sum += qDense[j][i] * b[j] // Using Q transpose
		}
		y[i] = sum
	}

	// Solve Rx = y using back substitution
	x := make([]float64, p)
	rDense := qr.R.ToDense()
	for i := p - 1; i >= 0; i-- {
		sum := y[i]
		for j := i + 1; j < p; j++ {
			sum -= rDense[i][j] * x[j]
		}
		if math.Abs(rDense[i][i]) < 1e-10 {
			return nil, fmt.Errorf("matrix is singular or nearly singular")
		}
		x[i] = sum / rDense[i][i]
	}

	return x, nil
}

// IsOrthogonal checks if Q is orthogonal by verifying Q^T Q ≈ I
func (qr *QRDecomposition) IsOrthogonal() bool {
	q := qr.Q.ToDense()
	n := len(q)
	tolerance := 1e-10

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
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
