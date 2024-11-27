package sparsem

import (
	"math"
	"testing"
)

func TestQRDecomposition(t *testing.T) {
	// Test matrix
	data := [][]float64{
		{12, -51, 4},
		{6, 167, -68},
		{-4, 24, -41},
	}
	matrix := NewCSRMatrix(data)

	qr, err := matrix.QR()
	if err != nil {
		t.Fatalf("QR decomposition failed: %v", err)
	}

	// Test if Q is orthogonal
	if !qr.IsOrthogonal() {
		t.Error("Q matrix is not orthogonal")
	}

	// Test if R is upper triangular
	r := qr.R.ToDense()
	for i := 1; i < len(r); i++ {
		for j := 0; j < i; j++ {
			if math.Abs(r[i][j]) > 1e-10 {
				t.Errorf("R is not upper triangular at position (%d,%d): %f", i, j, r[i][j])
			}
		}
	}

	// Test if QR = A
	q := qr.Q.ToDense()
	product := make([][]float64, len(data))
	for i := range product {
		product[i] = make([]float64, len(data[0]))
		for j := range product[i] {
			sum := 0.0
			for k := 0; k < len(q[0]); k++ {
				sum += q[i][k] * r[k][j]
			}
			if math.Abs(sum-data[i][j]) > 1e-10 {
				t.Errorf("QR != A at position (%d,%d): got %f, want %f",
					i, j, sum, data[i][j])
			}
		}
	}
}

func TestQRSolve(t *testing.T) {
	// Test system Ax = b
	data := [][]float64{
		{4, 1},
		{1, 3},
	}
	matrix := NewCSRMatrix(data)
	b := []float64{1, 2}

	qr, err := matrix.QR()
	if err != nil {
		t.Fatalf("QR decomposition failed: %v", err)
	}

	x, err := qr.Solve(b)
	if err != nil {
		t.Fatalf("QR solve failed: %v", err)
	}

	// Verify solution
	expected := []float64{0.0909090909090909, 0.6363636363636364}
	for i := range x {
		if math.Abs(x[i]-expected[i]) > 1e-10 {
			t.Errorf("Solution mismatch at position %d: got %f, want %f",
				i, x[i], expected[i])
		}
	}
}

func TestQRSingular(t *testing.T) {
	// Test singular matrix
	data := [][]float64{
		{1, 1},
		{1, 1},
	}
	matrix := NewCSRMatrix(data)
	b := []float64{1, 1}

	qr, err := matrix.QR()
	if err != nil {
		t.Fatalf("QR decomposition failed: %v", err)
	}

	_, err = qr.Solve(b)
	if err == nil {
		t.Error("Expected error for singular matrix, got nil")
	}
}

func TestQRDimensionMismatch(t *testing.T) {
	// Test matrix with more columns than rows
	data := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}
	matrix := NewCSRMatrix(data)

	_, err := matrix.QR()
	if err == nil {
		t.Error("Expected error for matrix with more columns than rows, got nil")
	}
}
