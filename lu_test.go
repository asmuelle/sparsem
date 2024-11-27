package sparsem

import (
	"math"
	"testing"
)

func TestLUDecomposition(t *testing.T) {
	// Test matrix
	data := [][]float64{
		{2, -1, 0},
		{-1, 2, -1},
		{0, -1, 2},
	}
	matrix := NewCSRMatrix(data)

	lu, err := matrix.LU()
	if err != nil {
		t.Fatalf("LU decomposition failed: %v", err)
	}

	// Test if L is lower triangular with ones on diagonal
	if !lu.IsLowerTriangular() {
		t.Error("L matrix is not lower triangular with ones on diagonal")
	}

	// Test if U is upper triangular
	if !lu.IsUpperTriangular() {
		t.Error("U matrix is not upper triangular")
	}

	// Test if PA = LU
	l := lu.L.ToDense()
	u := lu.U.ToDense()
	n := len(data)

	// Create permuted A
	pa := make([][]float64, n)
	for i := range pa {
		pa[i] = make([]float64, n)
		copy(pa[i], data[lu.P[i]])
	}

	// Compute LU
	product := make([][]float64, n)
	for i := range product {
		product[i] = make([]float64, n)
		for j := range product[i] {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += l[i][k] * u[k][j]
			}
			if math.Abs(sum-pa[i][j]) > 1e-10 {
				t.Errorf("PA != LU at position (%d,%d): got %f, want %f",
					i, j, sum, pa[i][j])
			}
		}
	}
}

func TestLUSolve(t *testing.T) {
	// Test system Ax = b
	data := [][]float64{
		{4, 1},
		{1, 3},
	}
	matrix := NewCSRMatrix(data)
	b := []float64{1, 2}

	lu, err := matrix.LU()
	if err != nil {
		t.Fatalf("LU decomposition failed: %v", err)
	}

	x, err := lu.Solve(b)
	if err != nil {
		t.Fatalf("LU solve failed: %v", err)
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

func TestLUSingular(t *testing.T) {
	// Test singular matrix
	data := [][]float64{
		{1, 1},
		{1, 1},
	}
	matrix := NewCSRMatrix(data)

	_, err := matrix.LU()
	if err == nil {
		t.Error("Expected error for singular matrix, got nil")
	}
}

func TestLUDet(t *testing.T) {
	tests := []struct {
		matrix [][]float64
		det    float64
	}{
		{
			matrix: [][]float64{
				{2, 0},
				{0, 2},
			},
			det: 4,
		},
		{
			matrix: [][]float64{
				{1, 2},
				{3, 4},
			},
			det: -2,
		},
		{
			matrix: [][]float64{
				{1, 0, 0},
				{0, 2, 0},
				{0, 0, 3},
			},
			det: 6,
		},
	}

	for i, test := range tests {
		matrix := NewCSRMatrix(test.matrix)
		lu, err := matrix.LU()
		if err != nil {
			t.Errorf("Test %d: LU decomposition failed: %v", i, err)
			continue
		}

		det := lu.Det()
		if math.Abs(det-test.det) > 1e-10 {
			t.Errorf("Test %d: Determinant mismatch: got %f, want %f",
				i, det, test.det)
		}
	}
}

func TestLUNonSquare(t *testing.T) {
	// Test non-square matrix
	data := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}
	matrix := NewCSRMatrix(data)

	_, err := matrix.LU()
	if err == nil {
		t.Error("Expected error for non-square matrix, got nil")
	}
}
