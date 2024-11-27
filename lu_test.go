package sparsem

import (
	"math"
	"testing"
)

func TestLUDecomposition(t *testing.T) {
	tests := []struct {
		name    string
		matrix  [][]float64
		wantErr bool
	}{
		{
			name: "3x3 non-singular matrix",
			matrix: [][]float64{
				{4, -3, 0},
				{2, 1, 2},
				{0, -1, 1},
			},
			wantErr: false,
		},
		{
			name: "3x3 singular matrix",
			matrix: [][]float64{
				{1, 2, 3},
				{2, 4, 6},
				{3, 6, 9},
			},
			wantErr: true,
		},
		{
			name: "3x3 nearly singular matrix",
			matrix: [][]float64{
				{1e-11, 2, 3},
				{2, 4, 6},
				{3, 6, 9.00000001},
			},
			wantErr: true,
		},
		{
			name: "3x3 diagonally dominant matrix",
			matrix: [][]float64{
				{10, -1, 2},
				{1, 12, -2},
				{-1, 2, 15},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := NewCSRMatrix(tt.matrix)
			lu, err := m.LU()

			if tt.wantErr {
				if err == nil {
					t.Errorf("LU() expected error for singular matrix")
				}
				return
			}

			if err != nil {
				t.Errorf("LU() unexpected error = %v", err)
				return
			}

			// Verify L is lower triangular with ones on diagonal
			l := lu.L.ToDense()
			for i := 0; i < len(l); i++ {
				if math.Abs(l[i][i]-1.0) > 1e-10 {
					t.Errorf("L diagonal element at (%d,%d) = %f, want 1.0", i, i, l[i][i])
				}
				for j := i + 1; j < len(l[i]); j++ {
					if math.Abs(l[i][j]) > 1e-10 {
						t.Errorf("L upper triangle element at (%d,%d) = %f, want 0.0", i, j, l[i][j])
					}
				}
			}

			// Verify U is upper triangular
			u := lu.U.ToDense()
			for i := 0; i < len(u); i++ {
				for j := 0; j < i; j++ {
					if math.Abs(u[i][j]) > 1e-10 {
						t.Errorf("U lower triangle element at (%d,%d) = %f, want 0.0", i, j, u[i][j])
					}
				}
			}

			// Verify L*U = P*A
			n := len(tt.matrix)
			pa := make([][]float64, n)
			for i := range pa {
				pa[i] = make([]float64, n)
				for j := range pa[i] {
					pa[i][j] = tt.matrix[lu.P[i]][j]
				}
			}
			lu_prod := multiplyMatrices(l, u)

			for i := 0; i < len(pa); i++ {
				for j := 0; j < len(pa[i]); j++ {
					if math.Abs(pa[i][j]-lu_prod[i][j]) > 1e-10 {
						t.Errorf("LU product mismatch at (%d,%d): got %f, want %f",
							i, j, lu_prod[i][j], pa[i][j])
					}
				}
			}
		})
	}
}

func multiplyMatrices(a, b [][]float64) [][]float64 {
	n := len(a)
	m := len(b[0])
	result := make([][]float64, n)
	for i := range result {
		result[i] = make([]float64, m)
		for j := range result[i] {
			for k := 0; k < len(b); k++ {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result
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
