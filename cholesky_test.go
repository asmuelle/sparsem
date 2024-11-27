package sparsem

import (
	"math"
	"testing"
)

func TestCholesky(t *testing.T) {
	tests := []struct {
		name    string
		matrix  [][]float64
		wantErr bool
	}{
		{
			name: "valid symmetric positive definite matrix",
			matrix: [][]float64{
				{4, 2, 0},
				{2, 5, 2},
				{0, 2, 5},
			},
			wantErr: false,
		},
		{
			name: "non-symmetric matrix",
			matrix: [][]float64{
				{4, 2, 0},
				{2, 5, 2},
				{1, 2, 5},
			},
			wantErr: true,
		},
		{
			name: "non-positive definite matrix",
			matrix: [][]float64{
				{1, 2, 3},
				{2, 4, 6},
				{3, 6, 9},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewCSRMatrix(tt.matrix)
			if err != nil {
				t.Fatalf("Failed to create CSR matrix: %v", err)
			}

			chol, err := m.Cholesky()
			if (err != nil) != tt.wantErr {
				t.Errorf("Cholesky() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err == nil {
				// Verify L * L^T = original matrix
				dense := m.ToDense()
				L := chol.L.ToDense()
				n := len(dense)
				
				// Compute L * L^T
				result := make([][]float64, n)
				for i := range result {
					result[i] = make([]float64, n)
					for j := range result[i] {
						sum := 0.0
						for k := 0; k <= min(i, j); k++ {
							sum += L[i][k] * L[j][k]
						}
						result[i][j] = sum
					}
				}

				// Compare with original matrix
				tol := 1e-10
				for i := 0; i < n; i++ {
					for j := 0; j < n; j++ {
						if math.Abs(result[i][j]-dense[i][j]) > tol {
							t.Errorf("Reconstruction error at (%d,%d): got %v, want %v",
								i, j, result[i][j], dense[i][j])
						}
					}
				}
			}
		})
	}
}

func TestCholeskyDecomposition_Solve(t *testing.T) {
	// Create a simple positive definite matrix
	matrix := [][]float64{
		{4, 2, 0},
		{2, 5, 2},
		{0, 2, 5},
	}
	m, err := NewCSRMatrix(matrix)
	if err != nil {
		t.Fatalf("Failed to create CSR matrix: %v", err)
	}

	chol, err := m.Cholesky()
	if err != nil {
		t.Fatalf("Cholesky decomposition failed: %v", err)
	}

	tests := []struct {
		name    string
		b       []float64
		wantErr bool
	}{
		{
			name:    "valid b vector",
			b:       []float64{1, 2, 3},
			wantErr: false,
		},
		{
			name:    "wrong size b vector",
			b:       []float64{1, 2},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x, err := chol.Solve(tt.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("Solve() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err == nil {
				// Verify Ax = b
				dense := m.ToDense()
				n := len(dense)
				result := make([]float64, n)
				for i := 0; i < n; i++ {
					sum := 0.0
					for j := 0; j < n; j++ {
						sum += dense[i][j] * x[j]
					}
					result[i] = sum
				}

				// Compare with original b vector
				tol := 1e-10
				for i := 0; i < n; i++ {
					if math.Abs(result[i]-tt.b[i]) > tol {
						t.Errorf("Solution error at %d: got %v, want %v",
							i, result[i], tt.b[i])
					}
				}
			}
		})
	}
}
