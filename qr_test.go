package sparsem

import (
	"math"
	"testing"
)

func TestQRDecomposition(t *testing.T) {
	tests := []struct {
		name    string
		matrix  [][]float64
		wantErr bool
	}{
		{
			name: "3x3 non-singular matrix",
			matrix: [][]float64{
				{12, -51, 4},
				{6, 167, -68},
				{-4, 24, -41},
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
			name: "3x2 rectangular matrix",
			matrix: [][]float64{
				{1, 2},
				{3, 4},
				{5, 6},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := NewCSRMatrix(tt.matrix)
			qr, err := m.QR()

			if tt.wantErr {
				if err == nil {
					t.Errorf("QR() expected error for singular matrix")
				}
				return
			}

			if err != nil {
				t.Errorf("QR() unexpected error = %v", err)
				return
			}

			// Verify Q is orthogonal
			if !qr.IsOrthogonal() {
				t.Error("Q matrix is not orthogonal")
			}

			// Verify R is upper triangular
			r := qr.R.ToDense()
			for i := 1; i < len(r); i++ {
				for j := 0; j < i && j < len(r[i]); j++ {
					if math.Abs(r[i][j]) > 1e-10 {
						t.Errorf("R lower triangle element at (%d,%d) = %f, want 0.0", i, j, r[i][j])
					}
				}
			}

			// Verify QR = A
			q := qr.Q.ToDense()
			qr_prod := make([][]float64, len(q))
			for i := range qr_prod {
				qr_prod[i] = make([]float64, len(r[0]))
				for j := range qr_prod[i] {
					for k := 0; k < len(r); k++ {
						qr_prod[i][j] += q[i][k] * r[k][j]
					}
				}
			}
			a := m.ToDense()

			for i := 0; i < len(a); i++ {
				for j := 0; j < len(a[i]); j++ {
					if math.Abs(qr_prod[i][j]-a[i][j]) > 1e-10 {
						t.Errorf("QR product mismatch at (%d,%d): got %f, want %f",
							i, j, qr_prod[i][j], a[i][j])
					}
				}
			}
		})
	}
}

func TestQRSolve(t *testing.T) {
	tests := []struct {
		name    string
		matrix  [][]float64
		b       []float64
		want    []float64
		wantErr bool
	}{
		{
			name: "3x3 system",
			matrix: [][]float64{
				{3, -1, 2},
				{-1, 4, -2},
				{2, -2, 5},
			},
			b:       []float64{4, -1, 1},
			want:    []float64{1, 1, 1},
			wantErr: false,
		},
		{
			name: "singular system",
			matrix: [][]float64{
				{1, 2, 3},
				{2, 4, 6},
				{3, 6, 9},
			},
			b:       []float64{1, 2, 3},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := NewCSRMatrix(tt.matrix)
			qr, err := m.QR()
			if err != nil {
				if !tt.wantErr {
					t.Errorf("QR() unexpected error = %v", err)
				}
				return
			}

			x, err := qr.Solve(tt.b)
			if tt.wantErr {
				if err == nil {
					t.Error("Solve() expected error for singular system")
				}
				return
			}

			if err != nil {
				t.Errorf("Solve() unexpected error = %v", err)
				return
			}

			// Verify solution
			for i := range tt.want {
				if math.Abs(x[i]-tt.want[i]) > 1e-10 {
					t.Errorf("Solution mismatch at %d: got %f, want %f",
						i, x[i], tt.want[i])
				}
			}
		})
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
