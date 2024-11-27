package sparsem

import (
	"math"
	"math/rand"
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
		{
			name: "Ill-conditioned matrix",
			matrix: [][]float64{
				{1e15, 1},
				{1, 1e-15},
			},
			wantErr: false,
		},
		{
			name: "Matrix with wide range of values",
			matrix: [][]float64{
				{1e8, 1e-8, 1},
				{1e-8, 1, 1e8},
				{1, 1e8, 1e-8},
			},
			wantErr: false,
		},
	}

	tolerance := 1e-10
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, err := NewCSRMatrix(tt.matrix)
			if err != nil {
				t.Fatalf("Failed to create test matrix: %v", err)
			}

			qr, err := a.QR()

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
					if math.Abs(r[i][j]) > tolerance {
						t.Errorf("R lower triangle element at (%d,%d) = %f, want 0.0", i, j, r[i][j])
					}
				}
			}

			// Verify QR = A with appropriate scaling
			q := qr.Q.ToDense()
			qr_prod := make([][]float64, len(q))
			maxA := 0.0
			for i := range tt.matrix {
				for j := range tt.matrix[i] {
					if math.Abs(tt.matrix[i][j]) > maxA {
						maxA = math.Abs(tt.matrix[i][j])
					}
				}
			}
			if maxA == 0 {
				maxA = 1
			}

			for i := range qr_prod {
				qr_prod[i] = make([]float64, len(r[0]))
				for j := range qr_prod[i] {
					sum := 0.0
					c := 0.0 // Kahan summation
					for k := 0; k < len(r); k++ {
						y := q[i][k]*r[k][j] - c
						t := sum + y
						c = (t - sum) - y
						sum = t
					}
					qr_prod[i][j] = sum
				}
			}

			// Compare with original matrix using relative error
			for i := 0; i < len(tt.matrix); i++ {
				for j := 0; j < len(tt.matrix[i]); j++ {
					relErr := math.Abs(qr_prod[i][j]-tt.matrix[i][j]) / maxA
					if relErr > tolerance {
						t.Errorf("QR product mismatch at (%d,%d): got %g, want %g, relative error %g",
							i, j, qr_prod[i][j], tt.matrix[i][j], relErr)
					}
				}
			}
		})
	}
}

func TestQROrthogonality(t *testing.T) {
    // Create test matrix
    dense := [][]float64{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
    }

    // Convert to CSR format
    a, err := NewCSRMatrix(dense)
    if err != nil {
        t.Fatalf("Failed to create test matrix: %v", err)
    }

    // Compute QR decomposition
    qr, err := a.QR()
    if err != nil {
        t.Fatalf("Failed to compute QR decomposition: %v", err)
    }

    // Check if Q is orthogonal
    if !qr.IsOrthogonal() {
        t.Error("Q is not orthogonal")
    }

    // Create b vector
    bDense := make([][]float64, 3)
    for i := range bDense {
        bDense[i] = []float64{1}
    }
    b, err := NewCSRMatrix(bDense)
    if err != nil {
        t.Fatalf("Failed to create b vector: %v", err)
    }

    // Solve the system
    x, err := qr.Solve(b)
    if err != nil {
        t.Errorf("Failed to solve system: %v", err)
    }

    // Verify solution by computing Ax - b
    ax, err := a.Multiply(x)
    if err != nil {
        t.Errorf("Failed to multiply A*x: %v", err)
    }

    // Convert ax and b to dense for comparison
    axDense := ax.ToDense()
    bDense = b.ToDense()

    // Check residual
    for i := 0; i < a.Rows; i++ {
        if math.Abs(axDense[i][0]-bDense[i][0]) > 1e-10 {
            t.Errorf("Solution verification failed at row %d: got %v, want %v",
                i, axDense[i][0], bDense[i][0])
        }
    }
}

func TestQRSolve(t *testing.T) {
	tests := []struct {
		name    string
		A       *CSRMatrix
		b       *CSRMatrix
		want    []float64
		wantErr bool
	}{
		{
			name: "simple system",
			A: &CSRMatrix{
				Values:   []float64{1, 1, 1, 1},
				ColInd:   []int{0, 1, 0, 1},
				RowPtr:   []int{0, 2, 4},
				Rows:     2,
				Cols:     2,
			},
			b: &CSRMatrix{
				Values:   []float64{3, 1},
				ColInd:   []int{0, 0},
				RowPtr:   []int{0, 1, 2},
				Rows:     2,
				Cols:     1,
			},
			want:    []float64{2, 1},
			wantErr: false,
		},
		{
			name: "singular system",
			A: &CSRMatrix{
				Values:   []float64{1, 1, 1, 1},
				ColInd:   []int{0, 1, 0, 1},
				RowPtr:   []int{0, 2, 4},
				Rows:     2,
				Cols:     2,
			},
			b: &CSRMatrix{
				Values:   []float64{1, 2},
				ColInd:   []int{0, 0},
				RowPtr:   []int{0, 1, 2},
				Rows:     2,
				Cols:     1,
			},
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			qr, err := tt.A.QR()
			if err != nil {
				t.Fatalf("Failed to compute QR decomposition: %v", err)
			}

			x, err := qr.Solve(tt.b)
			if (err != nil) != tt.wantErr {
				t.Errorf("QRDecomposition.Solve() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				// Convert CSRMatrix solution to dense vector for comparison
				xDense := make([]float64, x.Rows)
				for i := 0; i < x.Rows; i++ {
					for j := x.RowPtr[i]; j < x.RowPtr[i+1]; j++ {
						if x.ColInd[j] == 0 {
							xDense[i] = x.Values[j]
						}
					}
				}

				// Compare with expected solution
				for i, want := range tt.want {
					if math.Abs(xDense[i]-want) > 1e-10 {
						t.Errorf("QRDecomposition.Solve() solution mismatch at index %d: got %v, want %v",
							i, xDense[i], want)
					}
				}
			}
		})
	}
}

func TestQRSingular(t *testing.T) {
    data := [][]float64{
        {1, 1},
        {1, 1},
    }

    a, err := NewCSRMatrix(data)
    if err != nil {
        t.Fatalf("Failed to create test matrix: %v", err)
    }

    // Create b vector as CSRMatrix
    bDense := [][]float64{
        {1},
        {1},
    }
    b, err := NewCSRMatrix(bDense)
    if err != nil {
        t.Fatalf("Failed to create b vector: %v", err)
    }

    qr, err := a.QR()
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
    a, err := NewCSRMatrix(data)
    if err != nil {
        t.Fatalf("Failed to create test matrix: %v", err)
    }

    _, err = a.QR()
    if err == nil {
        t.Error("Expected error for matrix with more columns than rows, got nil")
    }
}

func TestQRDecompositionLargeMatrix(t *testing.T) {
    // Create a large matrix
    n := 100
    dense := make([][]float64, n)
    b := make([][]float64, n)
    for i := range dense {
        dense[i] = make([]float64, n)
        b[i] = make([]float64, 1)
        for j := range dense[i] {
            dense[i][j] = rand.Float64()
        }
        b[i][0] = rand.Float64()
    }

    // Convert to CSR format
    a, err := NewCSRMatrix(dense)
    if err != nil {
        t.Fatalf("Failed to create test matrix: %v", err)
    }
    bCSR, err := NewCSRMatrix(b)
    if err != nil {
        t.Fatalf("Failed to create b vector: %v", err)
    }

    // Compute QR decomposition
    qr, err := a.QR()
    if err != nil {
        t.Fatalf("Failed to compute QR decomposition: %v", err)
    }

    // Solve the system
    x, err := qr.Solve(bCSR)
    if err != nil {
        t.Fatalf("Failed to solve system: %v", err)
    }

    // Verify solution by computing Ax - b
    ax, err := a.Multiply(x)
    if err != nil {
        t.Errorf("Failed to multiply A*x: %v", err)
    }

    // Convert to dense format for comparison
    axDense := ax.ToDense()
    bDense := bCSR.ToDense()

    // Check residual
    for i := 0; i < n; i++ {
        if math.Abs(axDense[i][0]-bDense[i][0]) > 1e-8 {
            t.Errorf("Large residual at row %d: |Ax-b| = %v",
                i, math.Abs(axDense[i][0]-bDense[i][0]))
        }
    }
}

func TestQRDecompositionRank(t *testing.T) {
	tests := []struct {
		name    string
		matrix  [][]float64
		b       []float64
		want    int
		wantErr bool
	}{
		{
			name: "full rank 3x3",
			matrix: [][]float64{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
			b:       []float64{1, 1, 1},
			want:    3,
			wantErr: false,
		},
		{
			name: "rank deficient 3x3",
			matrix: [][]float64{
				{1, 1, 1},
				{1, 1, 1},
				{1, 1, 1},
			},
			b:       []float64{1, 1, 1},
			want:    1,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Convert matrix to CSR format
			a, err := NewCSRMatrix(tt.matrix)
			if err != nil {
				t.Fatalf("Failed to create test matrix: %v", err)
			}

			// Convert b to CSR format
			bDense := make([][]float64, len(tt.b))
			for i := range bDense {
				bDense[i] = []float64{tt.b[i]}
			}
			b, err := NewCSRMatrix(bDense)
			if err != nil {
				t.Fatalf("Failed to create b vector: %v", err)
			}

			// Compute QR decomposition
			qr, err := a.QR()
			if err != nil {
				if !tt.wantErr {
					t.Errorf("QR() unexpected error = %v", err)
				}
				return
			}

			// Try to solve the system
			_, err = qr.Solve(b)
			if err != nil && !tt.wantErr {
				t.Errorf("Solve() unexpected error = %v", err)
			}

			// Check rank by counting non-zero diagonal elements in R
			rank := 0
			for i := 0; i < min(qr.R.Rows, qr.R.Cols); i++ {
				if math.Abs(qr.R.At(i, i)) > 1e-10 {
					rank++
				}
			}

			if rank != tt.want {
				t.Errorf("Wrong rank: got %d, want %d", rank, tt.want)
			}
		})
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func TestQRDecompositionTallMatrix(t *testing.T) {
    // Create a tall matrix (more rows than columns)
    dense := [][]float64{
        {1, 2},
        {3, 4},
        {5, 6},
    }
    b := make([][]float64, 3)
    for i := range b {
        b[i] = []float64{float64(i + 1)}
    }

    // Convert to CSR format
    a, err := NewCSRMatrix(dense)
    if err != nil {
        t.Fatalf("Failed to create test matrix: %v", err)
    }
    bCSR, err := NewCSRMatrix(b)
    if err != nil {
        t.Fatalf("Failed to create b vector: %v", err)
    }

    // Compute QR decomposition
    qr, err := a.QR()
    if err != nil {
        t.Fatalf("Failed to compute QR decomposition: %v", err)
    }

    // Solve the system
    x, err := qr.Solve(bCSR)
    if err != nil {
        t.Fatalf("Failed to solve system: %v", err)
    }

    // Verify solution by computing Ax - b
    ax, err := a.Multiply(x)
    if err != nil {
        t.Errorf("Failed to multiply A*x: %v", err)
    }

    // Convert to dense format for comparison
    axDense := ax.ToDense()
    bDense := bCSR.ToDense()

    // Check residual
    for i := 0; i < a.Rows; i++ {
        if math.Abs(axDense[i][0]-bDense[i][0]) > 1e-10 {
            t.Errorf("Large residual at row %d: |Ax-b| = %v",
                i, math.Abs(axDense[i][0]-bDense[i][0]))
        }
    }
}

func TestQRDecompositionWideMatrix(t *testing.T) {
    // Create a wide matrix (more columns than rows)
    dense := [][]float64{
        {1, 2, 3},
        {4, 5, 6},
    }

    // Convert to CSR format
    a, err := NewCSRMatrix(dense)
    if err != nil {
        t.Fatalf("Failed to create test matrix: %v", err)
    }

    // Create b vector
    bDense := [][]float64{
        {1},
        {2},
    }
    b, err := NewCSRMatrix(bDense)
    if err != nil {
        t.Fatalf("Failed to create b vector: %v", err)
    }

    // Compute QR decomposition
    qr, err := a.QR()
    if err != nil {
        t.Fatalf("Failed to compute QR decomposition: %v", err)
    }

    // Solve the system
    x, err := qr.Solve(b)
    if err != nil {
        t.Fatalf("Failed to solve system: %v", err)
    }

    // Verify solution by computing Ax - b
    ax, err := a.Multiply(x)
    if err != nil {
        t.Errorf("Failed to multiply A*x: %v", err)
    }

    // Convert to dense format for comparison
    axDense := ax.ToDense()
    bDense = b.ToDense()

    // Check residual
    for i := 0; i < a.Rows; i++ {
        if math.Abs(axDense[i][0]-bDense[i][0]) > 1e-10 {
            t.Errorf("Large residual at row %d: |Ax-b| = %v",
                i, math.Abs(axDense[i][0]-bDense[i][0]))
        }
    }
}
