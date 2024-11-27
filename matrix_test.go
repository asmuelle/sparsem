package sparsem

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestNewCSRMatrixFromDense(t *testing.T) {
	dense := [][]float64{
		{1, 0, 2},
		{0, 3, 4},
	}
	matrix, err := NewCSRMatrix(dense)
	if err != nil {
		t.Fatalf("NewCSRMatrix returned error: %v", err)
	}

	// Check dimensions
	if matrix.Rows != 2 || matrix.Cols != 3 {
		t.Errorf("Wrong dimensions: got (%d, %d), want (2, 3)", matrix.Rows, matrix.Cols)
	}

	// Check non-zero values
	expectedValues := []float64{1, 2, 3, 4}
	if len(matrix.Values) != len(expectedValues) {
		t.Errorf("Wrong number of non-zero values: got %d, want %d", len(matrix.Values), len(expectedValues))
	}
	for i, v := range expectedValues {
		if i >= len(matrix.Values) || math.Abs(matrix.Values[i]-v) > 1e-10 {
			t.Errorf("Wrong value at index %d: got %f, want %f", i, matrix.Values[i], v)
		}
	}
}

func TestSortRowEntries(t *testing.T) {
	tests := []struct {
		name     string
		values   []float64
		colInd   []int
		want     struct {
			values   []float64
			colInd   []int
		}
	}{
		{
			name:     "Empty arrays",
			values:   []float64{},
			colInd:   []int{},
			want: struct {
				values   []float64
				colInd   []int
			}{
				values:   []float64{},
				colInd:   []int{},
			},
		},
		{
			name:     "Single element",
			values:   []float64{1.0},
			colInd:   []int{0},
			want: struct {
				values   []float64
				colInd   []int
			}{
				values:   []float64{1.0},
				colInd:   []int{0},
			},
		},
		{
			name:     "Small array (insertion sort)",
			values:   []float64{3.0, 1.0, 4.0, 2.0},
			colInd:   []int{3, 1, 4, 2},
			want: struct {
				values   []float64
				colInd   []int
			}{
				values:   []float64{1.0, 2.0, 3.0, 4.0},
				colInd:   []int{1, 2, 3, 4},
			},
		},
		{
			name:     "Large array (quicksort)",
			values:   []float64{5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0, 6.0, 0.0},
			colInd:   []int{5, 2, 8, 1, 9, 3, 7, 4, 6, 0},
			want: struct {
				values   []float64
				colInd   []int
			}{
				values:   []float64{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
				colInd:   []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
			},
		},
		{
			name:     "Duplicate indices",
			values:   []float64{1.0, 2.0, 3.0, 4.0},
			colInd:   []int{2, 1, 2, 1},
			want: struct {
				values   []float64
				colInd   []int
			}{
				values:   []float64{2.0, 4.0, 1.0, 3.0},
				colInd:   []int{1, 1, 2, 2},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make copies to avoid modifying test data
			values := make([]float64, len(tt.values))
			colInd := make([]int, len(tt.colInd))
			copy(values, tt.values)
			copy(colInd, tt.colInd)

			sortRowEntries(values, colInd)

			// Check lengths
			if len(values) != len(tt.want.values) || len(colInd) != len(tt.want.colInd) {
				t.Errorf("sortRowEntries() got lengths = (%d, %d), want (%d, %d)",
					len(values), len(colInd), len(tt.want.values), len(tt.want.colInd))
				return
			}

			// Check values
			for i := range values {
				if math.Abs(values[i]-tt.want.values[i]) > 1e-10 || colInd[i] != tt.want.colInd[i] {
					t.Errorf("sortRowEntries() got = (%v, %v), want (%v, %v)",
						values, colInd, tt.want.values, tt.want.colInd)
					return
				}
			}
		})
	}
}

func TestConditionNumber(t *testing.T) {
	tests := []struct {
		name      string
		matrix    [][]float64
		wantRange struct {
			min float64
			max float64
		}
	}{
		{
			name: "Identity matrix",
			matrix: [][]float64{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
			wantRange: struct {
				min float64
				max float64
			}{
				min: 0.9,
				max: 1.1,
			},
		},
		{
			name: "Diagonal matrix with varying entries",
			matrix: [][]float64{
				{1, 0, 0},
				{0, 2, 0},
				{0, 0, 4},
			},
			wantRange: struct {
				min float64
				max float64
			}{
				min: 3.5,
				max: 4.5,
			},
		},
		{
			name: "Nearly singular matrix",
			matrix: [][]float64{
				{1, 0, 0},
				{0, 1e-8, 0},
				{0, 0, 1},
			},
			wantRange: struct {
				min float64
				max float64
			}{
				min: 1e7,
				max: 1e9,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			matrix, err := NewCSRMatrix(tt.matrix)
			if err != nil {
				t.Fatalf("Failed to create test matrix: %v", err)
			}

			got := matrix.ConditionNumber()
			if got < tt.wantRange.min || got > tt.wantRange.max {
				t.Errorf("ConditionNumber() = %v, want range [%v, %v]",
					got, tt.wantRange.min, tt.wantRange.max)
			}
		})
	}
}

func TestNumericalHelpers(t *testing.T) {
	t.Run("kahanSum", func(t *testing.T) {
		// Test with numbers that would normally lose precision
		numbers := []float64{1.0, 1e-16, 1e-16, 1e-16}
		sum := kahanSum(numbers)
		expected := 1.0 + 3e-16
		if math.Abs(sum-expected) > 1e-16 {
			t.Errorf("kahanSum() = %v, want %v", sum, expected)
		}
	})

	t.Run("dotProduct", func(t *testing.T) {
		x := []float64{1, 2, 3}
		y := []float64{4, 5, 6}
		got := dotProduct(x, y)
		want := 32.0
		if math.Abs(got-want) > 1e-10 {
			t.Errorf("dotProduct() = %v, want %v", got, want)
		}
	})

	t.Run("norm2", func(t *testing.T) {
		x := []float64{3, 4}
		got := norm2(x)
		want := 5.0
		if math.Abs(got-want) > 1e-10 {
			t.Errorf("norm2() = %v, want %v", got, want)
		}
	})
}

func TestVectorOperations(t *testing.T) {
	t.Run("norm2", func(t *testing.T) {
		x := []float64{3, 4}
		got := norm2(x)
		want := 5.0
		if math.Abs(got-want) > 1e-10 {
			t.Errorf("norm2() = %v, want %v", got, want)
		}
	})
}

func TestMultiply(t *testing.T) {
	tests := []struct {
		name     string
		matrixA  [][]float64
		matrixB  [][]float64
		expected [][]float64
		wantErr  bool
	}{
		{
			name: "2x2 dense matrices",
			matrixA: [][]float64{
				{1, 2},
				{3, 4},
			},
			matrixB: [][]float64{
				{5, 6},
				{7, 8},
			},
			expected: [][]float64{
				{19, 22},
				{43, 50},
			},
			wantErr: false,
		},
		{
			name: "sparse matrices",
			matrixA: [][]float64{
				{1, 0, 2},
				{0, 3, 0},
				{4, 0, 5},
			},
			matrixB: [][]float64{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
			expected: [][]float64{
				{1, 0, 2},
				{0, 3, 0},
				{4, 0, 5},
			},
			wantErr: false,
		},
		{
			name: "incompatible dimensions",
			matrixA: [][]float64{
				{1, 2},
				{3, 4},
			},
			matrixB: [][]float64{
				{1, 2, 3},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, err := NewCSRMatrix(tt.matrixA)
			if err != nil {
				t.Fatalf("Failed to create matrix A: %v", err)
			}

			b, err := NewCSRMatrix(tt.matrixB)
			if err != nil {
				t.Fatalf("Failed to create matrix B: %v", err)
			}

			result, err := a.Multiply(b)
			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			// Convert result to dense for comparison
			resultDense := result.ToDense()
			for i := range tt.expected {
				for j := range tt.expected[i] {
					if math.Abs(resultDense[i][j]-tt.expected[i][j]) > 1e-10 {
						t.Errorf("Result mismatch at [%d][%d]: got %v, want %v",
							i, j, resultDense[i][j], tt.expected[i][j])
					}
				}
			}
		})
	}
}

func TestMultiplyCSC(t *testing.T) {
	tests := []struct {
		name    string
		csrA    *CSRMatrix
		cscB    *CSCMatrix
		want    *CSRMatrix
		wantErr bool
	}{
		{
			name: "compatible matrices",
			csrA: &CSRMatrix{
				Values:   []float64{1, 2, 3},
				ColInd:   []int{0, 1, 2},
				RowPtr:   []int{0, 1, 2, 3},
				Rows:     3,
				Cols:     3,
			},
			cscB: &CSCMatrix{
				Values:   []float64{4, 5, 6},
				RowIndex: []int{0, 1, 2},
				ColPtr:   []int{0, 1, 2, 3},
				Rows:     3,
				Cols:     3,
			},
			want: &CSRMatrix{
				Values:   []float64{4, 10, 18},
				ColInd:   []int{0, 1, 2},
				RowPtr:   []int{0, 1, 2, 3},
				Rows:     3,
				Cols:     3,
			},
			wantErr: false,
		},
		{
			name: "incompatible dimensions",
			csrA: &CSRMatrix{
				Values:   []float64{1, 2},
				ColInd:   []int{0, 1},
				RowPtr:   []int{0, 2},
				Rows:     2,
				Cols:     2,
			},
			cscB: &CSCMatrix{
				Values:   []float64{3, 4},
				RowIndex: []int{0, 1},
				ColPtr:   []int{0, 2},
				Rows:     3,
				Cols:     1,
			},
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.csrA.MultiplyCSC(tt.cscB)
			if (err != nil) != tt.wantErr {
				t.Errorf("MultiplyCSC() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && !compareCSR(got, tt.want) {
				t.Errorf("MultiplyCSC() = %v, want %v", got, tt.want)
			}
		})
	}
}

func compareCSR(a, b *CSRMatrix) bool {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return false
	}
	if len(a.Values) != len(b.Values) || len(a.ColInd) != len(b.ColInd) || len(a.RowPtr) != len(b.RowPtr) {
		return false
	}
	for i := range a.Values {
		if a.Values[i] != b.Values[i] {
			return false
		}
	}
	for i := range a.ColInd {
		if a.ColInd[i] != b.ColInd[i] {
			return false
		}
	}
	for i := range a.RowPtr {
		if a.RowPtr[i] != b.RowPtr[i] {
			return false
		}
	}
	return true
}

func TestToCSC(t *testing.T) {
	tests := []struct {
		name string
		csr  *CSRMatrix
		want *CSCMatrix
	}{
		{
			name: "simple CSR matrix",
			csr: &CSRMatrix{
				Values:   []float64{1, 2, 3, 4},
				ColInd:   []int{0, 2, 1, 3},
				RowPtr:   []int{0, 2, 4},
				Rows:     2,
				Cols:     4,
			},
			want: &CSCMatrix{
				Values:   []float64{1, 3, 2, 4},
				RowIndex: []int{0, 1, 0, 1},
				ColPtr:   []int{0, 1, 2, 3, 4},
				Rows:     2,
				Cols:     4,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.csr.ToCSC()
			if !compareCSC(got, tt.want) {
				t.Errorf("ToCSC() = %v, want %v", got, tt.want)
			}
		})
	}
}

func compareCSC(a, b *CSCMatrix) bool {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return false
	}
	if len(a.Values) != len(b.Values) || len(a.RowIndex) != len(b.RowIndex) || len(a.ColPtr) != len(b.ColPtr) {
		return false
	}
	for i := range a.Values {
		if a.Values[i] != b.Values[i] {
			return false
		}
	}
	for i := range a.RowIndex {
		if a.RowIndex[i] != b.RowIndex[i] {
			return false
		}
	}
	for i := range a.ColPtr {
		if a.ColPtr[i] != b.ColPtr[i] {
			return false
		}
	}
	return true
}

func BenchmarkSortRowEntries(b *testing.B) {
	sizes := []int{5, 50, 500, 5000}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			values := make([]float64, size)
			colInd := make([]int, size)
			for i := range values {
				values[i] = rand.Float64()
				colInd[i] = rand.Intn(size)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				valuesCopy := make([]float64, len(values))
				colIndCopy := make([]int, len(colInd))
				copy(valuesCopy, values)
				copy(colIndCopy, colInd)
				sortRowEntries(valuesCopy, colIndCopy)
			}
		})
	}
}

func TestIsSymmetric(t *testing.T) {
	tests := []struct {
		name    string
		matrix  *CSRMatrix
		want    bool
	}{
		{
			name: "symmetric matrix",
			matrix: &CSRMatrix{
				Values:   []float64{1, 2, 2, 3},
				ColInd:   []int{0, 1, 0, 1},
				RowPtr:   []int{0, 2, 4},
				Rows:     2,
				Cols:     2,
			},
			want: true,
		},
		{
			name: "non-symmetric matrix",
			matrix: &CSRMatrix{
				Values:   []float64{1, 2, 3},
				ColInd:   []int{0, 1, 2},
				RowPtr:   []int{0, 1, 3},
				Rows:     3,
				Cols:     3,
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.matrix.IsSymmetric()
			if got != tt.want {
				t.Errorf("IsSymmetric() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestClone(t *testing.T) {
	original := &CSRMatrix{
		Values:   []float64{1, 2, 3},
		ColInd:   []int{0, 1, 2},
		RowPtr:   []int{0, 1, 3},
		Rows:     3,
		Cols:     3,
	}
	clone := original.Clone()
	if !compareCSR(original, clone) {
		t.Errorf("Clone() did not produce an identical matrix")
	}
	// Modify original and check clone remains unchanged
	original.Values[0] = 10
	if compareCSR(original, clone) {
		t.Errorf("Clone() is not a deep copy")
	}
}

func TestTransposeMultiply(t *testing.T) {
	matrix := &CSRMatrix{
		Values:   []float64{1, 2, 3},
		ColInd:   []int{0, 1, 2},
		RowPtr:   []int{0, 1, 2, 3},  // Fixed: each row has 1 element
		Rows:     3,
		Cols:     3,
	}
	vector := []float64{1, 2, 3}
	want := []float64{1, 4, 9}
	got := matrix.TransposeMultiply(vector)
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("TransposeMultiply() = %v, want %v", got, want)
			break
		}
	}
}

func TestAdd(t *testing.T) {
	tests := []struct {
		name     string
		matrixA  [][]float64
		matrixB  [][]float64
		expected [][]float64
		wantErr  bool
	}{
		{
			name: "2x2 matrices",
			matrixA: [][]float64{
				{1, 2},
				{3, 4},
			},
			matrixB: [][]float64{
				{5, 6},
				{7, 8},
			},
			expected: [][]float64{
				{6, 8},
				{10, 12},
			},
			wantErr: false,
		},
		{
			name: "sparse matrices",
			matrixA: [][]float64{
				{1, 0, 0},
				{0, 2, 0},
				{0, 0, 3},
			},
			matrixB: [][]float64{
				{4, 0, 0},
				{0, 5, 0},
				{0, 0, 6},
			},
			expected: [][]float64{
				{5, 0, 0},
				{0, 7, 0},
				{0, 0, 9},
			},
			wantErr: false,
		},
		{
			name: "incompatible dimensions",
			matrixA: [][]float64{
				{1, 2},
				{3, 4},
			},
			matrixB: [][]float64{
				{5, 6, 7},
				{8, 9, 10},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, err := NewCSRMatrix(tt.matrixA)
			if err != nil {
				t.Fatalf("Failed to create matrix A: %v", err)
			}

			b, err := NewCSRMatrix(tt.matrixB)
			if err != nil {
				t.Fatalf("Failed to create matrix B: %v", err)
			}

			result, err := a.Add(b)
			if (err != nil) != tt.wantErr {
				t.Errorf("Add() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				expected, _ := NewCSRMatrix(tt.expected)
				if !compareCSR(result, expected) {
					t.Errorf("Add() got = %v, want %v", result.ToDense(), tt.expected)
				}
			}
		})
	}
}

func TestMaxAbs(t *testing.T) {
	tests := []struct {
		name     string
		matrix   [][]float64
		expected float64
	}{
		{
			name: "positive values",
			matrix: [][]float64{
				{1, 2},
				{3, 4},
			},
			expected: 4,
		},
		{
			name: "mixed signs",
			matrix: [][]float64{
				{-5, 2},
				{3, -1},
			},
			expected: 5,
		},
		{
			name: "sparse matrix",
			matrix: [][]float64{
				{0, -7, 0},
				{0, 0, 0},
				{2, 0, 3},
			},
			expected: 7,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewCSRMatrix(tt.matrix)
			if err != nil {
				t.Fatalf("Failed to create matrix: %v", err)
			}

			got := m.MaxAbs()
			if got != tt.expected {
				t.Errorf("MaxAbs() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestScale(t *testing.T) {
	tests := []struct {
		name     string
		matrix   [][]float64
		scalar   float64
		expected [][]float64
	}{
		{
			name: "scale by 2",
			matrix: [][]float64{
				{1, 2},
				{3, 4},
			},
			scalar: 2,
			expected: [][]float64{
				{2, 4},
				{6, 8},
			},
		},
		{
			name: "scale by 0",
			matrix: [][]float64{
				{1, 2},
				{3, 4},
			},
			scalar: 0,
			expected: [][]float64{
				{0, 0},
				{0, 0},
			},
		},
		{
			name: "scale sparse matrix",
			matrix: [][]float64{
				{0, 2, 0},
				{0, 0, 3},
				{1, 0, 0},
			},
			scalar: -1,
			expected: [][]float64{
				{0, -2, 0},
				{0, 0, -3},
				{-1, 0, 0},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewCSRMatrix(tt.matrix)
			if err != nil {
				t.Fatalf("Failed to create matrix: %v", err)
			}

			m.Scale(tt.scalar)
			expected, _ := NewCSRMatrix(tt.expected)
			if !compareCSR(m, expected) {
				t.Errorf("Scale() got = %v, want %v", m.ToDense(), tt.expected)
			}
		})
	}
}

func TestNewIdentityCSRMatrix(t *testing.T) {
	tests := []struct {
		name string
		size int
	}{
		{
			name: "1x1 identity",
			size: 1,
		},
		{
			name: "3x3 identity",
			size: 3,
		},
		{
			name: "5x5 identity",
			size: 5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := NewIdentityCSRMatrix(tt.size)

			// Check dimensions
			rows, cols := m.Dims()
			if rows != tt.size || cols != tt.size {
				t.Errorf("NewIdentityCSRMatrix() dimensions = %dx%d, want %dx%d",
					rows, cols, tt.size, tt.size)
			}

			// Check values
			for i := 0; i < tt.size; i++ {
				for j := 0; j < tt.size; j++ {
					expected := 0.0
					if i == j {
						expected = 1.0
					}
					if got := m.At(i, j); got != expected {
						t.Errorf("NewIdentityCSRMatrix() at (%d,%d) = %v, want %v",
							i, j, got, expected)
					}
				}
			}
		})
	}
}

func TestIsLowerTriangular(t *testing.T) {
	tests := []struct {
		name     string
		matrix   [][]float64
		expected bool
	}{
		{
			name: "lower triangular",
			matrix: [][]float64{
				{1, 0, 0},
				{2, 3, 0},
				{4, 5, 6},
			},
			expected: true,
		},
		{
			name: "not lower triangular",
			matrix: [][]float64{
				{1, 2, 0},
				{3, 4, 5},
				{6, 7, 8},
			},
			expected: false,
		},
		{
			name: "sparse lower triangular",
			matrix: [][]float64{
				{1, 0, 0},
				{0, 2, 0},
				{3, 0, 4},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewCSRMatrix(tt.matrix)
			if err != nil {
				t.Fatalf("Failed to create matrix: %v", err)
			}

			got := m.IsLowerTriangular()
			if got != tt.expected {
				t.Errorf("IsLowerTriangular() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestIsUpperTriangular(t *testing.T) {
	tests := []struct {
		name     string
		matrix   [][]float64
		expected bool
	}{
		{
			name: "upper triangular",
			matrix: [][]float64{
				{1, 2, 3},
				{0, 4, 5},
				{0, 0, 6},
			},
			expected: true,
		},
		{
			name: "not upper triangular",
			matrix: [][]float64{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
			},
			expected: false,
		},
		{
			name: "sparse upper triangular",
			matrix: [][]float64{
				{1, 0, 2},
				{0, 3, 0},
				{0, 0, 4},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewCSRMatrix(tt.matrix)
			if err != nil {
				t.Fatalf("Failed to create matrix: %v", err)
			}

			got := m.IsUpperTriangular()
			if got != tt.expected {
				t.Errorf("IsUpperTriangular() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestScaleVector(t *testing.T) {
	tests := []struct {
		name     string
		vector   []float64
		scalar   float64
		expected []float64
	}{
		{
			name:     "scale by 2",
			vector:   []float64{1, 2, 3},
			scalar:   2,
			expected: []float64{2, 4, 6},
		},
		{
			name:     "scale by 0",
			vector:   []float64{1, 2, 3},
			scalar:   0,
			expected: []float64{0, 0, 0},
		},
		{
			name:     "scale by negative",
			vector:   []float64{1, -2, 3},
			scalar:   -1,
			expected: []float64{-1, 2, -3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := make([]float64, len(tt.vector))
			copy(v, tt.vector)
			ScaleVector(v, tt.scalar)
			for i := range v {
				if v[i] != tt.expected[i] {
					t.Errorf("ScaleVector() at %d = %v, want %v", i, v[i], tt.expected[i])
				}
			}
		})
	}
}

func TestIsOrthogonal(t *testing.T) {
	tests := []struct {
		name     string
		matrix   [][]float64
		expected bool
	}{
		{
			name: "orthogonal matrix",
			matrix: [][]float64{
				{1/math.Sqrt(2), -1/math.Sqrt(2)},
				{1/math.Sqrt(2), 1/math.Sqrt(2)},
			},
			expected: true,
		},
		{
			name: "not orthogonal",
			matrix: [][]float64{
				{1, 1},
				{1, 1},
			},
			expected: false,
		},
		{
			name: "identity matrix",
			matrix: [][]float64{
				{1, 0},
				{0, 1},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewCSRMatrix(tt.matrix)
			if err != nil {
				t.Fatalf("Failed to create matrix: %v", err)
			}

			got := m.IsOrthogonal()
			if got != tt.expected {
				t.Errorf("IsOrthogonal() = %v, want %v", got, tt.expected)
			}
		})
	}
}
