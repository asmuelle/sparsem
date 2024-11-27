package sparsem

import (
	"math"
	"testing"
)

func TestNewCSRMatrix(t *testing.T) {
	dense := [][]float64{
		{1, 0, 2},
		{0, 3, 0},
		{4, 0, 5},
	}
	
	matrix := NewCSRMatrix(dense)
	
	if matrix.Rows != 3 || matrix.Cols != 3 {
		t.Errorf("Expected 3x3 matrix, got %dx%d", matrix.Rows, matrix.Cols)
	}
	
	// Check number of non-zero elements
	if len(matrix.Values) != 5 {
		t.Errorf("Expected 5 non-zero elements, got %d", len(matrix.Values))
	}
}

func TestMatrixMultiplication(t *testing.T) {
	a := NewCSRMatrix([][]float64{
		{1, 0},
		{0, 2},
	})
	
	b := NewCSRMatrix([][]float64{
		{3, 0},
		{0, 4},
	})
	
	c, err := a.Multiply(b)
	if err != nil {
		t.Fatalf("Multiplication failed: %v", err)
	}
	
	expected := [][]float64{
		{3, 0},
		{0, 8},
	}
	
	result := c.ToDense()
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if math.Abs(result[i][j]-expected[i][j]) > 1e-10 {
				t.Errorf("Expected %f at (%d,%d), got %f", expected[i][j], i, j, result[i][j])
			}
		}
	}
}

func TestCholesky(t *testing.T) {
	// Test matrix must be symmetric positive definite
	matrix := NewCSRMatrix([][]float64{
		{4, 1, 0},
		{1, 2, 1},
		{0, 1, 2},
	})
	
	chol, err := matrix.Cholesky()
	if err != nil {
		t.Fatalf("Cholesky decomposition failed: %v", err)
	}
	
	// Test solving a system
	b := []float64{1, 2, 3}
	x, err := chol.Solve(b)
	if err != nil {
		t.Fatalf("Solving system failed: %v", err)
	}
	
	// Verify solution by multiplying Ax
	dense := matrix.ToDense()
	result := make([]float64, len(b))
	for i := 0; i < len(b); i++ {
		for j := 0; j < len(b); j++ {
			result[i] += dense[i][j] * x[j]
		}
	}
	
	// Check if Ax ≈ b
	for i := 0; i < len(b); i++ {
		if math.Abs(result[i]-b[i]) > 1e-10 {
			t.Errorf("Solution verification failed at index %d: expected %f, got %f",
				i, b[i], result[i])
		}
	}
}
