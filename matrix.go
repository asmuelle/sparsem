// Package sparsem provides sparse matrix operations similar to R's SparseM package
package sparsem

import (
	"fmt"
	"math"
)

// CSRMatrix represents a sparse matrix in Compressed Sparse Row format
type CSRMatrix struct {
	Values    []float64 // Non-zero values
	ColIndex  []int     // Column indices for values
	RowPtr    []int     // Row pointers into values/colIndex
	Rows      int       // Number of rows
	Cols      int       // Number of columns
}

// NewCSRMatrix creates a new CSR matrix from dense format
func NewCSRMatrix(dense [][]float64) *CSRMatrix {
	if len(dense) == 0 {
		return &CSRMatrix{
			Values:    []float64{},
			ColIndex:  []int{},
			RowPtr:    []int{0},
			Rows:      0,
			Cols:      0,
		}
	}

	rows := len(dense)
	cols := len(dense[0])
	var values []float64
	var colIndex []int
	rowPtr := []int{0}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if dense[i][j] != 0 {
				values = append(values, dense[i][j])
				colIndex = append(colIndex, j)
			}
		}
		rowPtr = append(rowPtr, len(values))
	}

	return &CSRMatrix{
		Values:    values,
		ColIndex:  colIndex,
		RowPtr:    rowPtr,
		Rows:      rows,
		Cols:      cols,
	}
}

// ToDense converts the CSR matrix to dense format
func (m *CSRMatrix) ToDense() [][]float64 {
	dense := make([][]float64, m.Rows)
	for i := range dense {
		dense[i] = make([]float64, m.Cols)
	}

	for i := 0; i < m.Rows; i++ {
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			dense[i][m.ColIndex[j]] = m.Values[j]
		}
	}

	return dense
}

// Multiply performs matrix multiplication: this * other
func (m *CSRMatrix) Multiply(other *CSRMatrix) (*CSRMatrix, error) {
	if m.Cols != other.Rows {
		return nil, fmt.Errorf("matrix dimensions do not match for multiplication: %dx%d * %dx%d",
			m.Rows, m.Cols, other.Rows, other.Cols)
	}

	result := make([][]float64, m.Rows)
	for i := range result {
		result[i] = make([]float64, other.Cols)
	}

	for i := 0; i < m.Rows; i++ {
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			val := m.Values[j]
			col := m.ColIndex[j]
			
			for k := other.RowPtr[col]; k < other.RowPtr[col+1]; k++ {
				result[i][other.ColIndex[k]] += val * other.Values[k]
			}
		}
	}

	return NewCSRMatrix(result), nil
}

// Add performs matrix addition: this + other
func (m *CSRMatrix) Add(other *CSRMatrix) (*CSRMatrix, error) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		return nil, fmt.Errorf("matrix dimensions do not match for addition: %dx%d + %dx%d",
			m.Rows, m.Cols, other.Rows, other.Cols)
	}

	result := make([][]float64, m.Rows)
	for i := range result {
		result[i] = make([]float64, m.Cols)
	}

	// Add first matrix
	for i := 0; i < m.Rows; i++ {
		for j := m.RowPtr[i]; j < m.RowPtr[i+1]; j++ {
			result[i][m.ColIndex[j]] += m.Values[j]
		}
	}

	// Add second matrix
	for i := 0; i < other.Rows; i++ {
		for j := other.RowPtr[i]; j < other.RowPtr[i+1]; j++ {
			result[i][other.ColIndex[j]] += other.Values[j]
		}
	}

	return NewCSRMatrix(result), nil
}

// IsSymmetric checks if the matrix is symmetric
func (m *CSRMatrix) IsSymmetric() bool {
	if m.Rows != m.Cols {
		return false
	}

	dense := m.ToDense()
	for i := 0; i < m.Rows; i++ {
		for j := i + 1; j < m.Cols; j++ {
			if math.Abs(dense[i][j] - dense[j][i]) > 1e-10 {
				return false
			}
		}
	}
	return true
}
