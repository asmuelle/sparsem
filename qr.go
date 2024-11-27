package sparsem

// QR performs QR decomposition of matrix A = QR where Q is orthogonal and R is upper triangular.
// This implementation uses Householder reflections for better numerical stability.
func (m *CSRMatrix) QR() (*QRDecomposition, error) {
	return NewQRDecomposition(m)
}
