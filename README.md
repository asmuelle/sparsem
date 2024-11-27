# SparseM

A Go implementation of sparse matrix operations, inspired by the R package SparseM.

## Features

- Compressed Sparse Row (CSR) matrix format
- Basic matrix operations:
  - Matrix addition
  - Matrix multiplication
  - Conversion between dense and sparse formats
- Advanced linear algebra operations:
  - Cholesky decomposition
  - Linear system solving using Cholesky decomposition
- Matrix properties:
  - Symmetry checking

## Installation

```bash
go get github.com/andreasmuller/sparsem
```

## Usage

```go
package main

import (
    "fmt"
    "github.com/andreasmuller/sparsem"
)

func main() {
    // Create a sparse matrix from dense format
    dense := [][]float64{
        {1, 0, 0},
        {0, 2, 0},
        {0, 0, 3},
    }
    matrix := sparsem.NewCSRMatrix(dense)

    // Perform Cholesky decomposition
    chol, err := matrix.Cholesky()
    if err != nil {
        panic(err)
    }

    // Solve a linear system Ax = b
    b := []float64{1, 2, 3}
    x, err := chol.Solve(b)
    if err != nil {
        panic(err)
    }
    fmt.Println("Solution:", x)
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
