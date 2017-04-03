function res = isMatrixEquals(matrixA, matrixB)

 res = sum(sum(matrixA==matrixB)) == size(matrixB,1) * size(matrixB,2)