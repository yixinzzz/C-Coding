/*
 */


#include <stdio.h>


struct SparseMatrix
{
  int m, n;
  int *iRow;
  int *jCol;
  double *values;
};


struct Vector
{
  int n;
  double *values;
};


/* --- Create a sparse matrix --- */
struct SparseMatrix* CreateMatrix(int n, int comm_sz, int my_rank);


/* ---- Sparse Matrix Multiplication --- */
void MatVec(const struct SparseMatrix *A, struct Vector *x, struct Vector *y, int thread_count);


/* --- Print a sparse matrix --- */
void Printf(FILE *stream, const struct SparseMatrix *A);


/* --- Print a sparse matrix --- */
void PrintVector(FILE *stream, const struct Vector *b);


/* --- Timer Function --- */
double getTime();


