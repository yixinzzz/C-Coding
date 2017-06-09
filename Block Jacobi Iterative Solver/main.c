/*
    Yixin Zhang
    Amath 483
    HW #8
    06/07/15
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "MyHeader.h"


int main(int argc, char *argv[])
{

  /* Standard C90 requires to declare variables before any code */

  int n;
  int i, j, k, jj;
  int numThreads;
  int my_rank, numProc;
  int myNumRow, myRowShift;
    

  struct SparseMatrix A;

  double normbSqr = 0.0, normRSqr, tmpDouble = 0.0;
  struct Vector b, x, r, actual_x, total_x;

  FILE *out;

  double tol = 1.0e-04, tolSqr = 1.0e-08;
  int kmax;

  double timer = 0.0, time01 = 0.0, time02 = 0.0;

  /* --- Start coding section --- */

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &numProc);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (argc <= 2)
  {
    if (my_rank == 0)
    printf("\n ... The code requires two integer parameters ...\n\n");
    MPI_Finalize();
    return -1;
  }

  /* Get the integer input as matrix dimension */

  {
    char *ptr;
    n = (int) strtol(argv[1], &ptr, 10); 
    numThreads = (int) strtol(argv[2], &ptr, 10);
  }

  myNumRow = n / numProc;
  myRowShift = my_rank * myNumRow;

  /* Create the sparse matrix */

  A.m = myNumRow;
  A.n = n;

  A.iRow = (int*) malloc(sizeof(int)*(myNumRow+1));
  A.jCol = (int*) malloc(sizeof(int)*3*myNumRow);
  A.values = (double*) malloc(sizeof(double)*3*myNumRow);

  A.iRow[0] = 0;
  for (i = 0; i < myNumRow; ++i)
  {
    int myRow = i + myRowShift;
    if (myRow == 0)  
    {
      A.values[A.iRow[i]] = 2.0;
      A.jCol[A.iRow[i]] = myRow;
      A.values[A.iRow[i]+1] = -1.0;
      A.jCol[A.iRow[i]+1] = myRow+1;
      A.iRow[i+1] = A.iRow[i] + 2;
    }
    else if (myRow == n - 1)
    {
      A.values[A.iRow[i]] = -1.0;
      A.jCol[A.iRow[i]] = myRow-1;
      A.values[A.iRow[i]+1] = 2.0;
      A.jCol[A.iRow[i]+1] = myRow;
      A.iRow[i+1] = A.iRow[i] + 2;
    }
    else
    {
      A.values[A.iRow[i]] = -1.0;
      A.jCol[A.iRow[i]] = myRow - 1;
      A.values[A.iRow[i]+1] = 2.0;
      A.jCol[A.iRow[i]+1] = myRow;
      A.values[A.iRow[i]+2] = -1.0;
      A.jCol[A.iRow[i]+2] = myRow+1;
      A.iRow[i+1] = A.iRow[i] + 3;
    }
  } 

  /* Create a vector x and actual x */
    actual_x.n = myNumRow;
    actual_x.values = (double*) malloc(sizeof(double)*actual_x.n);
    for (i = 0; i < actual_x.n; ++i)
    {
        actual_x.values[i] = 0.0;
    }
    
    if (numProc > 1) {
        if ((my_rank == 0) || (my_rank == myNumRow - 1))
        {
            x.n = myNumRow + 1;
        } else {
            x.n = myNumRow + 2;
        }
    } else {
        x.n = myNumRow;
    }
    x.values = (double*) malloc(sizeof(double)*x.n);
    for (i = 0; i < x.n; ++i)
    {
        x.values[i] = 0.0;
    }

    total_x.n = n;
    total_x.values = (double*) malloc(sizeof(double)*n);
    
  /* Create a right hand side (of local size) */

  b.n = A.m;
  b.values = (double*) malloc(sizeof(double)*b.n);
  for (i = 0; i < b.n; ++i)
    b.values[i] = 1.0;
  for (i = 0; i < b.n; ++i)
    normbSqr += b.values[i] * b.values[i];

  tmpDouble = normbSqr;
  MPI_Allreduce(&tmpDouble, &normbSqr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
  /* Create a residual vector (of local size) */

  r.n = b.n;
  r.values = (double*) malloc(sizeof(double)*r.n);

  /* Loop for the Jacobi iterative algorithm */
    
  if (n < 268000)
  {
    kmax = 8000 * n;
  }
  else
  {
    /* Use the largest integral value - 1 */
    kmax = 214783646; 
  }  

  timer -= getTime();
  for (k = 1; k < kmax; ++k)
  {
    normRSqr = 0.0;

    time01 -= getTime();
      
      //Update x
        if (numProc > 1)
        {
            if (my_rank == 0)
            {
                #   pragma omp parallel for num_threads(numThreads) private(i)
                for(i = 0; i < actual_x.n; i++){
                    x.values[i] = actual_x.values[i];
                }
                MPI_Sendrecv(&actual_x.values[myNumRow-1], 1, MPI_DOUBLE, 1, 0, 
                             &x.values[myNumRow], 1, MPI_DOUBLE, 1, 0, 
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else { 
                #   pragma omp parallel for num_threads(numThreads) private(i)
                for(i = 0; i < actual_x.n; i++){
                    x.values[i+1] = actual_x.values[i];
                }
                if (my_rank == numProc-1) // the last p
                {
                     MPI_Sendrecv(&actual_x.values[0], 1, MPI_DOUBLE, my_rank-1, 0, 
                                  &x.values[0], 1, MPI_DOUBLE, my_rank-1, 0, 
                                  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                     MPI_Sendrecv(&actual_x.values[0], 1, MPI_DOUBLE, my_rank-1, 0, 
                                  &x.values[0], 1, MPI_DOUBLE, my_rank-1, 0, 
                                  MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                     MPI_Sendrecv(&actual_x.values[myNumRow-1], 1, MPI_DOUBLE, my_rank+1, 0, 
                                  &x.values[myNumRow+1], 1, MPI_DOUBLE, my_rank+1, 0, 
                                  MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                }
            }
        } else {
            #   pragma omp parallel for num_threads(numThreads) private(i)
            for (i = 0; i < actual_x.n; i++)
            {
                x.values[i] = actual_x.values[i];
            }
        }
   // compute A*x, r, norm
#   pragma omp parallel for num_threads(numThreads) \
    reduction(+: normRSqr) private(i)
        for (i = 0; i < A.m; ++i)
        {
            double ax = 0.0, res;
            for (jj = A.iRow[i]; jj < A.iRow[i+1]; ++jj)
            {
                if(my_rank == 0) {
                    ax += A.values[jj] * x.values[A.jCol[jj]];
                } else {
                    ax += A.values[jj] * x.values[A.jCol[jj]-myRowShift+1];
                }
            }
            res = b.values[i] - ax;
            r.values[i] = res;
            normRSqr += res*res;
        }
        tmpDouble = normRSqr;
        MPI_Allreduce(&tmpDouble, &normRSqr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        time01 += getTime();

    /* Test for convergence */
    if (normRSqr < tolSqr * normbSqr)
    {
      if (my_rank == 0)
        printf(" ... Iterate %d ... Norm Reduction %4.3e\n", k-1, sqrt(normRSqr/normbSqr));
      break;
    }

#ifdef VERBOSE
    if ((k < 25) || (k % 5000 == 1) || ((k < 5000) && (k % 100 == 1)))
    {
      if (my_rank == 0)
        printf(" ... Iterate %d ... Norm Reduction %4.3e\n", k-1, sqrt(normRSqr/normbSqr));
    }
#endif

    /* Compute D^{-1} * r and update x */
    time02 -= getTime();
#   pragma omp parallel for num_threads(numThreads) private(i)
    for (i = 0; i < actual_x.n; ++i) 
    {
        actual_x.values[i] +=  r.values[i]/2.0;
    }
    MPI_Allgather(actual_x.values, actual_x.n, MPI_DOUBLE, total_x.values, actual_x.n, MPI_DOUBLE, MPI_COMM_WORLD);
    time02 += getTime();

    if ((k == 1) && (my_rank == 0))
    {
      out = fopen("x1.txt", "w");
      PrintVector(out, &total_x);
      fclose(out);
    }

    if ((k == 2) && (my_rank == 0))
    {
      out = fopen("x2.txt", "w");
      PrintVector(out, &total_x);
      fclose(out);
    }

  }
  
  timer += getTime();

  if (my_rank == 0)
  {
    if (normRSqr >= tolSqr * normbSqr)
    {
      printf(" ... Jacobi did not converge after %d iterations ... \n", k);
      printf(" ... Final Residual Norm Reduction = %e ... \n", sqrt(normRSqr/normbSqr));
    }
    else
    {
      out = fopen("xfinal.txt", "w");
      PrintVector(out, &total_x);
      fclose(out);
    }
    
  }

  tmpDouble = timer;
  MPI_Reduce(&tmpDouble, &timer, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  tmpDouble = time01;
  MPI_Reduce(&tmpDouble, &time01, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  tmpDouble = time02;
  MPI_Reduce(&tmpDouble, &time02, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (my_rank == 0)
  {
    printf("\n ... Time Total: %e s (%e, %e) \n\n",
           timer, time01, time02);
    out = fopen("time.txt", "w");
    fprintf(out, "%e %e %e \n", timer, time01, time02);
    fclose(out);
  }

  /* Free the matrix A */

  free(A.iRow);
  free(A.jCol);
  free(A.values);


  /* Free the vectors b, x, r */

  free(b.values);
  free(x.values);
  free(actual_x.values);
  free(r.values);
  free (total_x.values);

  MPI_Finalize();

  return 0;

}



