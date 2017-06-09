
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>


#include "MyHeader.h"


/* --- Output a vector --- */
void PrintVector(FILE *stream, const struct Vector *b)
{

  int i;
  for (i = 0; i < b->n; ++i)
  {
      fprintf(stream, "%e\n", b->values[i]);
  }

}


/* --- Function to get current time --- */
double getTime() 
{

  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec + tp.tv_usec/1000000.0;

}


