#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "utils.h"

#define MTX_SIZE 16

int main(int argc, char** argv) {
  mpi_check(MPI_Init(&argc, &argv));
  // Set a sane error handler (return errors and don't kill our process)
  mpi_check(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

  int nprocs;
  mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
  err_check(nprocs == 1, "Not a distributed program");

	int a[MTX_SIZE][MTX_SIZE];
	int b[MTX_SIZE][MTX_SIZE];

	MPI_Datatype column, rev_matrix;

	mpi_check(MPI_Type_vector(MTX_SIZE, 1, MTX_SIZE, MPI_INT, &column));
	mpi_check(MPI_Type_create_hvector(MTX_SIZE, 1, sizeof(int), column, &rev_matrix));
	mpi_check(MPI_Type_commit(&rev_matrix));

  {
    int i = 0;
    for (int y = 0; y < MTX_SIZE; y++)
      for (int x = 0; x < MTX_SIZE; x++) {
        a[y][x] = i;
        i++;
      }
  }

  printf("Source matrix:\n");
  for (int y = 0; y < MTX_SIZE; y++) {
    for (int x = 0; x < MTX_SIZE; x++) {
      printf("%i ", a[y][x]);
    }
    printf("\n");
  }

	mpi_check(MPI_Sendrecv(a, MTX_SIZE * MTX_SIZE, MPI_INT, 0, 0, b, 1, rev_matrix, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
		
	printf("Result matrix:\n");
  for (int y = 0; y < MTX_SIZE; y++) {
    for (int x = 0; x < MTX_SIZE; x++) {
      printf("%i ", b[y][x]);
    }
    printf("\n");
  }

	mpi_check(MPI_Type_free(&rev_matrix));
	mpi_check(MPI_Type_free(&column));

  mpi_check(MPI_Finalize());
  return 0;
}
