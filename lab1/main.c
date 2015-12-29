#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "utils.h"

#define BUF_SIZE 255

int main(int argc, char** argv) {
  mpi_check(MPI_Init(&argc, &argv));
  // Set a sane error handler (return errors and don't kill our process)
  mpi_check(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

  int nprocs;
  mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
  int pid;
  mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &pid));

  if (pid == 0) {
    char buf[BUF_SIZE];
    int len = snprintf(buf, BUF_SIZE, "Hello, World!");
    err_check(len < BUF_SIZE && len >= 0, "Couldn't format string");

    for (int i = 1; i < nprocs; i++)
    {
      mpi_check(MPI_Send(buf, len + 1, MPI_CHAR, i, 0, MPI_COMM_WORLD));
    }
  } else {
    char* buf;
    int len;

    {
      MPI_Status status;
      mpi_check(MPI_Probe(0, 0, MPI_COMM_WORLD, &status));

      mpi_check(MPI_Get_count(&status, MPI_CHAR, &len));
      err_check(len != MPI_UNDEFINED, "Undefined message length");
    }

    buf = calloc(len, sizeof(char));
    err_check(buf != NULL, "Couldn't allocate buffer");

    mpi_check(MPI_Recv(buf, len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    printf("Rank: %d, message: %s\n", pid, buf);

    free(buf);
  }

  mpi_check(MPI_Finalize());
  return 0;
}
