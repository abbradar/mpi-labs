#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "utils.h"

#define TIMES 10
#define INTS_SIZE 16

int main(int argc, char** argv) {
  mpi_check(MPI_Init(&argc, &argv));
  // Set a sane error handler (return errors and don't kill our process)
  mpi_check(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

  int nprocs;
  mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
  err_check(nprocs % 2 == 0, "Needs mean number of workers");
  int pid;
  mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &pid));

  // Stupid but should work if clocks are synchronized more or less.
  // We wouldn't want a cluster with two seconds drift, right?
  srand(time(NULL) + pid * 2);

  for (int time = 0; time < TIMES; time++) {
    if (pid % 2 == 0) {
      int server = pid + 1;

      int msg[INTS_SIZE];
      for (int i = 0; i < INTS_SIZE; i++) {
        msg[i] = rand();
      }

      mpi_check(MPI_Send(msg, INTS_SIZE, MPI_INT, server, 0, MPI_COMM_WORLD));

      int msgback[INTS_SIZE];
      mpi_check(MPI_Recv(msgback, INTS_SIZE, MPI_INT, server, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      err_check(memcmp(msg, msgback, INTS_SIZE * sizeof(int)) == 0, "Ping message is corrupted");

      int sum = 0;
      for (int i = 0; i < INTS_SIZE; i++) {
        sum += msgback[i];
      }
      printf("Rank: %d, got valid pong from %i, checksum %i\n", pid, server, sum);
    } else {
      MPI_Status status;
      mpi_check(MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status));

      int len;
      mpi_check(MPI_Get_count(&status, MPI_INT, &len));
      err_check(len != MPI_UNDEFINED, "Undefined message length");

      int* buf = calloc(len, sizeof(int));
      err_check(buf != NULL, "Couldn't allocate buffer");

      mpi_check(MPI_Recv(buf, len, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      printf("Rank: %d, got ping from %i, sending back\n", pid, status.MPI_SOURCE);
      mpi_check(MPI_Send(buf, len, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD));
      
      free(buf);
    }
  }

  mpi_check(MPI_Finalize());
  return 0;
}
