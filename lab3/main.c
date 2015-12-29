#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "utils.h"

#define INTS_SIZE 16

int main(int argc, char** argv) {
  mpi_check(MPI_Init(&argc, &argv));
  // Set a sane error handler (return errors and don't kill our process)
  mpi_check(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

  int nprocs;
  mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
  int pid;
  mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &pid));

  int* recvcounts = calloc(nprocs, sizeof(int));
  err_check(recvcounts != NULL, "Couldn't allocate buffer");
  int* displs;
  if (pid == 0) {
    displs = calloc(nprocs, sizeof(int));
    err_check(displs != NULL, "Couldn't allocate buffer");
  }
  
  mpi_check(UMPI_Scatterv_lens(INTS_SIZE, recvcounts, displs, 0, MPI_COMM_WORLD));
  
  int recvlen = recvcounts[pid];
  int* recvbuf = calloc(recvlen, sizeof(int));
  err_check(recvbuf != NULL, "Couldn't allocate buffer");

  if (pid == 0) {
    srand(time(NULL));

    int msg[INTS_SIZE];
    for (int i = 0; i < INTS_SIZE; i++) {
      msg[i] = rand();
    }

    printf("Sending source message: ");
    for (int i = 0; i < INTS_SIZE; i++) {
      printf("%i ", msg[i]);
    }
    printf("\n");

    mpi_check(MPI_Scatterv(msg, recvcounts, displs, MPI_INT, recvbuf, recvlen, MPI_INT, 0, MPI_COMM_WORLD));
  } else {
    mpi_check(MPI_Scatterv(NULL, NULL, NULL, MPI_DATATYPE_NULL, recvbuf, recvlen, MPI_INT, 0, MPI_COMM_WORLD));
  }

  printf("Node %i got chunk of size %i: ", pid, recvlen);
  for (int i = 0; i < recvlen; i++) {
    printf("%i ", recvbuf[i]);
  }
  printf("\n");

  for (int i = 0; i < recvlen; i++) {
    recvbuf[i] += 1;
  }

  if (pid != 0) {
    mpi_check(MPI_Gatherv(recvbuf, recvlen, MPI_INT, NULL, NULL, NULL, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD));
  } else {
    int msg[INTS_SIZE];
    mpi_check(MPI_Gatherv(recvbuf, recvlen, MPI_INT, msg, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD));

    printf("Result: ");
    for (int i = 0; i < INTS_SIZE; i++) {
      printf("%i ", msg[i]);
    }
    printf("\n");
  }

  mpi_check(MPI_Finalize());
  return 0;
}
