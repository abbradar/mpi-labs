#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

#define ERR_SIZE (MPI_MAX_ERROR_STRING * 2 + 255)

void __err_class_check(int res, const char* cat, const char* error, const char* file, int line)
{
  if (!res) {
    fprintf(stderr, "%s [%i]: %s, %s\n", file, line, cat, error);
    exit(1);
  }
}

#define mpierr_check(res, error) err_class_check((res), "MPI error", (error))

void __mpi_check(int res, const char* file, const int line)
{
  if (res != MPI_SUCCESS) {
    char errbuf[MPI_MAX_ERROR_STRING];
    int errlen;
    mpierr_check(MPI_Error_string(res, errbuf, &errlen) == MPI_SUCCESS, "MPI_Error_string");

    int class;
    mpierr_check(MPI_Error_class(res, &class) == MPI_SUCCESS, "MPI_Error_class");
    
    char classbuf[MPI_MAX_ERROR_STRING];
    int classlen;
    mpierr_check(MPI_Error_string(class, classbuf, &classlen) == MPI_SUCCESS, "MPI_Error_string");

    char buf[ERR_SIZE];
    int buflen = snprintf(buf, ERR_SIZE, "Class %i (%s), error %i (%s)", class, classbuf, res, errbuf);
    mpierr_check(buflen < ERR_SIZE && buflen >= 0, "Cannot place error into buffer");
    __err_class_check(1, "MPI error", buf, file, line);
  }
}

#undef mpierr_check

int UMPI_Scatterv_lens(int count, int* recvcounts, int* displs, int root, MPI_Comm comm)
{
  int ret = MPI_SUCCESS;

  int nprocs;
  if ((ret = MPI_Comm_size(comm, &nprocs)) != MPI_SUCCESS) return ret;
  int pid;
  if ((ret = MPI_Comm_rank(comm, &pid)) != MPI_SUCCESS) return ret;

  if (recvcounts == NULL) return MPI_ERR_BUFFER;
  if (displs == NULL && pid == root) return MPI_ERR_BUFFER;

  int taken = 0;
  for (int i = 0; i < nprocs; i++) {
    int take = (count - taken) / (nprocs - i);
    recvcounts[i] = take;
    if (pid == root) {
      displs[i] = taken;
    } else {
      if (i == pid) break;
    }
    taken += take;
  }
  if (pid == root) {
    err_check(taken == count, "Invalid scattering result");
  }

  return MPI_SUCCESS;
}
