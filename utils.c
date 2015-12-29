#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
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
    int buflen = snprintf(buf, ERR_SIZE, "class %i (%s), error %i (%s)", class, classbuf, res, errbuf);
    mpierr_check(buflen < ERR_SIZE && buflen >= 0, "cannot place error into buffer");
    __err_class_check(1, "MPI error", buf, file, line);
  }
}

#undef mpierr_check
