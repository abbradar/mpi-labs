#ifndef UTILS_H
#define UTILS_H

#include <mpi.h>

// I'm disgusted by `"error message" && check` trick enough to define my own
// assert-like functions.
#define err_class_check(res, cat, error) __err_class_check((res), (cat), (error), __FILE__, __LINE__)

void __err_class_check(int res, const char* cat, const char* error, const char* file, int line);

#define mpi_check(call) __mpi_check((call), __FILE__, __LINE__)

void __mpi_check(int res, const char* file, const int line);

#define err_check(res, error) err_class_check((res), "Program error", (error))

#define divup(a, b) ((a) / (b)) + ((a) % (b) == 0 ? 0 : 1)

// If current rank is not root, only recvcounts[pid] is guaranteed to be correct.
int UMPI_Scatterv_lens(int count, int* recvcounts, int* displs, int root, MPI_Comm comm);

#endif
