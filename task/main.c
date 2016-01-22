#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#ifdef OMPI
#ifdef OMPI_PCONTROL
#include <pcontrol.h>
#endif
#include <mpi.h>
#endif

#include "utils.h"

#ifndef offsetof
#define offsetof(type, member)  __builtin_offsetof(type, member)
#endif

#define FLOAT_FMT "%8.2g"

#define sqr(x) ({             \
      __typeof__(x) _x = (x); \
      _x * _x;                \
    })

#define max(a, b) ({          \
      __typeof__(a) _a = (a); \
      __typeof__(b) _b = (b); \
      _a > _b ? _a : _b;      \
    })

#define min(a, b) ({          \
      __typeof__(a) _a = (a); \
      __typeof__(b) _b = (b); \
      _a < _b ? _a : _b;      \
    })

struct matrix {
  unsigned w;
  unsigned h;
  double data[];
};

struct matrix* newmatrix(unsigned w, unsigned h)
{
  struct matrix* mtx = calloc(1, sizeof(struct matrix) + w * h * sizeof(double));
  if (mtx != NULL) {
    mtx->w = w;
    mtx->h = h;
  }
  return mtx;
}

void freematrix(struct matrix* mtx)
{
  free(mtx);
}

#define val(mtx, x, y) (mtx->data[y * mtx->w + x])

void mfprint(FILE* file, const struct matrix* mtx)
{
  for (unsigned y = 0; y < mtx->h; ++y) {
    if (mtx->w > 0) {
      fprintf(file, FLOAT_FMT, val(mtx, 0, y));
      for (unsigned x = 1; x < mtx->w; ++x) {
        fprintf(file, " " FLOAT_FMT, val(mtx, x, y));
      }
    }
    fprintf(file, "\n");
  }
  fprintf(file, "\n");
}

void mmultiply(const struct matrix* a, const struct matrix* b, struct matrix* out)
{
  assert(a->w == b->h);
  assert(out->w == b->w && out->h == a->h);
#ifdef OMP
  #pragma omp for
#endif
  for (unsigned y = 0; y < out->h; y++) {
    for (unsigned x = 0; x < out->w; x++) {
      double r = 0;
      for (unsigned i = 0; i < a->w; i++) {
        r += val(a, i, y) * val(b, x, i);
      }
      val(out, x, y) = r;
    }
  }
}

// Householder method
// http://www.aip.de/groups/soe/local/numres/bookcpdf/c11-2.pdf
void tridiagonalize(struct matrix* mtx)
{
  assert(mtx->w == mtx->h);
#ifdef OMP
  #pragma omp parallel
#endif
  if (mtx->w > 0 && mtx->h > 0) {
    struct matrix* u = alloc_check(newmatrix(1, mtx->h));
    struct matrix* p = alloc_check(newmatrix(1, mtx->h));

    for (unsigned i = 1; i < mtx->w - 1; i++) {
      unsigned cx = i - 1;

      double mx = 0;
      for (unsigned y = i; y < mtx->h; y++) {
        mx += sqr(val(mtx, cx, y));
      }
      mx = sqrt(mx);

      for (unsigned y = 0; y < i; y++) {
        val(u, 0, y) = 0;
      }
      for (unsigned y = i; y < mtx->h; y++) {
        val(u, 0, y) = val(mtx, cx, y);
      }
      val(u, 0, i) -= mx;

      double mu2 = 0;
      for (unsigned y = 0; y < mtx->h; y++) {
        mu2 += sqr(val(u, 0, y));
      }

      mmultiply(mtx, u, p);
      for (unsigned y = 0; y < mtx->h; y++) {
        val(p, 0, y) /= mu2 / 2;
      }

      double k = 0;
      for (unsigned j = 0; j < mtx->h; j++) {
        k += val(u, 0, j) * val(p, 0, j) / mu2;
      }

      for (unsigned y = 0; y < mtx->h; y++) {
        val(p, 0, y) -= k * val(u, 0, y);
      }

#ifdef OMP
      #pragma omp for
#endif
      for (unsigned y = cx; y < mtx->h; y++) {
        for (unsigned x = cx; x <= y; x++) {
          val(mtx, x, y) -= val(p, 0, y) * val(u, 0, x) + val(u, 0, y) * val(p, 0, x);
          val(mtx, y, x) = val(mtx, x, y);
        }
      }
    }
    freematrix(u);
    freematrix(p);
  }
}

struct tridiag {
  unsigned len;
  double* d;
  double* e;
};

struct tridiag* newdiag(unsigned len)
{
  assert(len > 0);
  struct tridiag* diag = malloc(sizeof(struct tridiag));
  double* d = calloc(len, sizeof(double));
  double* e = calloc(len - 1, sizeof(double));
  if (diag == NULL || d == NULL || e == NULL) {
    if (diag != NULL) free(diag);
    if (d != NULL) free(d);
    if (e != NULL) free(e);
    return NULL;
  }
  diag->len = len;
  diag->d = d;
  diag->e = e;
  return diag;
}

void freediag(struct tridiag* diag)
{
  free(diag->d);
  free(diag->e);
  free(diag);
}

void todiagonals(const struct matrix* mtx, struct tridiag* diag)
{
  assert(mtx->w == mtx->h);
  assert(mtx->w == diag->len);
  for (unsigned i = 0; i < mtx->h; i++) {
    diag->d[i] = val(mtx, i, i);
  }
  for (unsigned i = 0; i < mtx->h - 1; i++) {
    diag->e[i] = val(mtx, i, i + 1);
  }
}

#define nextsturm(nq) {    \
    double curr_q = (nq);  \
    if (curr_q < 0) res++; \
    prev_q = curr_q;       \
  }

// Sturm sequence-augmented bisection
// http://www3.uji.es/~badia/pubs/nova98.pdf
unsigned sturmsigns(const struct tridiag* diag, double c)
{
  unsigned res = 0;
  double prev_q = 1;
  if (diag->len > 0) {
    nextsturm(diag->d[0] - c);
    for (unsigned i = 1; i < diag->len; i++) {
      nextsturm((diag->d[i] - c) - sqr(diag->e[i - 1]) / prev_q);
    }
  }
  return res;
}

#undef nextsturm

struct border {
  unsigned na;
  double a;
  double b;
};

void findborders(const struct tridiag* diag, double a, unsigned na, double b, unsigned nb, struct border* borders, unsigned* borders_num)
{
  assert(b >= a);
  assert(nb >= na);
  if (nb - na == 1) {
    borders[*borders_num].na = na;
    borders[*borders_num].a = a;
    borders[*borders_num].b = b;
    (*borders_num)++;
  } else if (nb - na > 1) {
    double c = (a + b) / 2;
    unsigned nc = sturmsigns(diag, c);
    findborders(diag, a, na, c, nc, borders, borders_num);
    findborders(diag, c, nc, b, nb, borders, borders_num);
  }
}

double bisect(const struct tridiag* diag, unsigned na, double a, double b, double acc)
{
  double res;

  do {
    res = (a + b) / 2;
    if (sturmsigns(diag, res) == na) {
      b = res;
    } else {
      a = res;
    }
  } while (fabs(a - b) > acc);

  return res;
}

void eigenvalues_borders(const struct tridiag* diag, double a, double b, struct border* borders)
{
    // First, find real eigenvalues' borders.
    while (sturmsigns(diag, a) > 0) a -= fabs(a) + 0.1;
    while (sturmsigns(diag, b) != diag->len) b += fabs(b) + 0.1;
    unsigned borders_num = 0;
    findborders(diag, a, 0, b, diag->len, borders, &borders_num);
    assert(borders_num == diag->len);
}

int main(int argc, char** argv)
{
  const int mtx_size = 20;
  const double range_from = -1;
  const double range_to = 1;
  const double precision = 0.0001;

#ifdef OMPI
  mpi_check(MPI_Init(&argc, &argv));
  // Set a sane error handler (return errors and don't kill our process)
  mpi_check(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

#ifdef OMPI_PCONTROL
  // Set trace parameters
  MPI_Pcontrol(TRACELEVEL, 1, 1, 1);
  {
    char tmpname[255] = "task.trace.tmp";
    char* name = getenv("LOADL_STEP_ID");
    if (name) {
      snprintf(tmpname, sizeof(tmpname) - 1, "task.trace.tmp.%s", name);
    }
    MPI_Pcontrol(TRACEFILES, tmpname, "task.trace", 0);
  }
  MPI_Pcontrol(TRACESTATISTICS, 200, 1, 1, 1, 1, 1);
  // Start trace
  MPI_Pcontrol(TRACENODE, 1024 * 1024, 1, 1);
#endif

  MPI_Datatype mpi_border_type;
  {
    int blocklengths[] = { 1, 1, 1 };
    MPI_Datatype types[] = { MPI_UNSIGNED, MPI_DOUBLE, MPI_DOUBLE };
    MPI_Aint offsets[] = { offsetof(struct border, na), offsetof(struct border, a), offsetof(struct border, b) };

    mpi_check(MPI_Type_create_struct(sizeof(blocklengths) / sizeof(int), blocklengths, offsets, types, &mpi_border_type));
    mpi_check(MPI_Type_commit(&mpi_border_type));
  }

  int pid;
  mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
  int nprocs;
  mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

  int* displs = NULL;
  if (pid == 0) {
    displs = alloc_check(calloc(nprocs, sizeof(int)));
  }

  int* recvcounts = alloc_check(calloc(nprocs, sizeof(int)));
  mpi_check(UMPI_Scatterv_lens(mtx_size, recvcounts, displs, 0, MPI_COMM_WORLD));
  int recvlen = recvcounts[pid];
  if (pid != 0) {
    free(recvcounts);
  }

  struct border* borders = alloc_check(calloc(recvlen, sizeof(struct border)));
#endif
  
  struct tridiag* diag = alloc_check(newdiag(mtx_size));

#ifdef OMPI_PCONTROL

#endif

#ifdef OMPI
  if (pid == 0) {
#endif
    // Generate a random real symmetric matrix.
    // srand(time(NULL));
    srand(0);
    struct matrix* mtx = alloc_check(newmatrix(mtx_size, mtx_size));
    for (unsigned y = 0; y < mtx->h; y++) {
      for (unsigned x = 0; x <= y; x++) {
        val(mtx, x, y) = (double)rand() / (RAND_MAX / 2) - 1;
        val(mtx, y, x) = val(mtx, y, x);
      }
    }

    // Make a tridiagonal out of it.
    tridiagonalize(mtx);
    // mfprint(stdout, mtx);
    todiagonals(mtx, diag);
    freematrix(mtx);

#ifdef OMPI
    struct border* new_borders = alloc_check(calloc(diag->len, sizeof(struct border)));

    // Find eigenvalues.
    eigenvalues_borders(diag, range_from, range_to, new_borders);

    // Send tridiagonal matrix.
    mpi_check(MPI_Bcast(diag->d, diag->len, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    mpi_check(MPI_Bcast(diag->e, diag->len - 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));

    // At last, actually bisect.
    mpi_check(MPI_Scatterv(new_borders, recvcounts, displs, mpi_border_type, borders, recvlen, mpi_border_type, 0, MPI_COMM_WORLD));
    free(new_borders);
  } else {
    // Send tridiagonal matrix.
    mpi_check(MPI_Bcast(diag->d, diag->len, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    mpi_check(MPI_Bcast(diag->e, diag->len - 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));

    mpi_check(MPI_Scatterv(NULL, NULL, NULL, MPI_DATATYPE_NULL, borders, recvlen, mpi_border_type, 0, MPI_COMM_WORLD));
  }
#else
  // Find eigenvalues.
  eigenvalues_borders(diag, range_from, range_to, borders);
#endif

  double* values = NULL;

#ifdef OMPI
  if (pid == 0) {
#endif
    values = alloc_check(calloc(sizeof(double), mtx_size));
#ifdef OMPI
  }
#endif
  
#ifdef OMPI
  double* sendbuf = alloc_check(calloc(recvlen, sizeof(double)));
  for (unsigned i = 0; i < recvlen; i++) {
    printf("pid: %i, na: %i, a: %lf, b: %lf\n", pid, borders[i].na, borders[i].a, borders[i].b);
    sendbuf[i] = bisect(diag, borders[i].na, borders[i].a, borders[i].b, precision);
  }

  if (pid != 0) {
    mpi_check(MPI_Gatherv(sendbuf, recvlen, MPI_DOUBLE, NULL, NULL, NULL, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD));
  } else {
    mpi_check(MPI_Gatherv(sendbuf, recvlen, MPI_DOUBLE, values, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD));
  }
  free(sendbuf);
#else
  for (unsigned i = 0; i < mtx_size; i++) {
    values[i] = bisect(diag, borders[i].na, borders[i].a, borders[i].b, precision);
  }
#endif

  free(borders);
  freediag(diag);

#ifdef OMPI
  if (pid == 0) {
#endif
    if (mtx_size > 0) {
      printf(FLOAT_FMT, values[0]);
      for (unsigned i = 1; i < mtx_size; i++) {
        printf(" " FLOAT_FMT, values[i]);
      }
   
    printf("\n");
    }
    free(values);
#ifdef OMPI
  }
#endif

#ifdef OMPI
  if (pid == 0) {
    free(recvcounts);
    free(displs);
  }

  mpi_check(MPI_Finalize());
#endif

  return 0;
}
