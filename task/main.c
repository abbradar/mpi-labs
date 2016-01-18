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
#include "utils.h"
#endif

#ifdef __GNUC__
#define offsetof(type, member)  __builtin_offsetof (type, member)
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
  mtx->w = w;
  mtx->h = h;
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
  #pragma omp parallel
  if (mtx->w > 0 && mtx->h > 0) {
    struct matrix* u = newmatrix(1, mtx->h);
    struct matrix* p = newmatrix(1, mtx->h);

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
        }
      }
      for (unsigned y = cx; y < mtx->h - 1; y++) {
        for (unsigned x = y + 1; x < mtx->w; x++) {
          val(mtx, x, y) = val(mtx, y, x);
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
  diag->len = len;
  diag->d = calloc(len, sizeof(double));
  diag->e = calloc(len - 1, sizeof(double));
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

  struct tridiag* diag = newdiag(mtx_size);
#ifdef OMPI
  mpi_check(MPI_Init(&argc, &argv));
  // Set a sane error handler (return errors and don't kill our process)
  mpi_check(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

#ifdef OMPI_PCONTROL
  // Set trace parameters
  MPI_Pcontrol(TRACELEVEL, 1, 1, 1);
  MPI_Pcontrol(TRACEFILES, "task.trace.tmp", "task.trace", 0);
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
    displs = calloc(nprocs, sizeof(int));
  }
  int* recvcounts = calloc(nprocs, sizeof(int));

  mpi_check(UMPI_Scatterv_lens(mtx_size, recvcounts, displs, 0, MPI_COMM_WORLD));
  int recvlen = recvcounts[pid];

  struct border* recvbuf = calloc(recvlen, sizeof(struct border));

  if (pid == 0) {
#endif
    // Generate a random real symmetric matrix.
    // srand(time(NULL));
    srand(0);
    struct matrix* mtx = newmatrix(mtx_size, mtx_size);
    for (unsigned y = 0; y < mtx->h; y++) {
      for (unsigned x = 0; x <= y; x++) {
        val(mtx, x, y) = (double)rand() / (RAND_MAX / 2) - 1;
      }
    }
    for (unsigned y = 0; y < mtx->h - 1; y++) {
      for (unsigned x = y + 1; x < mtx->w; x++) {
        val(mtx, x, y) = val(mtx, y, x);
      }
    }

    // Make a tridiagonal out of it.
    tridiagonalize(mtx);
    // mfprint(stdout, mtx);
    todiagonals(mtx, diag);
    freematrix(mtx);

    // Find eigenvalues.
    struct border* borders = calloc(diag->len, sizeof(struct border));
    eigenvalues_borders(diag, range_from, range_to, borders);

#ifdef OMPI
    // Send tridiagonal matrix.
    mpi_check(MPI_Bcast(diag->d, diag->len, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    mpi_check(MPI_Bcast(diag->e, diag->len - 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));

    // At last, actually bisect.
    mpi_check(MPI_Scatterv(borders, recvcounts, displs, mpi_border_type, recvbuf, recvlen, mpi_border_type, 0, MPI_COMM_WORLD));
    free(borders);
  } else {
    // Send tridiagonal matrix.
    mpi_check(MPI_Bcast(diag->d, diag->len, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    mpi_check(MPI_Bcast(diag->e, diag->len - 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));

    mpi_check(MPI_Scatterv(NULL, NULL, NULL, MPI_DATATYPE_NULL, recvbuf, recvlen, mpi_border_type, 0, MPI_COMM_WORLD));
  }

  double* sendbuf = calloc(recvlen, sizeof(double));
  for (unsigned i = 0; i < recvlen; i++) {
    printf("pid: %i, na: %i, a: %lf, b: %lf\n", pid, recvbuf[i].na, recvbuf[i].a, recvbuf[i].b);
    sendbuf[i] = bisect(diag, recvbuf[i].na, recvbuf[i].a, recvbuf[i].b, precision);
  }

  if (pid != 0) {
    mpi_check(MPI_Gatherv(sendbuf, recvlen, MPI_DOUBLE, NULL, NULL, NULL, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD));
  } else {
#endif
    double* values = calloc(sizeof(double), mtx_size);
#ifdef OMPI
    mpi_check(MPI_Gatherv(sendbuf, recvlen, MPI_DOUBLE, values, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD));
#else
    for (unsigned i = 0; i < mtx_size; i++) {
      values[i] = bisect(diag, borders[i].na, borders[i].a, borders[i].b, precision);
    }
#endif

    if (mtx_size > 0) {
      printf(FLOAT_FMT, values[0]);
      for (unsigned i = 1; i < mtx_size; i++) {
        printf(" " FLOAT_FMT, values[i]);
      }
    }
    printf("\n");

    free(values);
#ifdef OMPI
  }
#endif

  freediag(diag);

#ifdef OMPI
  free(sendbuf);

  free(recvcounts);
  if (pid == 0) {
    free(displs);
  }

  mpi_check(MPI_Finalize());
#else
  free(borders);
#endif

  return 0;
}
