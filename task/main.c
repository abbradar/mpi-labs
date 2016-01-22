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
  unsigned nb;
  double b;
};

void findborders(const struct tridiag* diag, double a, unsigned na, double b, unsigned nb, unsigned nmax, MPI_Datatype type)
{
  assert(b >= a);
  assert(nb >= na);
  if (nb - na != 0 && nb - na <= nmax) {
    struct border border;
    border.na = na;
    border.a = a;
    border.nb = nb;
    border.b = b;

    MPI_Status stat;
    mpi_check(MPI_Recv(NULL, 0, type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stat));
    mpi_check(MPI_Send(&border, 1, type, stat.MPI_SOURCE, 0, MPI_COMM_WORLD));
  } else if (nb - na > nmax) {
    double c = (a + b) / 2;
    unsigned nc = sturmsigns(diag, c);
    findborders(diag, a, na, c, nc, nmax, type);
    findborders(diag, c, nc, b, nb, nmax, type);
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

void multibisect(const struct tridiag* diag, struct border border, double acc, double* values, unsigned* nvalues)
{
  if (border.nb - border.na == 1) {
    values[*nvalues] = bisect(diag, border.na, border.a, border.b, acc);
    ++*nvalues;
  } else if (border.nb - border.na != 0) {
    double c = (border.a + border.b) / 2;
    unsigned nc = sturmsigns(diag, c);
    struct border b1 = { border.na, border.a, nc, c };
    multibisect(diag, b1, acc, values, nvalues);
    struct border b2 = { nc, c, border.nb, border.b };
    multibisect(diag, b2, acc, values, nvalues);
  }
}

void eigenvalues_borders(const struct tridiag* diag, double a, double b, unsigned nmax, MPI_Datatype type)
{
    // First, find real eigenvalues' borders.
    while (sturmsigns(diag, a) > 0) a -= fabs(a) + 0.1;
    while (sturmsigns(diag, b) != diag->len) b += fabs(b) + 0.1;
    findborders(diag, a, 0, b, diag->len, nmax, type);
}

int main(int argc, char** argv)
{
  const int mtx_size = 1600;
  const double range_from = -1;
  const double range_to = 1;
  const double precision = 0.0001;
  const double chunksize = 200;

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
    int blocklengths[] = { 1, 1, 1, 1 };
    MPI_Datatype types[] = { MPI_UNSIGNED, MPI_DOUBLE, MPI_UNSIGNED, MPI_DOUBLE };
    MPI_Aint offsets[] = { offsetof(struct border, na), offsetof(struct border, a), offsetof(struct border, nb), offsetof(struct border, b) };

    mpi_check(MPI_Type_create_struct(sizeof(blocklengths) / sizeof(int), blocklengths, offsets, types, &mpi_border_type));
    mpi_check(MPI_Type_commit(&mpi_border_type));
  }

  int pid;
  mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
  int nprocs;
  mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
  
  struct tridiag* diag = alloc_check(newdiag(mtx_size));

#ifdef OMPI_PCONTROL

#endif

  double* values = alloc_check(calloc(mtx_size, sizeof(double)));
  
  if (pid == 0) {
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

    // Send tridiagonal matrix.
    mpi_check(MPI_Bcast(diag->d, diag->len, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    mpi_check(MPI_Bcast(diag->e, diag->len - 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));
  } else {
    // Receive tridiagonal matrix.
    mpi_check(MPI_Bcast(diag->d, diag->len, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    mpi_check(MPI_Bcast(diag->e, diag->len - 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));
  }

  if (pid == 0) {
    eigenvalues_borders(diag, range_from, range_to, chunksize, mpi_border_type);

    int n;
    for (unsigned i = 0; i < mtx_size; i += n) {
      MPI_Status stat;
      mpi_check(MPI_Recv(NULL, 0, mpi_border_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stat));
      mpi_check(MPI_Send(NULL, 0, mpi_border_type, stat.MPI_SOURCE, 0, MPI_COMM_WORLD));
      mpi_check(MPI_Probe(stat.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stat));
      mpi_check(MPI_Get_count(&stat, MPI_DOUBLE, &n));
      mpi_check(MPI_Recv(values + i, n, MPI_DOUBLE, stat.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, NULL));
    }
  } else {
    unsigned i = 0;
    while (1) {
      MPI_Status stat;
      int n;
      mpi_check(MPI_Send(NULL, 0, mpi_border_type, 0, 0, MPI_COMM_WORLD));
      mpi_check(MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat));
      mpi_check(MPI_Get_count(&stat, mpi_border_type, &n));
      if (n == 0) {
        mpi_check(MPI_Recv(NULL, 0, mpi_border_type, 0, 0, MPI_COMM_WORLD, NULL));
        mpi_check(MPI_Send(values, i, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
        break;
      } else {
        struct border border;
        assert(n == 1);
        mpi_check(MPI_Recv(&border, 1, mpi_border_type, 0, 0, MPI_COMM_WORLD, NULL));
        multibisect(diag, border, precision, values, &i);
      }
    }
  }

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
#ifdef OMPI
  }
#endif

  free(values);
  freediag(diag);

#ifdef OMPI
  mpi_check(MPI_Finalize());
#endif

  return 0;
}
