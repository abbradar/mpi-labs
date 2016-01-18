#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <mpi.h>

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
  size_t w;
  size_t h;
  double data[];
};

struct matrix* newmatrix(size_t w, size_t h)
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
  for (size_t y = 0; y < mtx->h; ++y) {
    if (mtx->w > 0) {
      fprintf(file, FLOAT_FMT, val(mtx, 0, y));
      for (size_t x = 1; x < mtx->w; ++x) {
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
  for (size_t y = 0; y < out->h; y++) {
    for (size_t x = 0; x < out->w; x++) {
      double r = 0;
      for (size_t i = 0; i < a->w; i++) {
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
  if (mtx->w > 0 && mtx->h > 0) {
    struct matrix* u = newmatrix(1, mtx->h);
    struct matrix* p = newmatrix(1, mtx->h);

    for (size_t i = 1; i < mtx->w - 1; i++) {
      size_t cx = i - 1;

      double mx = 0;
      for (size_t y = i; y < mtx->h; y++) {
        mx += sqr(val(mtx, cx, y));
      }
      mx = sqrt(mx);

      for (size_t y = 0; y < i; y++) {
        val(u, 0, y) = 0;
      }
      for (size_t y = i; y < mtx->h; y++) {
        val(u, 0, y) = val(mtx, cx, y);
      }
      val(u, 0, i) -= mx;

      double mu2 = 0;
      for (size_t y = 0; y < mtx->h; y++) {
        mu2 += sqr(val(u, 0, y));
      }

      mmultiply(mtx, u, p);
      for (size_t y = 0; y < mtx->h; y++) {
        val(p, 0, y) /= mu2 / 2;
      }

      double k = 0;
      for (size_t j = 0; j < mtx->h; j++) {
        k += val(u, 0, j) * val(p, 0, j) / mu2;
      }

      for (size_t y = 0; y < mtx->h; y++) {
        val(p, 0, y) -= k * val(u, 0, y);
      }

      for (size_t y = cx; y < mtx->h; y++) {
        for (size_t x = cx; x <= y; x++) {
          val(mtx, x, y) -= val(p, 0, y) * val(u, 0, x) + val(u, 0, y) * val(p, 0, x);
        }
      }
      for (size_t y = cx; y < mtx->h - 1; y++) {
        for (size_t x = y + 1; x < mtx->w; x++) {
          val(mtx, x, y) = val(mtx, y, x);
        }
      }
    }
    freematrix(u);
    freematrix(p);
  }
}

struct tridiag {
  size_t len;
  double* d;
  double* e;
};

struct tridiag* newdiag(size_t len)
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
  for (size_t i = 0; i < mtx->h; i++) {
    diag->d[i] = val(mtx, i, i);
  }
  for (size_t i = 0; i < mtx->h - 1; i++) {
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
size_t sturmsigns(const struct tridiag* diag, double c)
{
  size_t res = 0;
  double prev_q = 1;
  if (diag->len > 0) {
    nextsturm(diag->d[0] - c);
    for (size_t i = 1; i < diag->len; i++) {
      nextsturm((diag->d[i] - c) - sqr(diag->e[i - 1]) / prev_q);
    }
  }
  return res;
}

#undef nextsturm

struct border {
  size_t na;
  double a;
  double b;
};

void findborders(const struct tridiag* diag, double a, size_t na, double b, size_t nb, struct border* borders, size_t* borders_num)
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
    size_t nc = sturmsigns(diag, c);
    findborders(diag, a, na, c, nc, borders, borders_num);
    findborders(diag, c, nc, b, nb, borders, borders_num);
  }
}

struct poly {
  size_t max_power;
  size_t power;
  double ks[];
};

struct poly* newpoly(size_t max_power)
{
  assert(max_power > 0);
  struct poly* poly = calloc(1, sizeof(struct poly) + max_power * sizeof(double));
  poly->max_power = max_power;
  return poly;
}

void freepoly(struct poly* poly)
{
  free(poly);
}

void clearpoly(struct poly* poly)
{
  memset(poly->ks, 0, (poly->power + 1) * sizeof(double));
  poly->power = 0;
}

void copypoly(const struct poly* from, struct poly* to)
{
  assert(from->power < to->max_power);
  if (to->power > from->power) {
    memset(to->ks + from->power + 1, 0, (from->power - to->power) * sizeof(double));
  }
  to->power = from->power;
  memcpy(to->ks, from->ks, (to->power + 1) * sizeof(double));
}

void addtopoly(struct poly* poly, size_t power, double val)
{
  assert(power < poly->max_power);
  poly->power = max(poly->power, power);
  poly->ks[power] = val;
}

void multopoly(struct poly* poly, double val)
{
  for (size_t i = 0; i <= poly->power; i++) {
    poly->ks[i] *= val;
  }
}

#define addsubpolys(name, op) \
  void name(struct poly* acc, const struct poly* from) \
  {                                                    \
    size_t minp = min(from->power, acc->power);        \
    size_t maxp = max(from->power, acc->power);        \
    assert(maxp < acc->max_power);                     \
    for (size_t i = 0; i < minp; i++) {                \
      acc->ks[i] = acc->ks[i] op from->ks[i];          \
    }                                                  \
    if (from->power > minp) {                          \
      for (size_t i = minp; i < from->power; i++) {    \
        acc->ks[i] = op from->ks[i];                   \
      }                                                \
      acc->power = from->power;                        \
    }                                                  \
  }

addsubpolys(addpolys, +)
addsubpolys(subpolys, -)

#undef addsubpolys

void mulpolys(const struct poly* a, const struct poly* b, struct poly* res)
{
  size_t new_power = a->power + b->power;
  assert(new_power < res->max_power);
  clearpoly(res);
  for (size_t i = 0; i < a->power + 1; i++) {
    for (size_t j = 0; j < b->power + 1; j++) {
      res->ks[i + j] = a->ks[i] * b->ks[j];
    }
  }
  res->power = new_power;
}

double polyval(const struct poly* poly, double val)
{
  double res = 0;
  for (size_t i = 0; i < poly->power + 1; i++) {
    res += poly->ks[i] * pow(val, i);
  }
  return res;
}

#define diagpoly(to, n) {            \
    addtopoly((to), 0, -diag->d[n]); \
    addtopoly((to), 1, 1);           \
  }

void diagcharpoly(const struct tridiag* diag, struct poly* poly)
{
  assert(diag->len > 0);
  assert(poly->max_power >= diag->len + 1);

  struct poly* prev_p = newpoly(diag->len + 1);
  diagpoly(prev_p, 0);

  if (diag->len == 1) {
    copypoly(prev_p, poly);
  } else {
    struct poly* tmp_p = newpoly(diag->len + 1);
    diagpoly(tmp_p, 1);
    mulpolys(prev_p, tmp_p, poly);
    addtopoly(poly, 0, sqr(diag->e[0]));

    if (diag->len > 2) {
      struct poly* tmp2_p = newpoly(diag->len + 1);

      for (size_t i = 2; i < diag->len; i++) {
        multopoly(prev_p, sqr(diag->e[i - 1]));
        clearpoly(tmp_p);
        diagpoly(tmp_p, i);
        mulpolys(poly, tmp_p, tmp2_p);
        subpolys(tmp2_p, prev_p);
        copypoly(poly, prev_p);
        copypoly(tmp2_p, poly);
      }

      freepoly(tmp2_p);
    }

    freepoly(tmp_p);
  }

  freepoly(prev_p);
}

#undef diagpoly

double bisect(const struct tridiag* diag, size_t na, double a, double b, double acc)
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

void eigenvalues(const struct tridiag* diag, double a, double b, double acc, double* values)
{
  assert(b > a);
  assert(acc > 0);
  // First, find real eigenvalues' borders.
  while (sturmsigns(diag, a) > 0) a -= fabs(a) + 0.1;
  while (sturmsigns(diag, b) != diag->len) b += fabs(b) + 0.1;
  struct border* borders = calloc(diag->len, sizeof(struct border));
  size_t borders_num = 0;
  findborders(diag, a, 0, b, diag->len, borders, &borders_num);
  assert(borders_num == diag->len);

  // At last, actually bisect.
  for (size_t i = 0; i < diag->len; i++) {
    values[i] = bisect(diag, borders[i].na, borders[i].a, borders[i].b, acc);
  }

  free(borders);
}

int main()
{
  const int mtx_size = 20;

  // Generate a random real symmetric matrix.
  // srand(time(NULL));
  srand(0);
  struct matrix* mtx = newmatrix(mtx_size, mtx_size);
  for (size_t y = 0; y < mtx->h; y++) {
    for (size_t x = 0; x <= y; x++) {
      val(mtx, x, y) = (double)rand() / (RAND_MAX / 2) - 1;
    }
  }
  for (size_t y = 0; y < mtx->h - 1; y++) {
    for (size_t x = y + 1; x < mtx->w; x++) {
      val(mtx, x, y) = val(mtx, y, x);
    }
  }

  // Make a tridiagonal out of it.
  tridiagonalize(mtx);
  mfprint(stdout, mtx);
  struct tridiag* diag = newdiag(mtx->w);
  todiagonals(mtx, diag);
  freematrix(mtx);

  // Find eigenvalues.
  double* values = calloc(sizeof(double), mtx_size);
  eigenvalues(diag, -1, 1, 0.001, values);
  freediag(diag);

  if (mtx_size > 0) {
    printf(FLOAT_FMT, values[0]);
    for (size_t i = 1; i < mtx_size; i++) {
      printf(" " FLOAT_FMT, values[i]);
    }
  }
  printf("\n");

  free(values);

  return 0;
}
