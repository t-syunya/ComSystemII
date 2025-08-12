#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

#define REAL double
#define BLOCKSIZE 32

// ブロッキング最適化のみ有効化
#define BLOCKING
//#define AVX2
//#define OMP
//#define AVX_OMP
//#define MKL
//#define UNROLL_ONLY
//#define UNROLL_OPTIMIZED
//#define BLOCKING_UNROLL
//#define OMP_UNROLL

#ifdef MKL
#include <mkl.h>
#endif

#ifdef AVX2
#include <immintrin.h>
#endif

/* unoptimized */
void dgemm_unopt(REAL *A, REAL *B, REAL *C, int n)
{
  int i, j, k;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      C[i + j * n] = 0.0;
      for (k = 0; k < n; k++)
        C[i + j * n] += A[i + k * n] * B[k + j * n];
    }
}

/* loop exchange */
void dgemm_jki(REAL *A, REAL *B, REAL *C, int n)
{
  int i, j, k;
  for (i = 0; i < n * n; i++)
    C[i] = 0.0;
  for (j = 0; j < n; j++)
    for (k = 0; k < n; k++)
      for (i = 0; i < n; i++)
        C[i + j * n] += A[i + k * n] * B[k + j * n];
}

/* blocking */
void do_block(int n, int si, int sj, int sk, REAL *A, REAL *B, REAL *C)
{
  int i, j, k;
  for (i = si; i < si + BLOCKSIZE; ++i)
    for (j = sj; j < sj + BLOCKSIZE; ++j)
    {
      for (k = sk; k < sk + BLOCKSIZE; ++k)
      {
        C[i + j * n] += A[i + k * n] * B[k + j * n];
      }
    }
}

void dgemm_blocking(REAL *A, REAL *B, REAL *C, int n)
{
  int sj, si, sk;
  for (sj = 0; sj < n; sj += BLOCKSIZE)
    for (si = 0; si < n; si += BLOCKSIZE)
      for (sk = 0; sk < n; sk += BLOCKSIZE)
        do_block(n, si, sj, sk, A, B, C);
}

/* AVX2 */
void dgemm_AVX2(REAL *A, REAL *B, REAL *C, int n)
{
  int i, j, k;

#if defined(FP_SINGLE)
  for (i = 0; i < n; i += 8)
  {
    for (j = 0; j < n; j++)
    {
      __m256 c0 = _mm256_loadu_ps(C + i + j * n);

      for (k = 0; k < n; k++)
        c0 =
            _mm256_add_ps(c0,
                          _mm256_mul_ps(_mm256_loadu_ps(A + i + k * n),
                                        _mm256_broadcast_ss(B + k +
                                                            j * n)));
      _mm256_storeu_ps(C + i + j * n, c0);
    }
  }
#else
  for (i = 0; i < n; i += 4)
  {
    for (j = 0; j < n; j++)
    {
      __m256d c0 = _mm256_loadu_pd(C + i + j * n);

      for (k = 0; k < n; k++)
        c0 =
            _mm256_add_pd(c0,
                          _mm256_mul_pd(_mm256_loadu_pd(A + i + k * n),
                                        _mm256_broadcast_sd(B + k +
                                                            j * n)));
      _mm256_storeu_pd(C + i + j * n, c0);
    }
  }
#endif
}

/* OpenMP */
void dgemm_OMP(REAL *A, REAL *B, REAL *C, int n)
{
  int i, j, k;
  for (i = 0; i < n * n; i++)
    C[i] = 0.0;
#pragma omp parallel for private(j, k)
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++)
        C[i + j * n] += A[i + k * n] * B[k + j * n];
}

/* AVX + OpenMP */
void dgemm_AVX_OMP(REAL *A, REAL *B, REAL *C, int n)
{
  int i, j, k;
#pragma omp parallel for private(j, k)
#if defined(FP_SINGLE)
  for (i = 0; i < n; i += 8)
  {
    for (j = 0; j < n; j++)
    {
      __m256 c0 = _mm256_loadu_ps(C + i + j * n);

      for (k = 0; k < n; k++)
        c0 =
            _mm256_add_ps(c0,
                          _mm256_mul_ps(_mm256_loadu_ps(A + i + k * n),
                                        _mm256_broadcast_ss(B + k +
                                                            j * n)));
      _mm256_storeu_ps(C + i + j * n, c0);
    }
  }
#else
  for (i = 0; i < n; i += 4)
  {
    for (j = 0; j < n; j++)
    {
      __m256d c0 = _mm256_loadu_pd(C + i + j * n);

      for (k = 0; k < n; k++)
        c0 =
            _mm256_add_pd(c0,
                          _mm256_mul_pd(_mm256_loadu_pd(A + i + k * n),
                                        _mm256_broadcast_sd(B + k +
                                                            j * n)));
      _mm256_storeu_pd(C + i + j * n, c0);
    }
  }
#endif
}

/* Loop Unrolling - 正しい実装（列優先アクセス） */
void dgemm_unroll(REAL *A, REAL *B, REAL *C, int n)
{
  int i, j, k;
  int unroll_factor = 4;

  // 初期化
  for (i = 0; i < n * n; i++)
    C[i] = 0.0;

  // 列優先アクセスパターン（正しい実装）
  for (j = 0; j < n; j++)
    for (k = 0; k < n; k++)
    {
      REAL bkj = B[k + j * n];
      // アンロール可能な部分
      for (i = 0; i <= n - unroll_factor; i += unroll_factor)
      {
        C[i + j * n] += A[i + k * n] * bkj;
        C[(i + 1) + j * n] += A[(i + 1) + k * n] * bkj;
        C[(i + 2) + j * n] += A[(i + 2) + k * n] * bkj;
        C[(i + 3) + j * n] += A[(i + 3) + k * n] * bkj;
      }
      // 残りの要素を処理
      for (; i < n; i++)
        C[i + j * n] += A[i + k * n] * bkj;
    }
}

/* より効率的なループアンローリング（最適化版） */
void dgemm_unroll_optimized(REAL *A, REAL *B, REAL *C, int n)
{
  int i, j, k;
  int unroll_factor = 8;

  // 初期化
  for (i = 0; i < n * n; i++)
    C[i] = 0.0;

  // 列優先アクセスパターン（最適化版）
  for (j = 0; j < n; j++)
    for (k = 0; k < n; k++)
    {
      REAL bkj = B[k + j * n];
      // より大きなアンロール係数（8）
      for (i = 0; i <= n - unroll_factor; i += unroll_factor)
      {
        C[i + j * n] += A[i + k * n] * bkj;
        C[(i + 1) + j * n] += A[(i + 1) + k * n] * bkj;
        C[(i + 2) + j * n] += A[(i + 2) + k * n] * bkj;
        C[(i + 3) + j * n] += A[(i + 3) + k * n] * bkj;
        C[(i + 4) + j * n] += A[(i + 4) + k * n] * bkj;
        C[(i + 5) + j * n] += A[(i + 5) + k * n] * bkj;
        C[(i + 6) + j * n] += A[(i + 6) + k * n] * bkj;
        C[(i + 7) + j * n] += A[(i + 7) + k * n] * bkj;
      }
      // 残りの要素を処理
      for (; i < n; i++)
        C[i + j * n] += A[i + k * n] * bkj;
    }
}

/* Blocking + Loop Unrolling */
void do_block_unroll(int n, int si, int sj, int sk, REAL *A, REAL *B, REAL *C)
{
  int i, j, k;
  int unroll_factor = 4;
  for (i = si; i < si + BLOCKSIZE; ++i)
  {
    for (j = sj; j < sj + BLOCKSIZE; ++j)
    {
      for (k = sk; k < sk + BLOCKSIZE - unroll_factor + 1; k += unroll_factor)
      {
        C[i + j * n] += A[i + k * n] * B[k + j * n];
        C[i + j * n] += A[i + (k+1) * n] * B[(k+1) + j * n];
        C[i + j * n] += A[i + (k+2) * n] * B[(k+2) + j * n];
        C[i + j * n] += A[i + (k+3) * n] * B[(k+3) + j * n];
      }
      for (; k < sk + BLOCKSIZE; k++) C[i + j * n] += A[i + k * n] * B[k + j * n];
    }
  }
}

void dgemm_blocking_unroll(REAL *A, REAL *B, REAL *C, int n)
{
  int sj, si, sk;
  for (sj = 0; sj < n; sj += BLOCKSIZE)
    for (si = 0; si < n; si += BLOCKSIZE)
      for (sk = 0; sk < n; sk += BLOCKSIZE)
        do_block_unroll(n, si, sj, sk, A, B, C);
}

/* OpenMP + Loop Unrolling */
void dgemm_OMP_unroll(REAL *A, REAL *B, REAL *C, int n)
{
  int i, j, k;
  int unroll_factor = 4;
  for (i = 0; i < n * n; i++) C[i] = 0.0;
#pragma omp parallel for private(j,k)
  for (i = 0; i < n; i++)
  {
    for (j = 0; j < n; j++)
    {
      for (k = 0; k <= n - unroll_factor; k += unroll_factor)
      {
        C[i + j * n] += A[i + k * n] * B[k + j * n];
        C[i + j * n] += A[i + (k+1) * n] * B[(k+1) + j * n];
        C[i + j * n] += A[i + (k+2) * n] * B[(k+2) + j * n];
        C[i + j * n] += A[i + (k+3) * n] * B[(k+3) + j * n];
      }
      for (; k < n; k++) C[i + j * n] += A[i + k * n] * B[k + j * n];
    }
  }
}

/* Timer */
double seconds()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + ((double)tv.tv_usec) / 1000000.0;
}

/* init matrics */
void int_mat(REAL *A, REAL *B, REAL *C, int N)
{
  int i, j;
  srand(1);
  for (i = 0; i < N; ++i)
    for (j = 0; j < N; ++j)
    {
      A[i + j * N] = (REAL)rand() / (10000 + i + j);
      B[i + j * N] = (REAL)rand() / (10000 + i + j);
      C[i + j * N] = (REAL)0.0;
    }
}

/* Check calculation*/
int check_mat(REAL *C, REAL *C_unopt, int N)
{
  int n, m;
  double max_err = 1.0e-5;
  for (n = 0; n < N; n++)
  {
    for (m = 0; m < N; m++)
    {
      if (fabs((C[n + N * m] - C_unopt[n + N * m]) / C_unopt[n + N * m]) > max_err)
      {
        printf("Error:   result is different in %d,%d  (%.5f, %.5f) delta %.5f > max_err %.5f \n",
               n, m, C[n + N * m], C_unopt[n + N * m],
               fabs(C[n + N * m] - C_unopt[n + N * m]), max_err);
      }
    }
  }
}

/*** Main(Matrix calculation) ***/
int main(int argc, char *argv[])
{
#ifdef MKL
#include <mkl.h>
#endif

  REAL *A, *B, *C, *C_unopt;
  int N;   /* N=matrix size */
  int itr; /* Number of iterations */
  int i;
  double t;

  if (argc < 3)
  {
    fprintf(stderr, "Specify M, #ITER\n");
    exit(1);
  }

  N = atoi(argv[1]);
  itr = atoi(argv[2]);

#if defined(FP_SINGLE)
  printf("data_size : float\n");
#else
  printf("data_size : double(default)\n");
#endif
  printf("array size N = %d\n", N);
  printf("blocking size = %d\n", BLOCKSIZE);
  printf("The number of threads= %s\n", getenv("OMP_NUM_THREADS"));
  printf("iterations = %d\n", itr);

  /** memory set **/
  A = (REAL *)malloc(N * N * sizeof(REAL));
  B = (REAL *)malloc(N * N * sizeof(REAL));
  C = (REAL *)malloc(N * N * sizeof(REAL));
  C_unopt = (REAL *)malloc(N * N * sizeof(REAL));

  /** calculation **/
  for (i = 0; i < itr; ++i)
  {
    /*unoptimized */
    int_mat(A, B, C_unopt, N);
    t = seconds();
    dgemm_unopt(A, B, C_unopt, N);
    t = seconds() - t;
    printf("\n%f [s]  GFLOPS %f  |unoptimized| \n", t,
           (float)N * N * N * 2 / t / 1000 / 1000 / 1000);

    /*unoptimized */
    int_mat(A, B, C_unopt, N);
    t = seconds();
    dgemm_jki(A, B, C_unopt, N);
    t = seconds() - t;
    printf("%f [s]  GFLOPS %f  |loop exchange| \n", t,
           (float)N * N * N * 2 / t / 1000 / 1000 / 1000);

    /*blocking */
#ifdef BLOCKING
    int_mat(A, B, C, N);
    t = seconds();
    dgemm_blocking(A, B, C, N);
    t = seconds() - t;
    check_mat(C, C_unopt, N);
    printf("%f [s]  GFLOPS %f  |blocking|\n", t,
           (float)N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif
  }

  free(A);
  free(B);
  free(C);
  free(C_unopt);
  return 0;
}
