//
//   This program is written in column-major fashion to keep compatibility for Fortran programs.
//


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <x86intrin.h>


#define BLOCKSIZE 32
#define UNROLL 4

//#define FP_SINGLE     /* Data Size: float -> ON */

#if defined(FP_SINGLE)
#define REAL float		/* Data Size: float -> ON */
#else
#define REAL double		/* Data Size: double(default) -> ON */
#endif

#define BLOCKING		/* blocking -> ON */
//#define AVX2			/* Intel AVX2-> ON */
//#define AVX512			/* Intel AVX512 -> ON */
//#define AVX512_UNROLL		/* Intel AVX512 & UNROLL -> ON */
#define OMP                           /* OpenMP -> ON */
#define AVX_OMP                       /* Intel AVX + OpenMP -> ON */
//#define AVX512_BLOCKING		/* Intel AVX512 & blocking -> ON */
//#define AVX512_UNROLL_BLOCKING	/* Intel AVX512 & UNROLL & blocking -> ON */
#define AVX512_UNROLL_BLOCKING_OMP	/* Intel AVX512 & UNROLL & blocking & OMP -> ON */
//#define BLOCKING_OMP          /* blocking + OpenMP -> ON */
//#define BLOCKING_OMP2         /* blocking + OpenMP(2places) -> ON */
//#define BLOCKING_AVX_OMP              /* blocking + Intel AVX + OpenMP -> ON */
//#define BLOCKING_AVX_OMP2             /* blocking + Intel AVX + OpenMP(2places) -> ON */

#define MKL


/* Unoptimized */
void
dgemm_unopt (REAL * A, REAL * B, REAL * C, int n)
{
  int i, j, k;
  REAL cij;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	cij = C[i + j * n];	/* cij = C[i][j] */
	for (k = 0; k < n; k++)
	  cij += A[i + k * n] * B[k + j * n];	/* cij+=A[i][k]*B[k][j] */
	C[i + j * n] = cij;	/* C[i][j] = cij */
      }
}

/* LoopExchange */
void
dgemm_jki(REAL * A, REAL * B, REAL * C, int n)
{
  int i, j, k;
  REAL cij;

  for (j = 0; j < n; j++)
    for (k = 0; k < n; k++){
      for (i = 0; i < n; i++){
	C[i + j * n] += A[i + k * n] * B[k + j * n];	/* cij+=A[i][k]*B[k][j] */
      }
    }
}


/* Blocking &  Blocking+AVX */
void
do_block (int n, int si, int sj, int sk, REAL * A, REAL * B, REAL * C)
{
  int i, j, k;
  REAL cij;

  for (i = si; i < si + BLOCKSIZE; ++i)
    {
      for (j = sj; j < sj + BLOCKSIZE; ++j)
	{
	  cij = C[i + j * n];	/* cij = C[i][j] */
	  for (k = sk; k < sk + BLOCKSIZE; k++)
	    cij += A[i + k * n] * B[k + j * n];	/* cij+=A[i][k]*B[k][j] */
	  C[i + j * n] = cij;	/* C[i][j] = cij */
	}
    }
}


void
dgemm_blocking (REAL * A, REAL * B, REAL * C, int n)
{
  int sj, si, sk;

  for (sj = 0; sj < n; sj += BLOCKSIZE)
    for (si = 0; si < n; si += BLOCKSIZE)
      for (sk = 0; sk < n; sk += BLOCKSIZE)
	do_block (n, si, sj, sk, A, B, C);
}


/* OpenMP */
void
dgemm_OMP (REAL * A, REAL * B, REAL * C, int n)
{
  int i, j, k;
  REAL cij;

#pragma omp parallel for 
  for (i = 0; i < n; i++)
    {
      for (j = 0; j < n; j++)
	{
	  cij = C[i + j * n];	/* cij = C[i][j] */
	  for (k = 0; k < n; k++)
	    cij += A[i + k * n] * B[k + j * n];	/* cij+=A[i][k]*B[k][j] */
	  C[i + j * n] = cij;	/* C[i][j] = cij */
	}
    }
}


/* Timer */
double
seconds ()
{
  struct timeval tv;
  gettimeofday (&tv, NULL);
  return (double) tv.tv_sec + ((double) tv.tv_usec) / 1000000.0;
}


/* init matrics */
void
int_mat (REAL * A, REAL * B, REAL * C, int N)
{
  int i, j;

  srand (1);

  for (i = 0; i < N; ++i)
    for (j = 0; j < N; ++j)
      {
	A[i + j * N] = (REAL) rand () / (10000 + i + j);
	B[i + j * N] = (REAL) rand () / (10000 + i + j);
	C[i + j * N] = (REAL) 0.0;
      }
}


/* Check calculation*/
int
check_mat (REAL * C, REAL * C_unopt, int N)
{
  int n, m;
  double max_err=1.0e-5;

  for (n = 0; n < N; n++)
    {
      for (m = 0; m < N; m++)
	{
	  if (fabs ((C[n + N * m] - C_unopt[n + N * m])/C_unopt[n + N * m]) > max_err)
	    {
	      printf("Error:   result is different in %d,%d  (%.5f, %.5f) delta %.5f > max_err %.5f \n",
		 n, m, C[n + N * m], C_unopt[n + N * m],
		 fabs (C[n + N * m] - C_unopt[n + N * m]), max_err);
	    }
	}
    }
}


/*** Main(Matrix calculation) ***/
int
main (int argc, char *argv[])
{
#ifdef MKL
	#include <mkl.h>
#endif

  REAL *A, *B, *C, *C_unopt;
  int N;			/* N=matrix size */
  int itr;			/* Number of iterations */
  int i;
  double t;

  if (argc < 3)
    {
      fprintf (stderr, "Specify M, #ITER\n");
      /* Argument 1:Array size, Argument 2:Number of iterations */
      exit (1);
    }

  N = atoi (argv[1]);		/* Argument 1:Array size */
  if (N % 8 != 0)
    {
      printf ("Please specify N that is a multipe of 8 for AVX 256 bit\n");
      return 0;
    }

  if (N % BLOCKSIZE != 0)
    {
      printf
	("Please specify N that is a multipe of BLOCKSIZE(%d) for Blocking\n",
	 BLOCKSIZE);
      return 0;
    }


  itr = atoi (argv[2]);		/* Argument 2:Number of iterations */


#if defined(FP_SINGLE)
  printf ("data_size : float\n");
#else
  printf ("data_size : double(default)\n");
#endif
  printf ("array size N = %d\n", N);

  printf ("blocking size = %d\n", BLOCKSIZE);
  printf ("The number of threads= %s\n", getenv ("OMP_NUM_THREADS"));

  printf ("iterations = %d\n", itr);

	/** memory set **/
  A = (REAL *) malloc (N * N * sizeof (REAL));
  B = (REAL *) malloc (N * N * sizeof (REAL));
  C = (REAL *) malloc (N * N * sizeof (REAL));
  C_unopt = (REAL *) malloc (N * N * sizeof (REAL));

	/** calculation **/
  for (i = 0; i < itr; ++i)
    {
      /*unoptimized */
      int_mat (A, B, C_unopt, N);
      t = seconds ();
      dgemm_unopt (A, B, C_unopt, N);
      t = seconds () - t;
      printf ("\n%f [s]  GFLOPS %f  |unoptimized| \n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);

      /*unoptimized */
      int_mat (A, B, C_unopt, N);
      t = seconds ();
      dgemm_jki (A, B, C_unopt, N);
      t = seconds () - t;
      printf ("%f [s]  GFLOPS %f  |loop exchange| \n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);


      /*blocking */
#ifdef BLOCKING
      int_mat (A, B, C, N);
      t = seconds ();
      dgemm_blocking (A, B, C, N);
      t = seconds () - t;
      check_mat (C, C_unopt, N);
      printf ("%f [s]  GFLOPS %f  |blocking|\n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

      /*AVX2 */
#ifdef AVX2
      int_mat (A, B, C, N);
      t = seconds ();
      dgemm_AVX2 (A, B, C, N);
      t = seconds () - t;
      check_mat (C, C_unopt, N);
      printf ("%f [s]  GFLOPS %f  |AVX2|\n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

      /*AVX512 */
#ifdef AVX512
      int_mat (A, B, C, N);
      t = seconds ();
      dgemm_AVX512 (A, B, C, N);
      t = seconds () - t;
      check_mat (C, C_unopt, N);
      printf ("%f [s]  GFLOPS %f  |AVX512|\n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

      /*AVX512_UNROLL */
#ifdef AVX512_UNROLL
      int_mat (A, B, C, N);
      t = seconds ();
      dgemm_AVX512_UNROLL (A, B, C, N);
      t = seconds () - t;
      check_mat (C, C_unopt, N);
      printf ("%f [s]  GFLOPS %f  |AVX512_UNROLL|\n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

      /*OpenMP */
#ifdef OMP
      int_mat (A, B, C, N);
      t = seconds ();
      dgemm_OMP (A, B, C, N);
      t = seconds () - t;
      check_mat (C, C_unopt, N);
      printf ("%f [s]  GFLOPS %f  |OpenMP|\n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

      /*AVX+OpenMP */
#ifdef AVX_OMP
      int_mat (A, B, C, N);
      t = seconds ();
      dgemm_AVX_OMP (A, B, C, N);
      t = seconds () - t;
      check_mat (C, C_unopt, N);
      printf ("%f [s]  GFLOPS %f  |AVX+OpenMP|\n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

      /*AVX512+blocking */
#ifdef AVX512_BLOCKING
      int_mat (A, B, C, N);
      t = seconds ();
      dgemm_AVX512_blocking (A, B, C, N);
      t = seconds () - t;
      check_mat (C, C_unopt, N);
      printf ("%f [s]  GFLOPS %f  |AVX512+blocking|\n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

      /*AVX512+UNROLL+blocking */
#ifdef AVX512_UNROLL_BLOCKING
      int_mat (A, B, C, N);
      t = seconds ();
      dgemm_AVX512_UNROLL_blocking (A, B, C, N);
      t = seconds () - t;
      check_mat (C, C_unopt, N);
      printf ("%f [s]  GFLOPS %f  |AVX512+UNROLL+blocking|\n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

      /*AVX512+UNROLL+blocking+OMP */
#ifdef AVX512_UNROLL_BLOCKING_OMP
      int_mat (A, B, C, N);
      t = seconds ();
      dgemm_AVX512_UNROLL_blocking_OMP (A, B, C, N);
      t = seconds () - t;
      check_mat (C, C_unopt, N);
      printf ("%f [s]  GFLOPS %f  |AVX512+UNROLL+blocking+OMP|\n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif


      /*blocking+OMP */
#ifdef BLOCKING_OMP
      int_mat (A, B, C, N);
      t = seconds ();
      dgemm_blocking_OMP (A, B, C, N);
      t = seconds () - t;
      check_mat (C, C_unopt, N);
      printf ("%f [s]  GFLOPS %f  |blocking+OMP|\n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

      /*blocking+OMP2 */
#ifdef BLOCKING_OMP2
      int_mat (A, B, C, N);
      t = seconds ();
      dgemm_blocking_OMP2 (A, B, C, N);
      t = seconds () - t;
      check_mat (C, C_unopt, N);
      printf ("%f [s]  GFLOPS %f  |blocking+OMP2|\n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

      /*blocking+AVX+OMP */
#ifdef BLOCKING_AVX_OMP
      int_mat (A, B, C, N);
      t = seconds ();
      dgemm_blocking_AVX_OMP (A, B, C, N);
      t = seconds () - t;
      check_mat (C, C_unopt, N);
      printf ("%f [s]  GFLOPS %f  |blocking+AVX+OMP|\n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

      /*blocking+AVX+OMP2 */
#ifdef BLOCKING_AVX_OMP2
      int_mat (A, B, C, N);
      t = seconds ();
      dgemm_blocking_AVX_OMP2 (A, B, C, N);
      t = seconds () - t;
      check_mat (C, C_unopt, N);
      printf ("%f [s]  GFLOPS %f  |blocking+AVX+OMP2|\n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

#ifdef MKL
      int_mat (A, B, C, N);
      t = seconds();

#if defined(FP_SINGLE)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 1.0, C, N);
#else
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 1.0, C, N);
#endif
      t = seconds() - t;
      check_mat (C, C_unopt, N);
      printf ("%f [s]  GFLOPS %f  |MKL|\n", t,
	      (float) N * N * N * 2 / t / 1000 / 1000 / 1000);
#endif

    }


	/** memory free **/
  free (A);
  free (B);
  free (C);
  free (C_unopt);

  return 0;

}


#if 1

void
do_AVX512_block (int n, int si, int sj, int sk, REAL * A, REAL * B, REAL * C)
{
  int i, j, k;

  for (i = si; i < si + BLOCKSIZE; i += 8)
    {
      for (j = sj; j < sj + BLOCKSIZE; ++j)
	{
	  __m512d c0 = _mm512_load_pd (C + i + j * n);	/* cij = C[i+j*n]] */

	  for (k = sk; k < sk + BLOCKSIZE; k++)
	    c0 =
	      _mm512_add_pd (c0,
			     _mm512_mul_pd (_mm512_load_pd (A + i + k * n),
					    _mm512_set1_pd (B[k + j * n])));
	  /* cij += A[i+k*n] * B[k+j*n] */
	  _mm512_store_pd (C + i + j * n, c0);	/* C[i+j*n] = cij */
	}
    }
}

void
do_AVX512_UNROLL_block (int n, int si, int sj, int sk, REAL * A, REAL * B,
			REAL * C)
{
  int i, j, k, r;

  for (i = si; i < si + BLOCKSIZE; i += UNROLL * 8)
    {
      for (j = sj; j < sj + BLOCKSIZE; j++)
	{
	  __m512d c[UNROLL];
	  for (r = 0; r < UNROLL; r++)
	    c[r] = _mm512_load_pd (C + i + r * 8 + j * n);	/* cij = C[i+j*n]] & UNROLL */

	  for (k = sk; k < sk + BLOCKSIZE; k++)
	    {
	      __m512d bb =
		_mm512_broadcastsd_pd (_mm_load_sd (B + j * n + k));
	      for (r = 0; r < UNROLL; r++)
		c[r] =
		  _mm512_fmadd_pd (_mm512_load_pd (A + n * k + r * 8 + i), bb,
				   c[r]);
	      /* cij += A[i+k*n] * B[k+j*n] */
	    }
	  for (r = 0; r < UNROLL; r++)
	    _mm512_store_pd ((C + i + r * 8 + j * n), c[r]);	/* C[i+j*n] = cij */

	}
    }

}


void
dgemm_AVX512_blocking (REAL * A, REAL * B, REAL * C, int n)
{
  int sj, si, sk;

  for (sj = 0; sj < n; sj += BLOCKSIZE)
    for (si = 0; si < n; si += BLOCKSIZE)
      for (sk = 0; sk < n; sk += BLOCKSIZE)
	do_AVX512_block (n, si, sj, sk, A, B, C);
}

void
dgemm_AVX512_UNROLL_blocking (REAL * A, REAL * B, REAL * C, int n)
{
  int sj, si, sk;

  for (sj = 0; sj < n; sj += BLOCKSIZE)
    for (si = 0; si < n; si += BLOCKSIZE)
      for (sk = 0; sk < n; sk += BLOCKSIZE)
	do_AVX512_UNROLL_block (n, si, sj, sk, A, B, C);
}

void
dgemm_AVX512_UNROLL_blocking_OMP (REAL * A, REAL * B, REAL * C, int n)
{
  int sj, si, sk;

#pragma omp parallel for private(si,sk)
  for (sj = 0; sj < n; sj += BLOCKSIZE)
    for (si = 0; si < n; si += BLOCKSIZE)
      for (sk = 0; sk < n; sk += BLOCKSIZE)
	do_AVX512_UNROLL_block (n, si, sj, sk, A, B, C);
}

/* AVX & AVX+OpenMP */
void
dgemm_AVX2 (REAL * A, REAL * B, REAL * C, int n)	/*AVX2 */
{
  int i, j, k;

#if defined(FP_SINGLE)
  for (i = 0; i < n; i += 8)
    {
      for (j = 0; j < n; j++)
	{
	  __m256 c0 = _mm256_load_ps (C + i + j * n);	/* cij = C[i+j*n]] */

	  for (k = 0; k < n; k++)
	    c0 =
	      _mm256_add_ps (c0,
			     _mm256_mul_ps (_mm256_load_ps (A + i + k * n),
					    _mm256_broadcast_ss (B + k +
								 j * n)));
	  /* cij += A[i+k*n] * B[k+j*n] */
	  _mm256_store_ps (C + i + j * n, c0);	/* C[i+j*n] = cij */
	}
    }
#else
  for (i = 0; i < n; i += 4)
    {
      for (j = 0; j < n; j++)
	{
	  __m256d c0 = _mm256_load_pd (C + i + j * n);	/* cij = C[i+j*n]] */

	  for (k = 0; k < n; k++)
	    c0 =
	      _mm256_add_pd (c0,
			     _mm256_mul_pd (_mm256_load_pd (A + i + k * n),
					    _mm256_broadcast_sd (B + k +
								 j * n)));
	  /* cij += A[i+k*n] * B[k+j*n] */
	  _mm256_store_pd (C + i + j * n, c0);	/* C[i+j*n] = cij */
	}
    }
#endif
}

void
dgemm_AVX512 (REAL * A, REAL * B, REAL * C, int n)
{
  int i, j, k;

  for (i = 0; i < n; i += 8)	/*AVX512 */
    {
      for (j = 0; j < n; j++)
	{
	  __m512d c0 = _mm512_load_pd (C + i + j * n);	/* cij = C[i+j*n]] */

	  for (k = 0; k < n; k++)
	    c0 =
	      _mm512_add_pd (c0,
			     _mm512_mul_pd (_mm512_load_pd (A + i + k * n),
					    _mm512_set1_pd (B[k + j * n])));
	  /* cij += A[i+k*n] * B[k+j*n] */
	  _mm512_store_pd (C + i + j * n, c0);	/* C[i+j*n] = cij */
	}
    }

}


void
dgemm_AVX512_UNROLL (REAL * A, REAL * B, REAL * C, int n)
{
  int i, j, k, r;

  for (i = 0; i < n; i += UNROLL * 8)	/*AVX512 & UNROLL */
    {
      for (j = 0; j < n; j++)
	{
	  __m512d c[UNROLL];
	  for (r = 0; r < UNROLL; r++)
	    c[r] = _mm512_load_pd (C + i + r * 8 + j * n);	/* cij = C[i+j*n]] & UNROLL */

	  for (k = 0; k < n; k++)
	    {
	      __m512d bb =
		_mm512_broadcastsd_pd (_mm_load_sd (B + j * n + k));
	      for (r = 0; r < UNROLL; r++)
		c[r] =
		  _mm512_fmadd_pd (_mm512_load_pd (A + n * k + r * 8 + i), bb,
				   c[r]);
	      /* cij += A[i+k*n] * B[k+j*n] */
	    }
	  for (r = 0; r < UNROLL; r++)
	    _mm512_store_pd ((C + i + r * 8 + j * n), c[r]);	/* C[i+j*n] = cij */
	}
    }
}


#if 1
void
dgemm_AVX_OMP (REAL * A, REAL * B, REAL * C, int n)
{
  int i, j, k;
#pragma omp parallel for private(j,k)
#if defined(FP_SINGLE)
  for (i = 0; i < n; i += 8)
    {
      for (j = 0; j < n; j++)
	{
	  __m256 c0 = _mm256_load_ps (C + i + j * n);	/* cij = C[i+j*n]] */

	  for (k = 0; k < n; k++)
	    c0 =
	      _mm256_add_ps (c0,
			     _mm256_mul_ps (_mm256_load_ps (A + i + k * n),
					    _mm256_broadcast_ss (B + k +
								 j * n)));
	  /* cij += A[i+k*n] * B[k+j*n] */
	  _mm256_store_ps (C + i + j * n, c0);	/* C[i+j*n] = cij */
	}
    }
#else
  for (i = 0; i < n; i += 4)
    {
      for (j = 0; j < n; j++)
	{
	  __m256d c0 = _mm256_load_pd (C + i + j * n);	/* cij = C[i+j*n]] */

	  for (k = 0; k < n; k++)
	    c0 =
	      _mm256_add_pd (c0,
			     _mm256_mul_pd (_mm256_load_pd (A + i + k * n),
					    _mm256_broadcast_sd (B + k +
								 j * n)));
	  /* cij += A[i+k*n] * B[k+j*n] */
	  _mm256_store_pd (C + i + j * n, c0);	/* C[i+j*n] = cij */
	}
    }
#endif
}
#endif

#endif
