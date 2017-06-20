#include <mkl.h>
#include <stdlib.h>
#include <math.h>

#include <chrono>
#include <iostream>
#include <eigen3/Eigen/Dense>
using namespace Eigen;

#define NOW() std::chrono::high_resolution_clock::now()
#define ELAPSED(msg, x) std::cout << msg << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(NOW() - x).count() << "ms" << std::endl

typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixXfR;

int mkl_gemm(size_t N, size_t M, size_t F) {
  float * a;
  float * b;
  float * c;
  a = (float *) malloc(F * N * sizeof(float));
  b = (float *) malloc(M * F * sizeof(float));
  c = (float *) malloc(M * N * sizeof(float));

  for (int i = 0; i < F; ++i) {
    for (int j = 0; j < N; ++j) {
      a[i * N + j] = ((i + j) % 10000 - 5000) / 5000.0;
    }
  }

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < F; ++j) {
      b[i * F + j] = ((i + j) % 10000 - 5000) / 5000.0;
    }
  }
  
  for (int i = 0; i < M * N; ++i) {
    c[i] = 0.0;
  }
  
  std::cout << M << "\t" << F << "\t" << N << std::endl;
  auto start = NOW();
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              M, N, F, 1.0f, b, F, a, N, 0.0f, c, N);
  ELAPSED("mkl", start);

  float avg = 0;
  for (int i = 0; i < M * N; ++i) {
    avg += c[i];
  }
  std::cout << "avg value of c: " << avg / (float)(M) / (float)(N) << std::endl;
  free(a);
  free(b);
  free(c);
}

int eigen_gemm(int N, int M, int F) {
  MatrixXfR a = MatrixXfR::Random(F, N);
  MatrixXfR b = MatrixXfR::Random(M, F);

  for (int i = 0; i < F; ++i) {
    for (int j = 0; j < N; ++j) {
      a(i, j) = ((i + j) % 10000 - 5000) / 5000.0;
    }
  }

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < F; ++j) {
      b(i, j) = ((i + j) % 10000 - 5000) / 5000.0;
    }
  }
  
  auto start = NOW();
  MatrixXfR c = b * a;
  ELAPSED("eigen", start);

  float avg = 0;
  for (int i = 0; i < M * N; ++i) {
    avg += c(i / N, i % N);
  }
  std::cout << "avg value of c: " << avg / (float)(M) / (float)(N) << std::endl;
}

int main(int argc, char ** argv) {
  int N = atoi(argv[1]);
  int M = 100;
  int F = 32;

  mkl_gemm(N, M, F);
  eigen_gemm(N, M, F);
  
}
