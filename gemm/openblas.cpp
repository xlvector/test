#include <cblas.h>
#include <stdlib.h>
#include <math.h>

#include <chrono>
#include <iostream>

#define NOW() std::chrono::high_resolution_clock::now()
#define ELAPSED(msg, x) std::cout << msg << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(NOW() - x).count() << "ms" << std::endl

int gemm(size_t N, size_t M, size_t F) {
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

int main(int argc, char ** argv) {
  int N = atoi(argv[1]);
  int M = 100;
  int F = 32;

  gemm(N, M, F);  
}
