// Nonlinear optimization in C++
// https://github.com/JSzitas/nlsolver
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2023- Juraj Szitas
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef NLSOLVER_H_
#define NLSOLVER_H_

#if defined(__clang__)
#pragma clang diagnostic push
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "./tinyqr.h"

// TODO(JSzitas): delete
template <typename T>
void print_vector(T x) {
  for (auto &val : x) {
    std::cout << val << ",";
  }
  std::cout << "\n";
}

// mostly dot products and other fun vector math stuff
namespace nlsolver::math {
template <typename T>
[[maybe_unused]] inline T dot(const T *x, const T *y, int f) {
  T s = 0;
  for (int z = 0; z < f; z++) {
    s += (*x) * (*y);
    x++;
    y++;
  }
  return s;
}

template <typename T>
[[maybe_unused]] inline T fast_sum(const T *x, int size) {
  T result = 0;
  for (int i = 0; i < size; i++) {
    result += (*x);
    x++;
  }
  return result;
}

template <typename T>
[[maybe_unused]] inline T vec_scalar_mult(const T *vec, const T *scalar,
                                          int f) {
  T result = 0;
  // load single scalar
  // Don't forget the remaining values.
  for (int i = 0; i < f; i++) {
    result += *vec * *scalar;
    vec++;
  }
  return result;
}
template <typename T>
[[maybe_unused]] inline T norm(const T *x, int f) {
  T s = 0;
  for (int z = 0; z < f; z++) {
    s += (*x) * (*x);
    x++;
  }
  return std::sqrt(s);
}
template <typename T>
[[maybe_unused]] inline T norm_diff(const T *x, const T *y, int f) {
  T s = 0;
  for (int z = 0; z < f; z++) {
    T tmp = *x + *y;
    s += tmp * tmp;
    x++;
    y++;
  }
  return std::sqrt(s);
}
template <typename T>
[[maybe_unused]] inline void a_plus_b(T *a, const T *b, int f) {
  for (int i = 0; i < f; i++) {
    *a += *b;
    a++;
    b++;
  }
}
template <typename T>
[[maybe_unused]] inline void a_minus_b(T *a, const T *b, int f) {
  for (int i = 0; i < f; i++) {
    *a -= *b;
    a++;
    b++;
  }
}
template <typename T>
[[maybe_unused]] inline void a_plus_b_to_c(const T *a, const T *b, T *c,
                                           int f) {
  for (int i = 0; i < f; i++) {
    *c = *a + *b;
    a++;
    b++;
    c++;
  }
}

template <typename T>
[[maybe_unused]] inline void a_minus_b_to_c(const T *a, const T *b, T *c,
                                            int f) {
  for (int i = 0; i < f; i++) {
    *c = *a - *b;
    a++;
    b++;
    c++;
  }
}

template <typename T>
[[maybe_unused]] inline T sum_a_plus_b_times_c(const T *a, const T *b,
                                               const T *c, int f) {
  T result = 0;
  for (int i = 0; i < f; i++) {
    result += (*a + *b) * (*c);
    a++;
    b++;
    c++;
  }
  return result;
}

template <typename T>
[[maybe_unused]] inline T sum_a_minus_b_times_c(const T *a, const T *b,
                                                const T *c, int f) {
  T result = 0;
  for (int i = 0; i < f; i++) {
    result += (*a - *b) * (*c);
    a++;
    b++;
    c++;
  }
  return result;
}
template <typename T>
[[maybe_unused]] inline void a_plus_scalar_to_b(const T *a, const T scalar,
                                                T *b, int f) {
  for (int i = 0; i < f; i++) {
    *b = (*a + scalar);
    a++;
    b++;
  }
}

template <typename T>
[[maybe_unused]] inline void a_minus_scalar_to_b(const T *a, const T scalar,
                                                 T *b, int f) {
  for (int i = 0; i < f; i++) {
    *b = (*a - scalar);
    a++;
    b++;
  }
}
template <typename T>
[[maybe_unused]] inline void a_mult_scalar_to_b(const T *a, const T scalar,
                                                T *b, int f) {
  for (int i = 0; i < f; i++) {
    *b = (*a * scalar);
    a++;
    b++;
  }
}
template <typename T>
[[maybe_unused]] inline void a_mult_scalar_add_b(const T *a, const T scalar,
                                                 T *b, int f) {
  for (int i = 0; i < f; i++) {
    *b += (*a * scalar);
    a++;
    b++;
  }
}
template <typename T>
[[maybe_unused]] inline void a_minus_b_mult_scalar_add_c(const T *a, const T *b,
                                                         const T scalar, T *c,
                                                         int f) {
  for (int i = 0; i < f; i++) {
    *c += (*a - *b) * scalar;
    a++;
    b++;
    c++;
  }
}
template <typename T>
[[maybe_unused]] inline void a_mul_scalar(T *a, const T scalar, int f) {
  for (int i = 0; i < f; i++) {
    *a *= scalar;
    a++;
  }
}
// this is best defined here even though it is not technically a function
// we probably want to reuse much
template <typename T>
[[maybe_unused]] inline void hessian_update_inner_loop(
    T *inv_hessian, const T *step, const T *grad_diff_inv_hess, const T rho,
    const T denom, const int n_dim) {
  for (int j = 0; j < n_dim; j++) {
    for (int i = 0; i < n_dim; i++) {
      // do not replace this with -= or the whole thing falls apart
      // because of operator order precedence - e.g. whole rhs would
      // get evaluated before -=, whereas we want to do inv_hessian - first part
      // + second part
      *(inv_hessian + j * n_dim + i) =
          *(inv_hessian + j * n_dim + i) -
          // first part
          rho * (*(step + i) * *(grad_diff_inv_hess + j) +
                 *(grad_diff_inv_hess + i) * *(step + j) +
                 // second part ->  multiply(step[i], denom * step[j])
                 denom * *(step + i) * *(step + j));
    }
  }
}
template <typename scalar_t>
void cholesky(std::vector<scalar_t> &A) {
  const auto n = static_cast<size_t>(std::sqrt(A.size()));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < i; ++j) {
      scalar_t sum = 0;
      for (size_t k = 0; k < j; ++k) {
        sum += A[i * n + k] * A[j * n + k];
      }
      A[i * n + j] = (1.0 / A[j * n + j] * (A[i * n + j] - sum));
    }
    // diagonal elements only
    scalar_t sum = 0;
    for (size_t k = 0; k < i; ++k) {
      sum += A[i * n + k] * A[i * n + k];
    }
    A[i * n + i] = sqrt(A[i * n + i] - sum);
  }
}
template <typename scalar_t>
void backsolve_inplace_t(std::vector<scalar_t> &U, std::vector<scalar_t> &b,
                         const size_t n) {
  int i = static_cast<int>(n) - 1;
  for (; i >= 0; --i) {
    scalar_t sum = 0.0;
    for (size_t j = i + 1; j < n; ++j) {
      sum += U[j * n + i] * b[j];
    }
    b[i] = (b[i] - sum) / U[i * n + i];
  }
}
template <typename scalar_t = double>
void forwardsolve_inplace(std::vector<scalar_t> &update,
                          const std::vector<scalar_t> &L,
                          const std::vector<scalar_t> &b, const size_t n) {
  std::fill(update.begin(), update.end(), 0.0);
  for (size_t i = 0; i < n; ++i) {
    scalar_t sum = 0.0;
    for (size_t j = 0; j < i; ++j) {
      sum += L[i * n + j] * update[j];
    }
    update[i] = (b[i] - sum) / L[i + i * n];
  }
}
template <typename T>
bool is_diagonal(const std::vector<T> &A) {
  const size_t size = std::sqrt(A.size());
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      if (i != j &&
          A[i * size + j] > std::numeric_limits<T>::epsilon() * 1e12) {
        return false;
      }
    }
  }
  return true;
}

template <typename scalar_t>
void get_update_with_hessian(std::vector<scalar_t> &update,
                             std::vector<scalar_t> &hess,
                             std::vector<scalar_t> &grad) {
  // if hessian is diagonal we have a much faster update - since the inverse
  // is just 1/diagonal entry, and the products are just simple multiplications
  const size_t n = grad.size();
  if (is_diagonal(hess)) {
    // much faster update loop
    for (size_t i = 0; i < n; i++) {
      update[i] = grad[i] / hess[i * n + i];
    }
    return;
  }
  // note that this overwrites the hessian due to the cholesky decomposition
  // being in-place
  // compute cholesky of hessian
  cholesky(hess);
  // use cholesky to solve system of equations
  forwardsolve_inplace(update, hess, grad, n);
  backsolve_inplace_t(hess, update, n);
}

// inspired by annoylib, see
// https://github.com/spotify/annoy/blob/main/src/annoylib.h
#if !defined(NO_MANUAL_VECTORIZATION) && defined(__GNUC__) && \
    (__GNUC__ > 6) && defined(__AVX512F__)
#define DOT_USE_AVX512
#endif
#if !defined(NO_MANUAL_VECTORIZATION) && defined(__AVX__) && \
    defined(__SSE__) && defined(__SSE2__) && defined(__SSE3__)
#define DOT_USE_AVX
#endif

#if defined(DOT_USE_AVX) || defined(DOT_USE_AVX512)
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__)
#include <immintrin.h>  //<x86intrin.h>
#endif
#endif

#ifdef DOT_USE_AVX
// Horizontal single sum of 256bit vector.
inline float hsum256_ps_avx(__m256 v) {
  const __m128 x128 =
      _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
  const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
  const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
  return _mm_cvtss_f32(x32);
}

inline double hsum256_pd_avx(__m256d v) {
  __m128d vlow = _mm256_castpd256_pd128(v);
  __m128d vhigh = _mm256_extractf128_pd(v, 1);  // high 128
  vlow = _mm_add_pd(vlow, vhigh);               // reduce down to 128
  __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
  return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}

// overload
template <>
[[maybe_unused]] inline float dot<float>(const float *x, const float *y,
                                         int f) {
  float result = 0;
  if (f > 7) {
    __m256 d = _mm256_setzero_ps();
    for (; f > 7; f -= 8) {
      d = _mm256_add_ps(d,
                        _mm256_mul_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y)));
      x += 8;
      y += 8;
    }
    // Sum all floats in dot register.
    result += hsum256_ps_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    result += *x * *y;
    x++;
    y++;
  }
  return result;
}

// second overload
template <>
[[maybe_unused]] inline double dot<double>(const double *x, const double *y,
                                           int f) {
  double result = 0;
  if (f > 3) {
    __m256d d = _mm256_setzero_pd();
    for (; f > 3; f -= 4) {
      d = _mm256_add_pd(d,
                        _mm256_mul_pd(_mm256_loadu_pd(x), _mm256_loadu_pd(y)));
      x += 4;
      y += 4;
    }
    // Sum all floats in dot register.
    result += hsum256_pd_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    result += *x * *y;
    x++;
    y++;
  }
  return result;
}

template <>
[[maybe_unused]] inline float fast_sum<float>(const float *x, int size) {
  float result = 0;
  if (size > 7) {
    __m256 d = _mm256_setzero_ps();
    for (; size > 7; size -= 8) {
      d = _mm256_add_ps(d, _mm256_loadu_ps(x));
      x += 8;
    }
    // Sum all floats in dot register.
    result += hsum256_ps_avx(d);
  }
  // Don't forget the remaining values.
  for (; size > 0; size--) {
    result += *x;
    x++;
  }
  return result;
}

template <>
[[maybe_unused]] inline double fast_sum<double>(const double *x, int size) {
  double result = 0;
  if (size > 3) {
    __m256d d = _mm256_setzero_pd();
    for (; size > 3; size -= 4) {
      d = _mm256_add_pd(d, _mm256_loadu_pd(x));
      x += 4;
    }
    // Sum all floats in dot register.
    result += hsum256_pd_avx(d);
  }
  // Don't forget the remaining values.
  for (; size > 0; size--) {
    result += *x;
    x++;
  }
  return result;
}

// multiply a vector by scalar
template <>
[[maybe_unused]] inline float vec_scalar_mult<float>(const float *vec,
                                                     const float *scalar,
                                                     int f) {
  float result = 0;
  // load single scalar
  const __m256 s = _mm256_set1_ps(*scalar);
  if (f > 7) {
    __m256 d = _mm256_setzero_ps();
    for (; f > 7; f -= 8) {
      d = _mm256_add_ps(d, _mm256_mul_ps(_mm256_loadu_ps(vec), s));
      vec += 8;
    }
    // Sum all floats in dot register.
    result += hsum256_ps_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    result += *vec * *scalar;
    vec++;
  }
  return result;
}

// overload for doubles
template <>
[[maybe_unused]] inline double vec_scalar_mult<double>(const double *vec,
                                                       const double *scalar,
                                                       int f) {
  double result = 0;
  // load single scalar
  const __m256d s = _mm256_set1_pd(*scalar);
  if (f > 3) {
    __m256d d = _mm256_setzero_pd();
    for (; f > 3; f -= 4) {
      d = _mm256_add_pd(d, _mm256_mul_pd(_mm256_loadu_pd(vec), s));
      vec += 8;
    }
    // Sum all floats in dot register.
    result += hsum256_pd_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    result += *vec * *scalar;
    vec++;
  }
  return result;
}
template <>
[[maybe_unused]] inline float norm<float>(const float *x, int f) {
  float result = 0;
  if (f > 7) {
    __m256 d = _mm256_setzero_ps();
    for (; f > 7; f -= 8) {
      d = _mm256_add_ps(d,
                        _mm256_mul_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(x)));
      x += 8;
    }
    // Sum all floats in dot register.
    result += hsum256_ps_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    result += (*x) * (*x);
    x++;
  }
  return std::sqrt(result);
}
template <>
[[maybe_unused]] inline double norm<double>(const double *x, int f) {
  double result = 0;
  if (f > 3) {
    __m256d d = _mm256_setzero_pd();
    for (; f > 3; f -= 4) {
      d = _mm256_add_pd(d,
                        _mm256_mul_pd(_mm256_loadu_pd(x), _mm256_loadu_pd(x)));
      x += 8;
    }
    // Sum all floats in dot register.
    result += hsum256_pd_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    result += std::pow(*x, 2);
    x++;
  }
  return sqrt(result);
}
template <>
[[maybe_unused]] inline float norm_diff<float>(const float *x, const float *y,
                                               int f) {
  float result = 0;
  if (f > 7) {
    __m256 d = _mm256_setzero_ps();
    __m256 tmp = _mm256_setzero_ps();
    for (; f > 7; f -= 8) {
      tmp = _mm256_sub_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
      d = _mm256_add_ps(d, _mm256_mul_ps(tmp, tmp));
      x += 8;
      y += 8;
    }
    // Sum all floats in dot register.
    result += hsum256_ps_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    float tmp = *x + *y;
    result += tmp * tmp;
    x++;
    y++;
  }
  return std::sqrt(result);
}
template <>
[[maybe_unused]] inline double norm_diff<double>(const double *x,
                                                 const double *y, int f) {
  double result = 0;
  if (f > 3) {
    __m256 d = _mm256_setzero_pd();
    __m256 tmp = _mm256_setzero_pd();
    for (; f > 3; f -= 4) {
      tmp = _mm256_sub_pd(_mm256_loadu_pd(x), _mm256_loadu_pd(y));
      d = _mm256_add_pd(d, _mm256_mul_pd(tmp, tmp));
      x += 4;
      y += 4;
    }
    // Sum all floats in dot register.
    result += hsum256_pd_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    double tmp = *x + *y;
    result += tmp * tmp;
    x++;
    y++;
  }
  return std::sqrt(result);
}
template <>
[[maybe_unused]] inline void a_plus_b(float *a, const float *b, int f) {
  if (f > 7) {
    for (; f > 7; f -= 8) {
      __m256 d = _mm256_add_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
      // store results
      _mm256_storeu_ps(a, d);
      // offset
      a += 8;
      b += 8;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *a += *b;
    a++;
    b++;
  }
}
template <>
[[maybe_unused]] inline void a_plus_b(double *a, const double *b, int f) {
  if (f > 3) {
    for (; f > 3; f -= 4) {
      __m256d d = _mm256_add_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b));
      // store results
      _mm256_storeu_pd(a, d);
      // offset
      a += 4;
      b += 4;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *a += *b;
    a++;
    b++;
  }
}
template <>
[[maybe_unused]] inline void a_minus_b(float *a, const float *b, int f) {
  if (f > 7) {
    for (; f > 7; f -= 8) {
      __m256 d = _mm256_setzero_ps();
      d = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
      // store results
      _mm256_storeu_ps(a, d);
      // offset
      a += 8;
      b += 8;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *a -= *b;
    a++;
    b++;
  }
}
template <>
[[maybe_unused]] inline void a_minus_b(double *a, const double *b, int f) {
  if (f > 3) {
    for (; f > 3; f -= 4) {
      __m256d d = _mm256_setzero_pd();
      d = _mm256_sub_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b));
      // store results
      _mm256_storeu_pd(a, d);
      // offset
      a += 4;
      b += 4;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *a -= *b;
    a++;
    b++;
  }
}
template <>
[[maybe_unused]] inline void a_plus_b_to_c(const float *a, const float *b,
                                           float *c, int f) {
  if (f > 7) {
    for (; f > 7; f -= 8) {
      __m256 d = _mm256_setzero_ps();
      d = _mm256_add_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
      // store results
      _mm256_storeu_ps(c, d);
      // offset
      a += 8;
      b += 8;
      c += 8;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *c = *a + *b;
    a++;
    b++;
    c++;
  }
}

template <>
[[maybe_unused]] inline void a_plus_b_to_c(const double *a, const double *b,
                                           double *c, int f) {
  if (f > 3) {
    for (; f > 3; f -= 4) {
      __m256d d = _mm256_setzero_pd();
      d = _mm256_add_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b));
      // store results
      _mm256_store_pd(c, d);
      // offset
      a += 4;
      b += 4;
      c += 4;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *c = *a + *b;
    a++;
    b++;
    c++;
  }
}

template <>
[[maybe_unused]] inline void a_minus_b_to_c(const float *a, const float *b,
                                            float *c, int f) {
  if (f > 7) {
    for (; f > 7; f -= 8) {
      __m256 d = _mm256_setzero_ps();
      d = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
      // store results
      _mm256_storeu_ps(c, d);
      // offset
      a += 8;
      b += 8;
      c += 8;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *c = *a - *b;
    a++;
    b++;
    c++;
  }
}

template <>
[[maybe_unused]] inline void a_minus_b_to_c(const double *a, const double *b,
                                            double *c, int f) {
  if (f > 3) {
    for (; f > 3; f -= 4) {
      __m256d d = _mm256_setzero_pd();
      d = _mm256_sub_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b));
      // store results
      _mm256_storeu_pd(c, d);
      // offset
      a += 4;
      b += 4;
      c += 4;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *c = *a - *b;
    a++;
    b++;
    c++;
  }
}
template <>
[[maybe_unused]] inline double sum_a_plus_b_times_c(const double *a,
                                                    const double *b,
                                                    const double *c, int f) {
  double result = 0;
  if (f > 4) {
    __m256d d = _mm256_setzero_pd();
    for (; f > 3; f -= 4) {
      d = _mm256_mul_pd(_mm256_loadu_pd(c),
                        _mm256_add_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b)));
      a += 4;
      b += 4;
      c += 4;
    }
    // Sum all floats in dot register.
    result += hsum256_pd_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    result += ((*a) + (*b)) * (*c);
    a++;
    b++;
    c++;
  }
  return result;
}

template <>
[[maybe_unused]] inline float sum_a_plus_b_times_c(const float *a,
                                                   const float *b,
                                                   const float *c, int f) {
  float result = 0;
  if (f > 8) {
    __m256 d = _mm256_setzero_ps();
    for (; f > 3; f -= 4) {
      d = _mm256_mul_ps(_mm256_loadu_ps(c),
                        _mm256_add_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b)));
      a += 8;
      b += 8;
      c += 8;
    }
    // Sum all floats in dot register.
    result += hsum256_ps_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    result += ((*a) + (*b)) * (*c);
    a++;
    b++;
    c++;
  }
  return result;
}

template <>
[[maybe_unused]] inline double sum_a_minus_b_times_c(const double *a,
                                                     const double *b,
                                                     const double *c, int f) {
  double result = 0;
  if (f > 3) {
    __m256d d = _mm256_setzero_pd();
    for (; f > 3; f -= 4) {
      d = _mm256_mul_pd(_mm256_loadu_pd(c),
                        _mm256_sub_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b)));
      a += 4;
      b += 4;
      c += 4;
    }
    // Sum all floats in dot register.
    result += hsum256_pd_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    result += ((*a) - (*b)) * (*c);
    a++;
    b++;
    c++;
  }
  return result;
}

template <>
[[maybe_unused]] inline float sum_a_minus_b_times_c(const float *a,
                                                    const float *b,
                                                    const float *c, int f) {
  float result = 0;
  if (f > 7) {
    __m256 d = _mm256_setzero_ps();
    for (; f > 7; f -= 8) {
      d = _mm256_mul_ps(_mm256_loadu_ps(c),
                        _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b)));
      a += 8;
      b += 8;
      c += 8;
    }
    // Sum all floats in dot register.
    result += hsum256_ps_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    result += ((*a) - (*b)) * (*c);
    a++;
    b++;
    c++;
  }
  return result;
}

template <>
[[maybe_unused]] inline void a_plus_scalar_to_b(const float *a,
                                                const float scalar, float *b,
                                                int f) {
  // load single scalar
  const __m256 s = _mm256_set1_ps(scalar);
  if (f > 7) {
    for (; f > 7; f -= 8) {
      __m256 d = _mm256_setzero_ps();
      d = _mm256_add_ps(_mm256_loadu_ps(a), s);
      // store results
      _mm256_storeu_ps(b, d);
      // offset
      a += 8;
      b += 8;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *b = *a + scalar;
    a++;
    b++;
  }
}

template <>
[[maybe_unused]] inline void a_plus_scalar_to_b(const double *a,
                                                const double scalar, double *b,
                                                int f) {
  // load single scalar
  const __m256d s = _mm256_set1_pd(scalar);
  if (f > 3) {
    for (; f > 3; f -= 4) {
      __m256d d = _mm256_setzero_pd();
      d = _mm256_add_pd(_mm256_loadu_pd(a), s);
      // store results
      _mm256_storeu_pd(b, d);
      // offset
      a += 4;
      b += 4;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *b = *a + scalar;
    a++;
    b++;
  }
}
template <>
[[maybe_unused]] inline void a_minus_scalar_to_b(const float *a,
                                                 const float scalar, float *b,
                                                 int f) {
  // load single scalar
  const __m256 s = _mm256_set1_ps(scalar);
  if (f > 7) {
    for (; f > 7; f -= 8) {
      __m256 d = _mm256_setzero_ps();
      d = _mm256_sub_ps(_mm256_loadu_ps(a), s);
      // store results
      _mm256_storeu_ps(b, d);
      // offset
      a += 8;
      b += 8;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *b = *a - scalar;
    a++;
    b++;
  }
}

template <>
[[maybe_unused]] inline void a_minus_scalar_to_b(const double *a,
                                                 const double scalar, double *b,
                                                 int f) {
  // load single scalar
  const __m256d s = _mm256_set1_pd(scalar);
  if (f > 3) {
    for (; f > 3; f -= 4) {
      __m256d d = _mm256_setzero_pd();
      d = _mm256_sub_pd(_mm256_loadu_pd(a), s);
      // store results
      _mm256_storeu_pd(b, d);
      // offset
      a += 4;
      b += 4;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *b = *a - scalar;
    a++;
    b++;
  }
}
template <>
[[maybe_unused]] inline void a_mult_scalar_to_b(const float *a,
                                                const float scalar, float *b,
                                                int f) {
  // load single scalar
  const __m256 s = _mm256_set1_ps(scalar);
  if (f > 7) {
    for (; f > 7; f -= 8) {
      __m256 d = _mm256_setzero_ps();
      d = _mm256_mul_ps(_mm256_loadu_ps(a), s);
      // store results
      _mm256_storeu_ps(b, d);
      // offset
      a += 8;
      b += 8;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *b = *a * scalar;
    a++;
    b++;
  }
}
template <>
[[maybe_unused]] inline void a_mult_scalar_to_b(const double *a,
                                                const double scalar, double *b,
                                                int f) {
  // load single scalar
  const __m256d s = _mm256_set1_pd(scalar);
  if (f > 3) {
    for (; f > 3; f -= 4) {
      __m256d d = _mm256_setzero_pd();
      d = _mm256_mul_pd(_mm256_loadu_pd(a), s);
      // store results
      _mm256_storeu_pd(b, d);
      // offset
      a += 4;
      b += 4;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *b = *a * scalar;
    a++;
    b++;
  }
}
template <>
[[maybe_unused]] inline void a_mult_scalar_add_b(const float *a,
                                                 const float scalar, float *b,
                                                 int f) {
  // load single scalar
  const __m256 s = _mm256_set1_ps(scalar);
  if (f > 7) {
    for (; f > 7; f -= 8) {
      __m256 d = _mm256_setzero_ps();
      d = _mm256_add_ps(_mm256_loadu_ps(b),
                        _mm256_mul_ps(_mm256_loadu_ps(a), s));
      // store results
      _mm256_storeu_ps(b, d);
      // offset
      a += 8;
      b += 8;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *b += *a * scalar;
    a++;
    b++;
  }
}
template <>
[[maybe_unused]] inline void a_mult_scalar_add_b(const double *a,
                                                 const double scalar, double *b,
                                                 int f) {
  // load single scalar
  const __m256d s = _mm256_set1_pd(scalar);
  if (f > 3) {
    for (; f > 3; f -= 4) {
      __m256d d = _mm256_setzero_pd();
      d = _mm256_add_pd(_mm256_loadu_pd(b),
                        _mm256_mul_pd(_mm256_loadu_pd(a), s));
      // store results
      _mm256_storeu_pd(b, d);
      // offset
      a += 4;
      b += 4;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *b += *a * scalar;
    a++;
    b++;
  }
}
template <>
[[maybe_unused]] inline void a_minus_b_mult_scalar_add_c(const float *a,
                                                         const float *b,
                                                         const float scalar,
                                                         float *c, int f) {
  // load single scalar
  const __m256 s = _mm256_set1_ps(scalar);
  if (f > 7) {
    for (; f > 7; f -= 8) {
      __m256 d = _mm256_setzero_ps();
      d = _mm256_add_ps(_mm256_loadu_ps(c),
                        _mm256_mul_ps(s, _mm256_sub_ps(_mm256_loadu_ps(a),
                                                       _mm256_loadu_ps(b))));
      // store results
      _mm256_storeu_ps(c, d);
      // offset
      a += 8;
      b += 8;
      c += 8;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *c += (*a - *b) * scalar;
    a++;
    b++;
    c++;
  }
}
template <>
[[maybe_unused]] inline void a_minus_b_mult_scalar_add_c(const double *a,
                                                         const double *b,
                                                         const double scalar,
                                                         double *c, int f) {
  // load single scalar
  const __m256d s = _mm256_set1_pd(scalar);
  if (f > 3) {
    for (; f > 3; f -= 4) {
      __m256d d = _mm256_setzero_pd();
      d = _mm256_add_pd(_mm256_loadu_pd(c),
                        _mm256_mul_pd(s, _mm256_sub_pd(_mm256_loadu_pd(a),
                                                       _mm256_loadu_pd(b))));
      // store results
      _mm256_storeu_pd(c, d);
      // offset
      a += 4;
      b += 4;
      c += 4;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *c += (*a - *b) * scalar;
    a++;
    b++;
    c++;
  }
}
template <>
[[maybe_unused]] inline void a_mul_scalar(float *a, const float scalar, int f) {
  // load single scalar
  const __m256 s = _mm256_set1_ps(scalar);
  if (f > 7) {
    for (; f > 7; f -= 8) {
      __m256 d = _mm256_setzero_ps();
      d = _mm256_mul_ps(s, _mm256_loadu_ps(a));
      // store results
      _mm256_storeu_ps(a, d);
      // offset
      a += 8;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *a *= scalar;
    a++;
  }
}
template <>
[[maybe_unused]] inline void a_mul_scalar(double *a, const double scalar,
                                          int f) {
  // load single scalar
  const __m256d s = _mm256_set1_pd(scalar);
  if (f > 3) {
    for (; f > 3; f -= 4) {
      __m256d d = _mm256_setzero_pd();
      d = _mm256_mul_pd(s, _mm256_loadu_pd(a));
      // store results
      _mm256_storeu_pd(a, d);
      // offset
      a += 4;
    }
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    *a *= scalar;
    a++;
  }
}
#endif
}  // namespace nlsolver::math

namespace nlsolver::rng {
#define MAX_SIZE_64_BIT_UINT (18446744073709551615U)
template <typename scalar_t = float>
struct [[maybe_unused]] halton {
  explicit halton<scalar_t>(const scalar_t base = 2)
      : b(base), y(1), n(0), d(1), x(1) {}
  scalar_t yield() {
    x = d - n;
    if (x == 1) {
      n = 1;
      d *= b;
    } else {
      y = d;
      while (x <= y) {
        y /= b;
        n = (b + 1) * y - x;
      }
    }
    return (scalar_t)(n / d);
  }
  scalar_t operator()() { return this->yield(); }
  [[maybe_unused]] void reset() {
    b = 2;
    y = 1;
    n = 0;
    d = 1;
    x = 1;
  }
  [[maybe_unused]] std::vector<scalar_t> get_state() const {
    std::vector<scalar_t> result(5);
    result[0] = b;
    result[1] = y;
    result[2] = n;
    result[3] = d;
    result[4] = x;
    return result;
  }
  [[maybe_unused]] void set_state(const scalar_t b_, const scalar_t y_,
                                  const scalar_t n_, const scalar_t d_,
                                  const scalar_t x_) {
    this->b = b_;
    this->y = y_;
    this->n = n_;
    this->d = d_;
    this->x = x_;
  }

 private:
  scalar_t b, y, n, d, x;
};

template <typename scalar_t = float>
struct [[maybe_unused]] recurrent {
  recurrent<scalar_t>() : seed_(0.5), alpha_(0.618034), z_(alpha_ + seed_) {
    this->z -= static_cast<scalar_t>(static_cast<uint64_t>(this->z_));
  }
  [[maybe_unused]] explicit recurrent(scalar_t seed)
      : seed_(seed), alpha_(0.618034), z_(alpha_ + seed_) {
    this->z_ -= static_cast<scalar_t>(static_cast<uint64_t>(this->z_));
  }
  scalar_t yield() {
    this->z_ += this->alpha_;
    // a slightly evil way to do z % 1 with floats
    this->z -= static_cast<scalar_t>(static_cast<uint64_t>(this->z_));
    return this->z_;
  }
  scalar_t operator()() { return this->yield(); }
  [[maybe_unused]] void reset() {
    this->alpha_ = 0.618034;
    this->seed_ = 0.5;
    this->z_ = 0;
  }
  [[maybe_unused]] std::vector<scalar_t> get_state() const {
    std::vector<scalar_t> result(2);
    result[0] = this->alpha_;
    result[1] = this->z_;
    return result;
  }
  [[maybe_unused]] void set_state(scalar_t alpha = 0.618034, scalar_t z = 0) {
    this->alpha_ = alpha;
    this->z_ = z;
  }

 private:
  scalar_t alpha_ = 0.618034, seed_ = 0.5, z_ = 0;
};

template <typename scalar_t = float>
struct splitmix {
  explicit splitmix<scalar_t>() : s(12374563468) {}
  scalar_t yield() {
    uint64_t result = (s += 0x9E3779B97f4A7C15);
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
    return (scalar_t)(result ^ (result >> 31)) / (scalar_t)MAX_SIZE_64_BIT_UINT;
  }
  scalar_t operator()() { return this->yield(); }
  uint64_t yield_init() {
    uint64_t result = (s += 0x9E3779B97f4A7C15);
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
    return result ^ (result >> 31);
  }
  [[maybe_unused]] void set_state(uint64_t seed) { this->s = seed; }
  [[maybe_unused]] std::vector<scalar_t> get_state() const {
    std::vector<scalar_t> result(1);
    result[0] = this->s;
    return result;
  }

 private:
  uint64_t s;
};
template <typename scalar_t = float>
struct xoshiro {
  xoshiro<scalar_t>() {  // NOLINT
    splitmix<scalar_t> gn;
    s[0] = gn.yield_init();
    s[1] = s[0] >> 32;
    s[2] = gn.yield();
    s[3] = s[2] >> 32;
  }
  scalar_t yield() {
    uint64_t const result = s[0] + s[3];
    uint64_t const t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = bitwise_rotate(s[3], 64, 45);

    return (scalar_t)result / (scalar_t)MAX_SIZE_64_BIT_UINT;
  }
  uint64_t static bitwise_rotate(uint64_t x, int bits, int rotate_bits) {
    return (x << rotate_bits) | (x >> (bits - rotate_bits));
  }
  scalar_t operator()() { return this->yield(); }
  [[maybe_unused]] void reset() {
    splitmix<scalar_t> gn;
    s[0] = gn.yield_init();
    s[1] = s[0] >> 32;

    s[2] = gn.yield();
    s[3] = s[2] >> 32;
  }
  [[maybe_unused]] void set_state(uint64_t x, uint64_t y, uint64_t z,
                                  uint64_t t) {
    this->s[0] = x;
    this->s[1] = y;
    this->s[2] = z;
    this->s[3] = t;
  }
  [[maybe_unused]] std::vector<scalar_t> get_state() const {
    std::vector<scalar_t> result(4);
    for (size_t i = 0; i < 4; i++) {
      result[i] = this->s[i];
    }
    return result;
  }

 private:
  uint64_t s[4];
};

template <typename scalar_t = float>
struct xorshift {
  xorshift<scalar_t>() {  // NOLINT
    splitmix<scalar_t> gn;
    x[0] = gn.yield_init();
    x[1] = x[0] >> 32;
  }
  scalar_t yield() {
    uint64_t t = x[0];
    uint64_t const s = x[1];
    x[0] = s;
    t ^= t << 23;  // a
    t ^= t >> 18;  // b -- Again, the shifts and the multipliers are tunable
    t ^= s ^ (s >> 5);  // c
    x[1] = t;
    return static_cast<scalar_t>((t + s) /
                                 static_cast<scalar_t>(MAX_SIZE_64_BIT_UINT));
  }
  scalar_t operator()() { return this->yield(); }
  [[maybe_unused]] void reset() {
    splitmix<scalar_t> gn;
    x[0] = gn.yield_init();
    x[1] = x[0] >> 32;
  }
  [[maybe_unused]] void set_state(uint64_t y, uint64_t z) {
    x[0] = y;
    x[1] = z;
  }
  [[maybe_unused]] std::vector<scalar_t> get_state() const {
    std::vector<scalar_t> result(2);
    for (size_t i = 0; i < 2; i++) {
      result[i] = this->x[i];
    }
    return result;
  }

 private:
  uint64_t x[2]{};
};
}  // namespace nlsolver::rng

namespace nlsolver::finite_difference {
// The 'accuracy' can be 0, 1, 2, 3.
template <typename Callable, typename scalar_t, const size_t accuracy = 0>
void finite_difference_gradient(Callable &f, std::vector<scalar_t> &x,
                                std::vector<scalar_t> &grad) {
  // all constexpr values - this is why accuracy is a template parameter
  constexpr scalar_t eps = std::numeric_limits<scalar_t>::epsilon() * 10e7;
  constexpr std::array<scalar_t, 20> coeff = {1,   -1,   1,   -8,   8,  -1, -1,
                                              9,   -45,  45,  -9,   1,  3,  -32,
                                              168, -672, 672, -168, 32, -3};
  constexpr std::array<scalar_t, 20> coeff2 = {
      1, -1, -2, -1, 1, 2, -3, -2, -1, 1, 2, 3, -4, -3, -2, -1, 1, 2, 3, 4};
  constexpr std::array<scalar_t, 4> dd = {2, 12, 60, 840};
  constexpr int innerSteps = 2 * (accuracy + 1);
  constexpr scalar_t ddVal = dd[accuracy] * eps;
  constexpr std::array<size_t, 4> offset_index = {0, 2, 6, 12};
  constexpr size_t offset = offset_index[accuracy];
  // actual stuff that should exist at runtime
  const size_t x_size = x.size();
  std::fill(grad.begin(), grad.end(), 0.0);
  for (size_t d = 0; d < x_size; d++) {
    for (size_t s = 0; s < innerSteps; ++s) {
      scalar_t tmp = x[d];
      x[d] += coeff2[offset + s] * eps;
      grad[d] += coeff[offset + s] * f(x);
      x[d] = tmp;
    }
    grad[d] /= ddVal;
  }
}

template <typename Callable, typename scalar_t, const size_t accuracy = 0>
void finite_difference_hessian(Callable &f, std::vector<scalar_t> &x,  // NOLINT
                               std::vector<scalar_t> &hess) {
  constexpr auto eps_mult = static_cast<scalar_t>(1.0 / 4.0);
  const scalar_t eps =
      std::pow(std::numeric_limits<scalar_t>::epsilon(), eps_mult);
  // std::cout << "Epsilon: "<< eps;
  const size_t p = x.size();
  if constexpr (accuracy == 0) {
    const scalar_t denom = eps * eps;
    // this can be hoisted entirely out of the loop - it is the first
    // evaluation at current x
    const scalar_t f4 = f(x);
    for (size_t i = 0; i < p; i++) {
      const scalar_t temp_i = x[i];
      for (size_t j = 0; j < p; j++) {
        const scalar_t temp_j = x[j];
        // evaluate at x_0
        scalar_t result = f(x);
        x[i] += eps;  // x_i + eps
        x[j] += eps;  // x_j + eps
        result += f(x);
        x[j] -= eps;  // x_j - 2 * eps
        result -= f(x);
        x[i] -= eps;  // x_i
        x[j] += eps;  // x_j - eps
        result -= f(x);
        // replace x_i and x_j
        x[i] = temp_i;
        x[j] = temp_j;
        hess[i * p + j] = result / denom;
      }
    }
  } else {
    const scalar_t denom = (600.0 * eps * eps), two_eps = 2 * eps,
                   three_eps = 3 * eps, four_eps = 4 * eps;
    for (size_t i = 0; i < p; i++) {
      const scalar_t temp_i = x[i];
      for (size_t j = 0; j < p; j++) {
        scalar_t result = 0.0, temp = 0.0;
        scalar_t temp_j = x[j];
        x[i] += eps;      // x_i + eps
        x[j] -= two_eps;  // x_j - 2 * eps
        temp += f(x);
        x[i] += eps;  // x_i + 2 * eps
        x[j] += eps;  // x_j - eps
        temp += f(x);
        x[i] -= four_eps;  // x_i - 2 * eps
        x[j] += two_eps;   // x_j + eps
        temp += f(x);
        x[i] += eps;  // x_i - eps
        x[j] += eps;  // x_j + 2 * eps
        temp += f(x);
        result -= 63 * temp;
        temp = 0.0;
        // x_i remains at (x_i - eps)
        x[j] -= four_eps;  // x_j - 2 * eps
        temp += f(x);
        x[i] -= eps;  // x_i - 2 * eps
        x[j] += eps;  // x_j - eps
        temp += f(x);
        x[i] += three_eps;  // x_i + eps
        x[j] += three_eps;  // x_j + 2 * eps
        temp += f(x);
        x[i] += eps;  // x_i + 2 * eps
        x[j] -= eps;  // x_j + eps
        temp += f(x);
        result += 63 * temp;
        temp = 0.0;
        // x_i remains at (x_i + 2 * eps)
        x[j] -= three_eps;  // x_j -2 * eps
        temp += f(x);
        x[i] -= four_eps;  // x_i - 2 * eps
        x[j] += four_eps;  // x_j + 2 * eps
        temp += f(x);
        // x_i remains at (x_i - 2 * eps)
        x[j] -= four_eps;  // x_j - 2 * eps
        temp -= f(x);
        x[i] += four_eps;  // x_i + 2 * eps
        x[j] += four_eps;  // x_j + 2 * eps
        temp -= f(x);
        result += 44 * temp;
        temp = 0.0;
        x[i] -= three_eps;  // x_i - eps
        x[j] -= three_eps;  // x_j - eps
        temp += f(x);
        x[i] += two_eps;  // x_i + eps
        x[j] += two_eps;  // x_j + eps
        temp += f(x);
        // x_i remains at (x_i + eps)
        x[j] -= two_eps;  // x_j - eps
        temp -= f(x);
        x[i] -= two_eps;  // x_i - eps
        x[j] += two_eps;  // x_j + eps
        temp -= f(x);
        result += 74 * temp;
        // reset
        x[i] = temp_i;
        x[j] = temp_j;
        hess[i * p + j] = result / denom;
      }
    }
  }
}
}  // namespace nlsolver::finite_difference

namespace nlsolver::linesearch {
template <typename scalar_t>
scalar_t max_abs(scalar_t x, scalar_t y, scalar_t z) {
  return std::max(std::abs(x), std::max(std::abs(y), std::abs(z)));
}
// this is in a bit of a sad state since the result gets discarded
// and everything is done by reference - potentially might also
// benefit from branch elimination
template <typename scalar_t>
[[maybe_unused]] static int cstep(scalar_t &stx, scalar_t &fx,
                                  scalar_t &dx,   // NOLINT
                                  scalar_t &sty,  // NOLINT
                                  scalar_t &fy, scalar_t &dy,
                                  scalar_t &stp,  // NOLINT
                                  scalar_t &fp,   // NOLINT
                                  scalar_t &dp, bool &brackt,
                                  scalar_t &stpmin,               // NOLINT
                                  scalar_t &stpmax, int &info) {  // NOLINT
  info = 0;
  bool bound;

  // Check the input parameters for errors.
  if ((brackt & ((stp <= std::min<scalar_t>(stx, sty)) ||
                 (stp >= std::max<scalar_t>(stx, sty)))) ||
      (dx * (stp - stx) >= 0.0) || (stpmax < stpmin)) {
    return -1;
  }

  scalar_t sgnd = dp * (dx / fabs(dx));
  scalar_t stpf = 0;
  scalar_t stpc;
  scalar_t stpq;

  if (fp > fx) {
    info = 1;
    bound = true;
    scalar_t theta = 3. * (fx - fp) / (stp - stx) + dx + dp;
    scalar_t s = max_abs(theta, dx, dp);
    scalar_t gamma = s * sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
    if (stp < stx) gamma = -gamma;
    scalar_t p = (gamma - dx) + theta;
    scalar_t q = ((gamma - dx) + gamma) + dp;
    scalar_t r = p / q;
    stpc = stx + r * (stp - stx);
    stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.) * (stp - stx);
    if (fabs(stpc - stx) < fabs(stpq - stx))
      stpf = stpc;
    else
      stpf = stpc + (stpq - stpc) / 2;
    brackt = true;
  } else if (sgnd < 0.0) {
    info = 2;
    bound = false;
    scalar_t theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
    scalar_t s = max_abs(theta, dx, dp);
    scalar_t gamma = s * sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
    if (stp > stx) gamma = -gamma;

    scalar_t p = (gamma - dp) + theta;
    scalar_t q = ((gamma - dp) + gamma) + dx;
    scalar_t r = p / q;
    stpc = stp + r * (stx - stp);
    stpq = stp + (dp / (dp - dx)) * (stx - stp);
    if (fabs(stpc - stp) > fabs(stpq - stp))
      stpf = stpc;
    else
      stpf = stpq;
    brackt = true;
  } else if (fabs(dp) < fabs(dx)) {
    info = 3;
    bound = true;
    scalar_t theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
    scalar_t s = max_abs(theta, dx, dp);
    scalar_t gamma = s * sqrt(std::max<scalar_t>(
                             static_cast<scalar_t>(0.),
                             (theta / s) * (theta / s) - (dx / s) * (dp / s)));
    if (stp > stx) gamma = -gamma;
    scalar_t p = (gamma - dp) + theta;
    scalar_t q = (gamma + (dx - dp)) + gamma;
    scalar_t r = p / q;
    if ((r < 0.0) & (gamma != 0.0)) {
      stpc = stp + r * (stx - stp);
    } else if (stp > stx) {
      stpc = stpmax;
    } else {
      stpc = stpmin;
    }
    stpq = stp + (dp / (dp - dx)) * (stx - stp);
    if (brackt) {
      if (fabs(stp - stpc) < fabs(stp - stpq)) {
        stpf = stpc;
      } else {
        stpf = stpq;
      }
    } else {
      if (fabs(stp - stpc) > fabs(stp - stpq)) {
        stpf = stpc;
      } else {
        stpf = stpq;
      }
    }
  } else {
    info = 4;
    bound = false;
    if (brackt) {
      scalar_t theta = 3 * (fp - fy) / (sty - stp) + dy + dp;
      scalar_t s = max_abs(theta, dy, dp);
      scalar_t gamma =
          s * sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s));
      if (stp > sty) gamma = -gamma;

      scalar_t p = (gamma - dp) + theta;
      scalar_t q = ((gamma - dp) + gamma) + dy;
      scalar_t r = p / q;
      stpc = stp + r * (sty - stp);
      stpf = stpc;
    } else if (stp > stx) {
      stpf = stpmax;
    } else {
      stpf = stpmin;
    }
  }

  if (fp > fx) {
    sty = stp;
    fy = fp;
    dy = dp;
  } else {
    if (sgnd < 0.0) {
      sty = stx;
      fy = fx;
      dy = dx;
    }

    stx = stp;
    fx = fp;
    dx = dp;
  }

  stpf = std::clamp(stpf, stpmin, stpmax);
  stp = stpf;

  if (brackt & bound) {
    if (sty > stx) {
      stp = std::min<scalar_t>(stx + static_cast<scalar_t>(0.66) * (sty - stx),
                               stp);
    } else {
      stp = std::max<scalar_t>(stx + static_cast<scalar_t>(0.66) * (sty - stx),
                               stp);
    }
  }
  return 0;
}

template <typename Callable, typename Grad, typename scalar_t = double>
[[maybe_unused]] static int cvsrch(
    Callable &f, std::vector<scalar_t> &x, scalar_t current_f_value,
    std::vector<scalar_t> &gradient, scalar_t *stp,
    const std::vector<scalar_t> &search_direction,
    std::vector<scalar_t> &linesearch_temp, Grad &g) {
  // we rewrite this from MIN-LAPACK and some MATLAB code
  int info = 0;
  int infoc = 1;
  constexpr scalar_t xtol = 1e-15;
  constexpr scalar_t ftol = 1e-4;
  constexpr scalar_t gtol = 1e-2;
  constexpr scalar_t stpmin = 1e-15;
  constexpr scalar_t stpmax = 1e15;
  constexpr scalar_t xtrapf = 4;
  constexpr int maxfev = 20;
  int nfev = 0;

  scalar_t dginit =
      nlsolver::math::dot(gradient.data(), search_direction.data(), x.size());
  if (dginit >= 0.0) {
    return -1;
  }

  bool brackt = false;
  bool stage1 = true;

  scalar_t finit = current_f_value;
  scalar_t dgtest = ftol * dginit;
  scalar_t width = stpmax - stpmin;
  scalar_t width1 = 2 * width;
  // vector_t wa = x->eval();

  scalar_t stx = 0.0;
  scalar_t fx = finit;
  scalar_t dgx = dginit;
  scalar_t sty = 0.0;
  scalar_t fy = finit;
  scalar_t dgy = dginit;

  scalar_t stmin;
  scalar_t stmax;

  while (true) {
    // Make sure we stay in the interval when setting min/max-step-width.
    if (brackt) {
      stmin = std::min<scalar_t>(stx, sty);
      stmax = std::max<scalar_t>(stx, sty);
    } else {
      stmin = stx;
      stmax = *stp + xtrapf * (*stp - stx);
    }

    // Force the step to be within the bounds stpmax and stpmin.
    *stp = std::clamp(*stp, stpmin, stpmax);

    // Oops, let us return the last reliable values.
    if ((brackt && ((*stp <= stmin) || (*stp >= stmax))) ||
        (nfev >= maxfev - 1) || (infoc == 0) ||
        (brackt && ((stmax - stmin) <= (xtol * stmax)))) {
      *stp = stx;
    }

    // Test new point.
    for (size_t i = 0; i < x.size(); i++) {
      linesearch_temp[i] = x[i] + *stp * search_direction[i];
    }
    current_f_value = f(linesearch_temp);
    g(linesearch_temp, gradient);
    nfev++;
    scalar_t dg =
        nlsolver::math::dot(gradient.data(), search_direction.data(), x.size());
    scalar_t ftest1 = finit + *stp * dgtest;

    // All possible convergence tests.
    if ((brackt & ((*stp <= stmin) | (*stp >= stmax))) | (infoc == 0)) info = 6;
    if ((*stp == stpmax) & (current_f_value <= ftest1) & (dg <= dgtest))
      info = 5;
    if ((*stp == stpmin) & ((current_f_value > ftest1) | (dg >= dgtest)))
      info = 4;
    if (nfev >= maxfev) info = 3;
    if (brackt & (stmax - stmin <= xtol * stmax)) info = 2;
    if ((current_f_value <= ftest1) & (fabs(dg) <= gtol * (-dginit))) info = 1;
    // Terminate when convergence reached.
    if (info != 0) return -1;

    if (stage1 & (current_f_value <= ftest1) &
        (dg >= std::min<scalar_t>(ftol, gtol) * dginit))
      stage1 = false;

    if (stage1 & (current_f_value <= fx) & (current_f_value > ftest1)) {
      scalar_t fm = current_f_value - *stp * dgtest;
      scalar_t fxm = fx - stx * dgtest;
      scalar_t fym = fy - sty * dgtest;
      scalar_t dgm = dg - dgtest;
      scalar_t dgxm = dgx - dgtest;
      scalar_t dgym = dgy - dgtest;

      cstep(stx, fxm, dgxm, sty, fym, dgym, *stp, fm, dgm, brackt, stmin, stmax,
            infoc);

      fx = fxm + stx * dgtest;
      fy = fym + sty * dgtest;
      dgx = dgxm + dgtest;
      dgy = dgym + dgtest;
    } else {
      // This is ugly and some variables should be moved to the class scope.
      cstep(stx, fx, dgx, sty, fy, dgy, *stp, current_f_value, dg, brackt,
            stmin, stmax, infoc);
    }

    if (brackt) {
      if (fabs(sty - stx) >= 0.66 * width1) {
        *stp = stx + 0.5 * (sty - stx);
      }
      width1 = width;
      width = fabs(sty - stx);
    }
  }
  return 0;
}
template <typename Callable, typename scalar_t>
[[nodiscard]] inline scalar_t line_at_alpha(
    Callable &f, std::vector<scalar_t> &linesearch_temp,
    std::vector<scalar_t> &x, const std::vector<scalar_t> &search_direction,
    const scalar_t alpha = 1.0) {
  const size_t x_size = x.size();
  for (size_t i = 0; i < x_size; ++i) {
    linesearch_temp[i] = x[i] + alpha * search_direction[i];
  }
  return f(linesearch_temp);
}
template <typename Callable, typename scalar_t = double>
[[nodiscard]] [[maybe_unused]] scalar_t armijo_search(
    Callable &f, const scalar_t current_f_val, std::vector<scalar_t> &x,
    std::vector<scalar_t> &gradient,
    const std::vector<scalar_t> &search_direction, scalar_t alpha = 1.0) {
  constexpr scalar_t c = 0.2, rho = 0.9;
  const size_t x_size = x.size();
  scalar_t limit = nlsolver::math::dot(gradient.data(), search_direction.data(),
                                       static_cast<int>(x_size)) *
                   c;

  std::vector<scalar_t> linesearch_temp(x_size);
  scalar_t search_val = line_at_alpha<Callable, scalar_t>(
      f, linesearch_temp, x, search_direction, alpha);
  while (search_val > (current_f_val + alpha * limit)) {
    alpha *= rho;
    search_val = line_at_alpha<Callable, scalar_t>(f, linesearch_temp, x,
                                                   search_direction, alpha);
  }
  return alpha;
}
// this is just an overload which allows us to pass a temporary
template <typename Callable, typename scalar_t = double>
[[maybe_unused]] scalar_t armijo_search(
    Callable &f, const scalar_t current_f_val, std::vector<scalar_t> &x,
    std::vector<scalar_t> &gradient,
    const std::vector<scalar_t> &search_direction,
    std::vector<scalar_t> &linesearch_temp, scalar_t alpha = 1.0) {
  constexpr scalar_t c = 0.2, rho = 0.9;
  const size_t x_size = x.size();
  scalar_t limit = nlsolver::math::dot(gradient.data(), search_direction.data(),
                                       static_cast<int>(x_size)) *
                   c;
  scalar_t search_val =
      line_at_alpha(f, linesearch_temp, x, search_direction, alpha);
  while (search_val > (current_f_val + alpha * limit)) {
    alpha *= rho;
    search_val = line_at_alpha(f, linesearch_temp, x, search_direction, alpha);
  }
  return alpha;
}
// this is just an overload which allows us to pass a temporary
template <typename Callable, typename scalar_t = double>
[[maybe_unused]] scalar_t armijo_search(
    Callable &f, std::vector<scalar_t> &x, std::vector<scalar_t> &gradient,
    const std::vector<scalar_t> &search_direction,
    std::vector<scalar_t> &linesearch_temp, scalar_t alpha = 1.0) {
  constexpr scalar_t c = 0.2, rho = 0.9;
  const scalar_t current_f_val = f(x);
  const size_t x_size = x.size();
  scalar_t limit = nlsolver::math::dot(gradient.data(), search_direction.data(),
                                       static_cast<int>(x_size)) *
                   c;
  scalar_t search_val =
      line_at_alpha(f, linesearch_temp, x, search_direction, alpha);
  while (search_val > (current_f_val + alpha * limit)) {
    alpha *= rho;
    search_val = line_at_alpha(f, linesearch_temp, x, search_direction, alpha);
  }
  return alpha;
}

//
template <typename Callable, typename Grad, typename scalar_t = double>
scalar_t more_thuente_search(Callable &f, const scalar_t current_f_val,
                             std::vector<scalar_t> &x,
                             std::vector<scalar_t> &gradient,
                             const std::vector<scalar_t> &search_direction,
                             std::vector<scalar_t> &linesearch_temp,
                             scalar_t alpha, Grad g) {
  scalar_t alpha_ = alpha;
  cvsrch(f, x, current_f_val, gradient, &alpha_, search_direction,
         linesearch_temp, g);
  return alpha_;
}
//
template <typename Callable, typename Grad, typename scalar_t = double>
[[maybe_unused]] scalar_t more_thuente_search(
    Callable &f, std::vector<scalar_t> &x, std::vector<scalar_t> &gradient,
    const std::vector<scalar_t> &search_direction,
    std::vector<scalar_t> &linesearch_temp, scalar_t alpha, Grad g) {
  const scalar_t current_f_val = f(x);
  scalar_t alpha_ = alpha;
  cvsrch(f, x, current_f_val, gradient, &alpha_, search_direction,
         linesearch_temp, g);
  return alpha_;
}
}  // namespace nlsolver::linesearch

namespace nlsolver {
template <typename scalar_t = double>
inline scalar_t max_abs_vec(const std::vector<scalar_t> &x) {
  auto result = std::abs(x[0]);
  for (size_t i = 1; i < x.size(); i++) {
    scalar_t temp = std::abs(x[i]);
    if (result < temp) {
      result = temp;
    }
  }
  return result;
}

template <typename scalar_t = double>
struct simplex {
  explicit simplex<scalar_t>(const size_t i = 0) {
    this->vals = std::vector<std::vector<scalar_t>>(i + 1);
  }
  explicit simplex<scalar_t>(const std::vector<scalar_t> &x,
                             const scalar_t step = -1) {
    std::vector<std::vector<scalar_t>> init_simplex(x.size() + 1);
    // init_simplex[0] = x;
    //  this follows Gao and Han, see:
    //  'Proper initialization is crucial for the Nelder–Mead simplex search.'
    //  (2019), Wessing, S.  Optimization Letters 13, p. 847–856
    //  (also at https://link.springer.com/article/10.1007/s11590-018-1284-4)
    //  default initialization
    if (step < 0) {
      // get infinity norm of initial vector
      scalar_t x_inf_norm = max_abs_vec(x);
      // if smaller than 1, set to 1
      scalar_t a = x_inf_norm < 1.0 ? 1.0 : x_inf_norm;
      // if larger than 10, set to 10
      scalar_t scale = a < 10 ? a : 10;
      for (auto &vertex : init_simplex) {
        vertex = x;
      }
      for (size_t i = 1; i < init_simplex.size(); i++) {
        init_simplex[i][i] += scale;
      }
      // update first simplex point
      auto n = static_cast<scalar_t>(x.size());
      for (size_t i = 0; i < x.size(); i++) {
        init_simplex[0][i] = x[i] + ((1.0 - sqrt(n + 1.0)) / n * scale);
      }
      // otherwise, first element of simplex has unchanged starting values
    } else {
      for (auto &vertex : init_simplex) {
        vertex = x;
      }
      for (size_t i = 1; i < init_simplex.size(); i++) {
        init_simplex[i][i] += step;
      }
    }
    this->vals = init_simplex;
  }
  void replace(std::vector<scalar_t> &new_val, const size_t at) {
    this->vals[at] = new_val;
  }
  [[maybe_unused]] void replace(std::vector<scalar_t> &new_val, const size_t at,
                                const std::vector<scalar_t> &upper,
                                const std::vector<scalar_t> &lower,
                                const scalar_t inversion_eps = 0.00001) {
    for (size_t i = 0; i < new_val.size(); i++) {
      this->vals[at][i] = new_val[i] < lower[i]   ? lower[i] + inversion_eps
                          : new_val[i] > upper[i] ? upper[i] - inversion_eps
                                                  : new_val[i];
    }
  }
  [[nodiscard]] size_t size() const { return this->vals.size(); }
  std::vector<std::vector<scalar_t>> vals;
};

template <typename scalar_t = double>
inline void update_centroid(std::vector<scalar_t> &centroid,
                            const simplex<scalar_t> &x, const size_t except) {
  // reset centroid - fill with 0
  std::fill(centroid.begin(), centroid.end(), 0.0);
  size_t i = 0;
  for (; i < except; i++) {
    // TODO(JSzitas): SIMD Candidate
    for (size_t j = 0; j < centroid.size(); j++) {
      centroid[j] += x.vals[i][j];
    }
  }
  i = except + 1;
  for (; i < x.size(); i++) {
    for (size_t j = 0; j < centroid.size(); j++) {
      centroid[j] += x.vals[i][j];
    }
  }
  for (auto &val : centroid) val /= static_cast<scalar_t>(i - 1);
}
// bound version
template <typename scalar_t = double, const bool reflect = false,
          const bool bound = false>
inline void simplex_transform(const std::vector<scalar_t> &point,
                              const std::vector<scalar_t> &centroid,
                              std::vector<scalar_t> &result,
                              const scalar_t coef,
                              const std::vector<scalar_t> &upper,
                              const std::vector<scalar_t> &lower) {
  for (size_t i = 0; i < point.size(); i++) {
    scalar_t temp = 0.0;
    // TODO(JSzitas): SIMD Candidate
    if constexpr (reflect) {
      temp = centroid[i] + coef * (centroid[i] - point[i]);
    } else {
      temp = centroid[i] + coef * (point[i] - centroid[i]);
    }
    if constexpr (bound) {
      temp = std::clamp(temp, lower[i], upper[i]);
    }
    result[i] = temp;
  }
}

template <typename scalar_t = double>
inline void shrink(simplex<scalar_t> &current_simplex, const size_t best,
                   const scalar_t sigma) {
  // take a reference to the best vector
  std::vector<scalar_t> &best_val = current_simplex.vals[best];
  const size_t n = best_val.size();
  for (size_t i = 0; i < best; i++) {
    // update all items in current vector using the best vector -
    // hopefully the contiguous data here can help a bit with cache
    // locality
    for (size_t j = 0; j < n; j++) {
      // TODO(JSzitas): SIMD Candidate
      current_simplex.vals[i][j] =
          best_val[j] + sigma * (current_simplex.vals[i][j] - best_val[j]);
    }
  }
  // skip the best point - this uses separate loops, so we do not have to do
  // extra work (e.g. check i == best) which could lead to a branch
  // mis-prediction
  for (size_t i = best + 1; i < current_simplex.size(); i++) {
    for (size_t j = 0; j < n; j++) {
      // TODO(JSzitas): SIMD Candidate
      current_simplex.vals[i][j] =
          best_val[j] + sigma * (current_simplex.vals[i][j] - best_val[j]);
    }
  }
}

template <typename scalar_t = double>
static inline scalar_t std_err(const std::vector<scalar_t> &x) {
  size_t i = 0;
  scalar_t mean_val = 0, result = 0;
  // TODO(JSzitas): SIMD Candidate
  for (; i < x.size(); i++) {
    mean_val += x[i];
  }
  mean_val /= static_cast<scalar_t>(i);
  i = 0;
  for (; i < x.size(); i++) {
    result += pow(x[i] - mean_val, 2);
  }
  result /= static_cast<scalar_t>(i - 1);
  return sqrt(result);
}

template <typename scalar_t = double>
struct solver_status {
  solver_status<scalar_t>(const scalar_t f_val, const size_t iter_used,
                          const size_t f_calls_used,
                          const size_t grad_evals_used = 0ul,
                          const size_t hess_evals_used = 0ul)
      : f_value(f_val),
        iteration(iter_used),
        function_calls_used(f_calls_used),
        gradient_evals_used(grad_evals_used),
        hessian_evals_used(hess_evals_used) {}
  void print() const {
    std::cout << "Function calls used: " << this->function_calls_used
              << std::endl;
    std::cout << "Algorithm iterations used: " << this->iteration << std::endl;
    if (gradient_evals_used > 0) {
      std::cout << "Gradient evaluations used: " << this->gradient_evals_used
                << std::endl;
    }
    if (hessian_evals_used > 0) {
      std::cout << "Hessian evaluations used: " << this->hessian_evals_used
                << std::endl;
    }
    std::cout << "With final function value of " << this->f_value << std::endl;
  }
  std::tuple<size_t, size_t, scalar_t, size_t, size_t> get_summary() const {
    return std::make_tuple(this->function_calls_used, this->iteration,
                           this->f_value, this->gradient_evals_used,
                           this->hessian_evals_used);
  }
  void add(const solver_status<scalar_t> &additional_runs) {
    auto other = additional_runs.get_summary();
    this->function_calls_used += std::get<0>(other);
    this->iteration += std::get<1>(other);
    this->f_value = std::get<2>(other);
    this->gradient_evals_used += std::get<3>(other);
    this->hessian_evals_used += std::get<4>(other);
  }

 private:
  scalar_t f_value;
  size_t iteration, function_calls_used, gradient_evals_used,
      hessian_evals_used;
};

template <typename Callable, typename scalar_t = double>
class NelderMead {
 private:
  Callable &f;
  const scalar_t step, alpha, gamma, rho, sigma;
  scalar_t eps;
  std::vector<scalar_t> point_values;
  const size_t max_iter, no_change_best_tol, restarts;

 public:
  // constructor
  explicit NelderMead<Callable, scalar_t>(
      Callable &f, const scalar_t step = -1, const scalar_t alpha = 1,
      const scalar_t gamma = 2, const scalar_t rho = 0.5,
      const scalar_t sigma = 0.5, const scalar_t eps = 1e-6,
      const size_t max_iter = 500, const size_t no_change_best_tol = 20,
      const size_t restarts = 0)
      : f(f),
        step(step),
        alpha(alpha),
        gamma(gamma),
        rho(rho),
        sigma(sigma),
        eps(eps),
        max_iter(max_iter),
        no_change_best_tol(no_change_best_tol),
        restarts(restarts) {}
  // minimize interface
  [[maybe_unused]] solver_status<scalar_t> minimize(std::vector<scalar_t> &x) {
    std::vector<scalar_t> upper, lower;
    auto res = this->solve<true, false>(x, upper, lower);
    for (size_t i = 0; i < this->restarts; i++) {
      res.add(this->solve<true, false>(x, upper, lower));
    }
    return res;
  }
  // minimize with known bounds interface
  [[maybe_unused]] solver_status<scalar_t> minimize(
      std::vector<scalar_t> &x, const std::vector<scalar_t> &upper,
      const std::vector<scalar_t> &lower) {
    auto res = this->solve<true, true>(x, upper, lower);
    for (size_t i = 0; i < this->restarts; i++) {
      res.add(this->solve<true, true>(x, upper, lower));
    }
    return res;
  }
  // maximize interface
  [[maybe_unused]] solver_status<scalar_t> maximize(std::vector<scalar_t> &x) {
    std::vector<scalar_t> upper, lower;
    auto res = this->solve<false, false>(x, upper, lower);
    for (size_t i = 0; i < this->restarts; i++) {
      res.add(this->solve<false, false>(x, upper, lower));
    }
    return res;
  }
  // maximize with known bounds interface
  [[maybe_unused]] solver_status<scalar_t> maximize(
      std::vector<scalar_t> &x, const std::vector<scalar_t> &upper,
      const std::vector<scalar_t> &lower) {
    auto res = this->solve<false, true>(x, upper, lower);
    for (size_t i = 0; i < this->restarts; i++) {
      res.add(this->solve<false, true>(x, upper, lower));
    }
    return res;
  }

 private:
  template <const bool minimize = true, const bool bound = false>
  solver_status<scalar_t> solve(std::vector<scalar_t> &x,
                                const std::vector<scalar_t> &upper,
                                const std::vector<scalar_t> &lower) {
    // set up simplex
    simplex<scalar_t> current_simplex(x, this->step);
    std::vector<scalar_t> scores(current_simplex.size());
    /* this basically ensures that for minimization we are seeking
     * minimum of function **f**, and for maximization we are seeking minimum of
     * **-f** */
    size_t function_calls_used = 0;
    auto f_lam = [&](decltype(x) &x_) {
      constexpr scalar_t f_multiplier = minimize ? 1.0 : -1.0;
      auto val = f_multiplier * f(x_);
      function_calls_used++;
      return val;
    };
    // score simplex values
    for (size_t i = 0; i < current_simplex.size(); i++) {
      scores[i] = f_lam(current_simplex.vals[i]);
    }
    // set relative convergence tolerance using evaluation with initial
    // parameters
    this->eps = eps * (scores[0] * eps);
    // find best and worst score
    size_t best, worst = 0, second_worst = 0, prev_worst = 0,
                 last_best = 99999999, no_change_iter = 0;

    size_t iter = 0;
    std::vector<scalar_t> centroid(x.size()), temp_reflect(x.size()),
        temp_expand(x.size()), temp_contract(x.size());
    scalar_t ref_score, exp_score, cont_score, fun_std_err;
    bool we_called_shrink = false;
    // simplex iteration
    while (true) {
      // find best, worst and second worst scores
      best = 0;
      prev_worst = worst;
      worst = 0;  // check if worst changed rather than directly overwriting;
      // if it did not, we do not need to recompute the centroid unless we
      // shrunk
      second_worst = 0;
      fun_std_err = std_err(scores);
      for (size_t i = 1; i < scores.size(); i++) {
        // if function value is lower than the current smallest,
        // this is the new best point
        if (scores[i] < scores[best]) {
          best = i;
          // otherwise if it's worse than the current worst, we know it is the
          // new worst point - and the old worst point becomes the new second
          // worst
        } else if (scores[i] > scores[worst]) {
          second_worst = worst;
          worst = i;
        }
      }
      // check if we changed the best value
      if (last_best == best) {
        // if not, increment counter for last value change
        no_change_iter++;
      } else {
        // otherwise reset counter and reassign last_change
        no_change_iter = 0;
        last_best = best;
      }
      // check whether we should stop - either by exceeding iterations or by
      // reaching tolerance
      if (iter >= this->max_iter || fun_std_err < this->eps ||
          no_change_iter >= this->no_change_best_tol) {
        x = current_simplex.vals[best];
        return solver_status<scalar_t>(scores[best], iter, function_calls_used);
      }
      iter++;
      // compute centroid of all points except for the worst one
      if (prev_worst != worst || we_called_shrink) {
        update_centroid(centroid, current_simplex, worst);
        we_called_shrink = false;
      }
      // reflect worst point
      simplex_transform<scalar_t, true, bound>(current_simplex.vals[worst],
                                               centroid, temp_reflect,
                                               this->alpha, upper, lower);
      // score reflected point
      ref_score = f_lam(temp_reflect);
      // if reflected point is better than second worst, not better than best
      if (ref_score >= scores[best] && ref_score < scores[second_worst]) {
        current_simplex.replace(temp_reflect, worst);
        scores[worst] = ref_score;
        // otherwise if this is the best score so far, expand
      } else if (ref_score < scores[best]) {
        simplex_transform<scalar_t, false, bound>(
            temp_reflect, centroid, temp_expand, this->gamma, upper, lower);
        // obtain score for expanded point
        exp_score = f_lam(temp_expand);
        // if this is better than the expanded point score, replace worst point
        // with the expanded point, otherwise replace it with reflected point
        current_simplex.replace(
            exp_score < ref_score ? temp_expand : temp_reflect, worst);
        scores[worst] = exp_score < ref_score ? exp_score : ref_score;
        // otherwise we have a point  worse than the 'second worst'
      } else {
        // contract outside
        simplex_transform<scalar_t, true, bound>(
            ref_score < scores[worst]
                ? temp_reflect
                :
                // or point is the worst point so far - contract inside
                current_simplex.vals[worst],
            centroid, temp_contract, this->rho, upper, lower);
        cont_score = f_lam(temp_contract);
        // if this contraction is better than the reflected point or worst
        if (cont_score <
            (ref_score < scores[worst] ? ref_score : scores[worst])) {
          // replace worst point with contracted point
          current_simplex.replace(temp_contract, worst);
          scores[worst] = cont_score;
          // otherwise shrink
        } else {
          // if we had not violated the bounds before shrinking, shrinking
          // will not cause new violations - hence no bounds applied here
          shrink(current_simplex, best, this->sigma);
          // only in this case do we have to score again
          for (size_t i = 0; i < best; i++) {
            scores[i] = f_lam(current_simplex.vals[i]);
          }
          // we have not updated the best value - hence no need to 'rescore'
          for (size_t i = best + 1; i < current_simplex.size(); i++) {
            scores[i] = f_lam(current_simplex.vals[i]);
          }
          we_called_shrink = true;
        }
      }
    }
  }
};

template <typename RNG, typename scalar_t = double>
static inline std::vector<scalar_t> generate_sequence(
    const std::vector<scalar_t> &offset, RNG &generator) {
  const size_t samples = offset.size();
  std::vector<scalar_t> result(samples);
  for (size_t i = 0; i < samples; i++) {
    // the -0.5 achieves centering around offset
    result[i] = (generator() - 0.5) * offset[i];
  }
  return result;
}

template <typename RNG, typename scalar_t = double>
static inline std::vector<std::vector<scalar_t>> init_agents(
    const std::vector<scalar_t> &init, RNG &generator, const size_t n_agents) {
  std::vector<std::vector<scalar_t>> agents(n_agents);
  // first element of simplex is unchanged starting values
  for (auto &agent : agents) {
    agent = generate_sequence(init, generator);
  }
  return agents;
}
// static inline
template <typename RNG>
size_t generate_index(const size_t max, RNG &generator) {
  // a slightly evil typecast
  return static_cast<size_t>(generator() * max);
}

template <typename RNG>
static inline std::array<size_t, 4> generate_indices(const size_t fixed,
                                                     const size_t max,
                                                     RNG &generator) {
  // prepare set for uniqueness checks
  std::unordered_set<size_t> used_set = {};
  // fixed is the reference agent - hence should already be in the set
  used_set.insert(fixed);
  // set result array
  std::array<size_t, 4> result;  // NOLINT
  result[0] = fixed;
  size_t proposal;
  size_t samples = 1;
  while (true) {
    proposal = generate_index(max, generator);
    if (!used_set.count(proposal)) {
      result[samples] = proposal;
      samples++;
      if (samples == 4) {
        return result;
      }
      used_set.insert(proposal);
    }
  }
}

template <typename RNG, typename scalar_t = double>
static inline void propose_new_agent(
    const std::array<size_t, 4> &ids, std::vector<scalar_t> &proposal,
    const std::vector<std::vector<scalar_t>> &agents,
    const scalar_t crossover_probability, const scalar_t diff_weight,
    RNG &generator) {
  // pick dimensionality to always change
  size_t dim = generate_index(proposal.size(), generator);
  for (size_t i = 0; i < proposal.size(); i++) {
    // check if we mutate
    if (generator() < crossover_probability || i == dim) {
      proposal[i] = agents[ids[1]][i] +
                    diff_weight * (agents[ids[2]][i] - agents[ids[3]][i]);
    } else {
      // no replacement
      proposal[i] = agents[ids[0]][i];
    }
  }
}

enum RecombinationStrategy { best, random };

template <typename Callable, typename RNG, typename scalar_t = double,
          RecombinationStrategy RecombinationType = random>
class DE {
 private:
  Callable &f;
  RNG &generator;
  const scalar_t crossover_prob, differential_weight, eps;
  const size_t pop_size, max_iter, best_value_no_change;

 public:
  // constructor
  [[maybe_unused]] DE<Callable, RNG, scalar_t, RecombinationType>(
      Callable &f, RNG &generator, const scalar_t crossover_prob = 0.9,
      const scalar_t differential_weight = 0.8, const scalar_t eps = 10e-4,
      const size_t pop_size = 50, const size_t max_iter = 1000,
      const size_t best_val_no_change = 50)
      : f(f),
        generator(generator),
        crossover_prob(crossover_prob),
        differential_weight(differential_weight),
        eps(eps),
        pop_size(pop_size),
        max_iter(max_iter),
        best_value_no_change(best_val_no_change) {}
  // minimize interface
  [[maybe_unused]] solver_status<scalar_t> minimize(std::vector<scalar_t> &x) {
    return this->solve<true>(x);
  }
  // maximize interface
  [[maybe_unused]] solver_status<scalar_t> maximize(std::vector<scalar_t> &x) {
    return this->solve<false>(x);
  }

 private:
  template <const bool minimize = true>
  solver_status<scalar_t> solve(std::vector<scalar_t> &x) {
    std::vector<std::vector<scalar_t>> agents =
        init_agents(x, this->generator, this->pop_size);
    std::array<size_t, 4> new_indices = {0, 0, 0, 0};
    constexpr scalar_t f_multiplier = minimize ? 1.0 : -1.0;
    std::vector<scalar_t> proposal_temp(x.size());

    std::vector<scalar_t> scores(agents.size());
    // evaluate all randomly generated agents
    for (size_t i = 0; i < agents.size(); i++) {
      scores[i] = f_multiplier * this->f(agents[i]);
    }
    size_t function_calls_used = agents.size();
    size_t iter = 0;
    size_t best_id = 0, val_no_change = 0;
    while (true) {
      bool not_updated = true;
      // track how good the solutions are
      for (size_t i = 0; i < scores.size(); i++) {
        if (scores[i] < scores[best_id]) {
          best_id = i;
          not_updated = false;
        }
      }
      // if we did not update the best function value, increment counter
      val_no_change = not_updated * (val_no_change + 1);
      // if agents have stabilized, return
      if (iter >= this->max_iter ||
          val_no_change >= this->best_value_no_change ||
          std_err(scores) < this->eps) {
        x = agents[best_id];
        return solver_status<scalar_t>(scores[best_id], iter,
                                       function_calls_used);
      }
      // main loop - this can in principle be parallelized
      for (size_t i = 0; i < agents.size(); i++) {
        // generate agent indices - either using the best or the current agent
        if constexpr (RecombinationType == random) {
          new_indices = generate_indices(i, agents.size(), this->generator);
        }
        if constexpr (RecombinationType == best) {
          new_indices =
              generate_indices(best_id, agents.size(), this->generator);
        }
        // create new mutate proposal
        propose_new_agent(new_indices, proposal_temp, agents,
                          this->crossover_prob, this->differential_weight,
                          this->generator);
        // evaluate proposal
        const scalar_t score = f_multiplier * f(proposal_temp);
        function_calls_used++;
        // if score is better than previous score, update agent
        if (score < scores[i]) {
          for (size_t j = 0; j < proposal_temp.size(); j++) {
            agents[i][j] = proposal_temp[j];
          }
          scores[i] = score;
        }
      }
      // increment iteration counter
      iter++;
    }
  }
};

template <typename scalar_t, typename RNG>
static inline scalar_t rnorm(RNG &generator) {
  // this is not a particularly good generator, but it is 'good enough' for
  // our purposes.
  constexpr scalar_t pi_ = 3.141593;
  return sqrt(-2 * log(generator())) * cos(2 * pi_ * generator());
}

template <typename scalar_t, typename RNG, typename RandomIt>
static inline void rnorm(RNG &generator, RandomIt first, RandomIt last) {
  // this is not a particularly good generator, but it is 'good enough' for
  // our purposes.
  for (; first != last; ++first) {
    (*first) = rnorm<scalar_t, RNG>(generator);
  }
}

enum PSOType { Vanilla, Accelerated };

template <typename Callable, typename RNG, typename scalar_t = double,
          PSOType Type = Vanilla>
class PSO {
 private:
  // user supplied
  RNG &generator;
  Callable &f;
  scalar_t init_inertia, inertia;
  const scalar_t cognitive_coef, social_coef;
  std::vector<scalar_t> lower_, upper_;
  // static, derived from above
  size_t n_dim;
  // internally created
  std::vector<std::vector<scalar_t>> particle_positions, particle_velocities,
      particle_best_positions;
  std::vector<scalar_t> particle_best_values, swarm_best_position;
  scalar_t swarm_best_value;
  // bookkeeping
  size_t val_no_change, f_evals;
  // static limits
  const size_t n_particles, max_iter, best_val_no_change;
  const scalar_t eps;

 public:
  [[maybe_unused]] PSO<Callable, RNG, scalar_t, Type>(
      Callable &f, RNG &generator, const scalar_t inertia = 0.8,
      const scalar_t cognitive_coef = 1.8, const scalar_t social_coef = 1.8,
      const size_t n_particles = 10, const size_t max_iter = 5000,
      const size_t best_val_no_change = 50, const scalar_t eps = 10e-4)
      : generator(generator),
        f(f),
        inertia(inertia),
        cognitive_coef(cognitive_coef),
        social_coef(social_coef),
        n_dim(0),
        val_no_change(0),
        f_evals(0),
        n_particles(n_particles),
        max_iter(max_iter),
        best_val_no_change(best_val_no_change),
        eps(eps) {
    this->particle_positions =
        std::vector<std::vector<scalar_t>>(this->n_particles);
    if constexpr (Type == Vanilla) {
      this->particle_velocities =
          std::vector<std::vector<scalar_t>>(this->n_particles);
    }
    if constexpr (Type == Accelerated) {
      // keep track of original inertia
      this->init_inertia = inertia;
    }
    this->particle_best_positions =
        std::vector<std::vector<scalar_t>>(this->n_particles);
  }
  // minimize interface
  [[maybe_unused]] solver_status<scalar_t> minimize(std::vector<scalar_t> &x) {
    std::vector<scalar_t> lower(x.size());
    std::vector<scalar_t> upper(x.size());
    for (size_t i = 0; i < x.size(); i++) {
      scalar_t temp = std::abs(x[i]);
      lower[i] = -temp;
      upper[i] = temp;
    }
    this->init_solver_state(lower, upper);
    return this->solve<true, false>(x);
  }
  // maximize helper
  [[maybe_unused]] solver_status<scalar_t> maximize(std::vector<scalar_t> &x) {
    std::vector<scalar_t> lower(x.size());
    std::vector<scalar_t> upper(x.size());
    for (size_t i = 0; i < x.size(); i++) {
      scalar_t temp = std::abs(x[i]);
      lower[i] = -temp;
      upper[i] = temp;
    }
    this->init_solver_state(lower, upper);
    return this->solve<false, false>(x);
  }
  // minimize interface
  [[maybe_unused]] solver_status<scalar_t> minimize(
      std::vector<scalar_t> &x, const std::vector<scalar_t> &lower,
      const std::vector<scalar_t> &upper) {
    this->init_solver_state(lower, upper);
    return this->solve<true, true>(x);
  }
  // maximize helper
  [[maybe_unused]] solver_status<scalar_t> maximize(
      std::vector<scalar_t> &x, const std::vector<scalar_t> &lower,
      const std::vector<scalar_t> &upper) {
    this->init_solver_state(lower, upper);
    return this->solve<false, true>(x);
  }

 private:
  template <const bool minimize = true, const bool constrained = true>
  solver_status<scalar_t> solve(std::vector<scalar_t> &x) {
    size_t iter = 0;
    this->update_best_positions<minimize>();
    while (true) {
      // if particles have stabilized (no improvement in objective iteration or
      // no heterogeneity of particles) or we are over the limit, return
      if (iter >= this->max_iter || val_no_change >= best_val_no_change ||
          std_err(this->particle_best_values) < this->eps) {
        x = this->swarm_best_position;
        // best scores, iteration number and function calls used total
        return solver_status<scalar_t>(this->swarm_best_value, iter,
                                       this->f_evals);
      }
      if constexpr (Type == Vanilla) {
        // Vanilla velocity update
        this->update_velocities();
      }
      if constexpr (Type == Accelerated) {
        // update inertia - we might want to create a nicer way to do this
        // updating schedule... maybe a functor for it too?
        this->inertia = pow(this->init_inertia, iter);
        // for accelerated pso update_positions also updated velocities
      }
      this->update_positions();
      if constexpr (constrained) {
        this->threshold_positions();
      }
      this->update_best_positions<minimize>();
      // increment iteration counter
      iter++;
    }
  }
  // for repeated initializations we will initialize solver with new bounds
  void init_solver_state(const std::vector<scalar_t> &lower,
                         const std::vector<scalar_t> &upper) {
    this->n_dim = lower.size();
    this->upper_ = upper;
    this->lower_ = lower;
    this->swarm_best_value = 100000.0;
    this->f_evals = 0;
    this->val_no_change = 0;
    // create particles
    for (size_t i = 0; i < this->n_particles; i++) {
      this->particle_positions[i] = std::vector<scalar_t>(this->n_dim);
      if constexpr (Type == Vanilla) {
        this->particle_velocities[i] = std::vector<scalar_t>(this->n_dim);
      }
      this->particle_best_positions[i] = std::vector<scalar_t>(this->n_dim);
    }
    for (size_t i = 0; i < n_particles; i++) {
      for (size_t j = 0; j < this->n_dim; j++) {
        // update velocities and positions
        scalar_t temp = std::abs(upper[j] - lower[j]);
        this->particle_positions[i][j] =
            lower[j] + ((upper[j] - lower[j]) * this->generator());
        if constexpr (Type == Vanilla) {
          this->particle_velocities[i][j] = -temp + (this->generator() * temp);
        }
        // update particle best positions
        this->particle_best_positions[i][j] = this->particle_positions[i][j];
      }
    }
    this->particle_best_values =
        std::vector<scalar_t>(this->n_particles, 10000);
  }
  void update_velocities() {
    // scalar_t r_p = 0, r_g = 0;
    for (size_t i = 0; i < this->n_particles; i++) {
      for (size_t j = 0; j < this->n_dim; j++) {
        // generate random movements
        const scalar_t r_p = generator(), r_g = generator();
        // update current velocity for current particle - inertia update
        this->particle_velocities[i][j] =
            (this->inertia * this->particle_velocities[i][j]) +
            // cognitive update (moving more if further away from 'best'
            // position of particle )
            this->cognitive_coef * r_p *
                (particle_positions[i][j] - particle_positions[i][j]) +
            // social update (moving more if further away from 'best' position
            // of swarm)
            this->social_coef * r_g *
                (this->swarm_best_position[i] - particle_positions[i][j]);
      }
    }
  }
  void update_positions() {
    if constexpr (Type == Vanilla) {
      for (size_t i = 0; i < this->n_particles; i++) {
        for (size_t j = 0; j < this->n_dim; j++) {
          // update positions using current velocity
          this->particle_positions[i][j] += this->particle_velocities[i][j];
        }
      }
    }
    if constexpr (Type == Accelerated) {
      for (size_t i = 0; i < this->n_particles; i++) {
        for (size_t j = 0; j < this->n_dim; j++) {
          // no need to use velocity - all can be inlined here
          this->particle_positions[i][j] =
              this->inertia * rnorm<scalar_t>(this->generator) +
              // discount position
              (1 - this->cognitive_coef) * this->particle_positions[i][j] +
              // add best position
              this->social_coef * swarm_best_position[j];
        }
      }
    }
  }
  void threshold_positions() {
    for (size_t i = 0; i < this->n_particles; i++) {
      for (size_t j = 0; j < this->n_dim; j++) {
        // threshold velocities between lower and upper
        this->particle_positions[i][j] =
            this->particle_positions[i][j] < this->lower_[j]
                ? this->lower_[j]
                : this->particle_positions[i][j];
        this->particle_positions[i][j] =
            this->particle_positions[i][j] > this->upper_[j]
                ? this->upper_[j]
                : this->particle_positions[i][j];
      }
    }
  }
  template <const bool minimize = true>
  void update_best_positions() {
    size_t best_index = 0;
    constexpr scalar_t f_multiplier = minimize ? 1.0 : -1.0;
    bool update_happened = false;
    for (size_t i = 0; i < this->n_particles; i++) {
      scalar_t temp = f_multiplier * f(particle_positions[i]);
      if (temp < this->swarm_best_value) {
        this->swarm_best_value = temp;
        // save update of swarm best position for after the loop, so we do not
        // by chance do many copies here
        best_index = i;
        update_happened = true;
      }
      if (temp < this->particle_best_values[i]) {
        this->particle_best_values[i] = temp;
      }
    }
    this->f_evals += n_particles;
    if (update_happened) {
      this->swarm_best_position = this->particle_positions[best_index];
    }
    // either increment to indicate no change in the best objective value,
    // or reset to 0
    this->val_no_change = (best_index == 0) * (this->val_no_change + 1);
  }
};

template <typename Callable, typename RNG, typename scalar_t = double>
class SANN {
 private:
  // user supplied
  RNG &generator;
  Callable &f;
  // bookkeeping
  size_t f_evals;
  // static limits
  const size_t max_iter, temperature_iter;
  const scalar_t temperature_max;

 public:
  [[maybe_unused]] SANN<Callable, RNG, scalar_t>(
      Callable &f, RNG &generator, const size_t max_iter = 5000,
      const size_t temperature_iter = 10, const scalar_t temperature_max = 10.0)
      : generator(generator),
        f(f),
        f_evals(0),
        max_iter(max_iter),
        temperature_iter(temperature_iter),
        temperature_max(temperature_max) {}
  // minimize interface
  [[maybe_unused]] solver_status<scalar_t> minimize(std::vector<scalar_t> &x) {
    return this->solve<true>(x);
  }
  // maximize helper
  [[maybe_unused]] solver_status<scalar_t> maximize(std::vector<scalar_t> &x) {
    return this->solve<false>(x);
  }

 private:
  template <const bool minimize = true>
  solver_status<scalar_t> solve(std::vector<scalar_t> &x) {
    constexpr scalar_t f_multiplier = minimize ? 1.0 : -1.0,
                       e_minus_1 = 1.7182818;
    scalar_t best_val = f_multiplier * this->f(x),
             scale = 1.0 / temperature_max;
    this->f_evals++;
    const size_t n_dim = x.size();
    std::vector<scalar_t> p = x, ptry = x;
    size_t iter = 0;
    while (true) {
      if (iter >= this->max_iter) {
        // best scores, iteration number and function calls used total
        return solver_status<scalar_t>(best_val, iter, this->f_evals);
      }
      // temperature annealing schedule - cooling
      const scalar_t t =
          temperature_max / std::log(static_cast<scalar_t>(iter) + e_minus_1);
      for (size_t j = 1; j < this->temperature_iter; j++) {
        const scalar_t current_scale = t * scale;
        // use random normal variates - this should allow user specified values
        for (size_t i = 0; i < n_dim; i++) {
          // generate new candidate function values
          ptry[i] = p[i] + current_scale * rnorm<scalar_t>(this->generator);
        }
        const scalar_t current_val = f_multiplier * f(ptry);
        this->f_evals++;
        const scalar_t difference = current_val - best_val;
        if ((difference <= 0.0) || (this->generator() < exp(-difference / t))) {
          for (size_t k = 0; k < n_dim; k++) p[k] = ptry[k];
          if (current_val <= best_val) {
            for (size_t k = 0; k < n_dim; k++) x[k] = p[k];
            best_val = current_val;
          }
        }
      }
      iter++;
    }
  }
};
enum GradientStepType {
  Linesearch,
  Fixed,
  Bigstep,
  Anneal,
  PAGE
  // Momentum
};

template <const size_t level>
constexpr size_t bigstep_offset() {
  if constexpr (level == 1) return 0;
  if constexpr (level == 2) return 2;
  if constexpr (level == 3) return 5;
  if constexpr (level == 4) return 12;
  if constexpr (level == 5) return 27;
  if constexpr (level == 6) return 58;
  if constexpr (level == 7) return 121;
  return 0;
}

template <const size_t level>
constexpr size_t bigstep_len() {
  if constexpr (level == 1) return 2;
  if constexpr (level == 2) return 3;
  if constexpr (level == 3) return 7;
  if constexpr (level == 4) return 15;
  if constexpr (level == 5) return 31;
  if constexpr (level == 6) return 63;
  if constexpr (level == 7) return 127;
  return 0;
}
template <typename Callable, typename scalar_t>
struct fin_diff {
  void operator()(Callable &f, std::vector<scalar_t> &x,
                  std::vector<scalar_t> &gradient) {
    nlsolver::finite_difference::finite_difference_gradient<Callable, scalar_t,
                                                            1>(f, x, gradient);
  }
};
template <typename Callable, typename scalar_t>
struct fin_diff_h {
  void operator()(Callable &f, std::vector<scalar_t> &x,
                  std::vector<scalar_t> &hessian) {
    nlsolver::finite_difference::finite_difference_hessian<Callable, scalar_t,
                                                           1>(f, x, hessian);
  }
};
template <typename Callable, typename scalar_t,
          const GradientStepType step = GradientStepType::Fixed,
          const size_t bigstep_level = 5,
          const bool grad_norm_lipschitz_scaling = true,
          typename Grad = fin_diff<Callable, scalar_t>>
class GradientDescent {
  Callable &f;
  Grad g;
  const size_t max_iter, minibatch, minibatch_prime;
  const scalar_t grad_eps, alpha;
  nlsolver::rng::xorshift<scalar_t> generator;
  constexpr static std::array<scalar_t, 248> fixed_steps = {
      2.9, 1.5,                                 // pattern length 2 => type 1
      1.5, 4.9,  1.5,                           // type 2
      1.5, 2.2,  1.5,  12.0, 1.5,  2.2,   1.5,  // type 3
      1.4, 2.0,  1.4,  4.5,  1.4,  2.0,   1.4, 29.7, 1.4,  2.0, 1.4, 4.5,   1.4,
      2.0, 1.4,  // type 4
      1.4, 2.0,  1.4,  3.9,  1.4,  2.0,   1.4, 8.2,  1.4,  2.0, 1.4, 3.9,   1.4,
      2.0, 1.4,  72.3, 1.4,  2.0,  1.4,   3.9, 1.4,  2.0,  1.4, 8.2, 1.4,   2.0,
      1.4, 3.9,  1.4,  2.0,  1.4,  // type 5
      1.4, 2.0,  1.4,  3.9,  1.4,  2.0,   1.4, 7.2,  1.4,  2.0, 1.4, 3.9,   1.4,
      2.0, 1.4,  14.2, 1.4,  2.0,  1.4,   3.9, 1.4,  2.0,  1.4, 7.2, 1.4,   2.0,
      1.4, 3.9,  1.4,  2.0,  1.4,  164.0, 1.4, 2.0,  1.4,  3.9, 1.4, 2.0,   1.4,
      7.2, 1.4,  2.0,  1.4,  3.9,  1.4,   2.0, 1.4,  14.2, 1.4, 2.0, 1.4,   3.9,
      1.4, 2.0,  1.4,  7.2,  1.4,  2.0,   1.4, 3.9,  1.4,  2.0, 1.4,  // type 6
      1.4, 2.0,  1.4,  3.9,  1.4,  2.0,   1.4, 7.2,  1.4,  2.0, 1.4, 3.9,   1.4,
      2.0, 1.4,  12.6, 1.4,  2.0,  1.4,   3.9, 1.4,  2.0,  1.4, 7.2, 1.4,   2.0,
      1.4, 3.9,  1.4,  2.0,  1.4,  23.5,  1.4, 2.0,  1.4,  3.9, 1.4, 2.0,   1.4,
      7.2, 1.4,  2.0,  1.4,  3.9,  1.4,   2.0, 1.4,  12.6, 1.4, 2.0, 1.4,   3.9,
      1.4, 2.0,  1.4,  7.2,  1.4,  2.0,   1.4, 3.9,  1.4,  2.0, 1.4, 370.0, 1.4,
      2.0, 1.4,  3.9,  1.4,  2.0,  1.4,   7.2, 1.4,  2.0,  1.4, 3.9, 1.4,   2.0,
      1.4, 12.6, 1.4,  2.0,  1.4,  3.9,   1.4, 2.0,  1.4,  7.2, 1.4, 2.0,   1.4,
      3.9, 1.4,  2.0,  1.4,  23.5, 1.4,   2.0, 1.4,  3.9,  1.4, 2.0, 1.4,   7.5,
      1.4, 2.0,  1.4,  3.9,  1.4,  2.0,   1.4, 12.6, 1.4,  2.0, 1.4, 3.9,   1.4,
      2.0, 1.4,  7.2,  1.4,  2.0,  1.4,   3.9, 1.4,  2.0,  1.4  // type 7
  };
  std::vector<scalar_t> search_direction, linesearch_temp, gradient_temp;

 public:
  explicit GradientDescent<Callable, scalar_t, step, bigstep_level,
                           grad_norm_lipschitz_scaling, Grad>(
      Callable &f, const scalar_t alpha = 1, const size_t max_iter = 500,
      const scalar_t grad_eps = 1e-12, const size_t minibatch_b = 128,
      const size_t minibatch_b_prime = 11,
      Grad g = fin_diff<Callable, scalar_t>())
      : f(f),
        g(g),
        max_iter(max_iter),
        minibatch(minibatch_b),
        minibatch_prime(minibatch_b_prime),
        grad_eps(grad_eps),
        alpha(alpha),
        generator(nlsolver::rng::xorshift<scalar_t>()) {}
  // minimize interface
  [[maybe_unused]] solver_status<scalar_t> minimize(std::vector<scalar_t> &x) {
    return this->solve<true>(x);
  }
  // maximize interface
  [[maybe_unused]] solver_status<scalar_t> maximize(std::vector<scalar_t> &x) {
    return this->solve<false>(x);
  }

 private:
  template <const bool minimize = true>
  solver_status<scalar_t> solve(std::vector<scalar_t> &x) {
    const size_t n_dim = x.size();
    int i_dim = static_cast<int>(n_dim);
    std::vector<scalar_t> gradient = std::vector<scalar_t>(n_dim, 0.0),
                          prev_gradient = std::vector<scalar_t>(n_dim, 0.0);
    if constexpr (step == GradientStepType::Linesearch) {
      // we need additional temporaries for linesearch
      this->search_direction = std::vector<scalar_t>(n_dim, 0.0);
      this->linesearch_temp = std::vector<scalar_t>(n_dim, 0.0);
      this->gradient_temp = std::vector<scalar_t>(n_dim, 0.0);
    }
    scalar_t alpha_ = this->alpha;
    size_t iter = 0, function_calls_used = 0, grad_evals_used = 0;
    constexpr scalar_t f_multiplier = minimize ? -1.0 : 1.0;
    scalar_t max_grad_norm = 0;
    // only necessary and interesting for PAGE
    const scalar_t p = minibatch / (minibatch_prime + minibatch);
    const scalar_t ratio = static_cast<scalar_t>(minibatch) /
                           static_cast<scalar_t>(minibatch_prime);
    // construct lambda that takes f and enables function evaluation counting
    auto f_lam = [&](decltype(x) &coef) {
      function_calls_used++;
      return this->f(coef);
    };
    auto g_lam = [&](decltype(x) &coef, decltype(gradient) &grad) {
      grad_evals_used++;
      // simple optimization - finite difference gradient is actually stateless
      // so in that case this wrapper is entirely valid
      if constexpr (std::is_same<Grad, fin_diff<Callable, scalar_t>>::value) {
        fin_diff<decltype(f_lam), scalar_t>()(f_lam, coef, grad);
        return;
      }
      /* otherwise we cannot keep track of function evaluations that way
       * and our users will have to implement their own gradient function (that
       * might be smarter, actually) - we can however implement our own counter
       * for gradient evaluations
       */
      this->g(this->f, coef, grad);
      return;
    };
    // compute gradient
    g_lam(x, gradient);
    while (true) {
      const scalar_t grad_norm = math::norm(gradient.data(), i_dim);
      max_grad_norm = std::max(max_grad_norm, grad_norm);
      if (iter >= this->max_iter || grad_norm < grad_eps ||
          std::isinf(grad_norm)) {
        // evaluate at current parameters
        scalar_t current_val = f_lam(x);
        return solver_status<scalar_t>(current_val, iter, function_calls_used,
                                       grad_evals_used);
      }
      if constexpr (step == GradientStepType::Linesearch) {
        nlsolver::math::a_mult_scalar_to_b(gradient.data(), f_multiplier,
                                           this->search_direction.data(),
                                           i_dim);
        for (size_t i = 0; i < n_dim; i++) {
          // this->search_direction[i] = f_multiplier * gradient[i];
          this->gradient_temp[i] = gradient[i];
        }
        alpha_ = nlsolver::linesearch::more_thuente_search(
            f_lam, x, this->gradient_temp, this->search_direction,
            this->linesearch_temp, this->alpha, g_lam);
      }
      // do nothing - this is here just to make it more obvious
      if constexpr (step == GradientStepType::Fixed) {
      }
      if constexpr (step == GradientStepType::Anneal) {
        // update alpha using a cooling schedule
        alpha_ = this->alpha / (1.0 + (static_cast<scalar_t>(iter) / max_iter));
      }
      if constexpr (step == GradientStepType::Bigstep) {
        constexpr size_t offset = bigstep_offset<bigstep_level>();
        constexpr size_t step_len = bigstep_len<bigstep_level>();
        const size_t current_step = offset + iter % step_len;
        if constexpr (bigstep_level == 0) {
          alpha_ = ((current_step == 0) * (fixed_steps[current_step] - alpha)) +
                   ((current_step != 0) * fixed_steps[current_step]);
        }
        if constexpr (bigstep_level != 0) {
          alpha_ = fixed_steps[current_step];
        }
        if constexpr (grad_norm_lipschitz_scaling) {
          alpha_ /= max_grad_norm;
        }
      }
      // update parameters
      alpha_ *= f_multiplier;
      nlsolver::math::a_mult_scalar_add_b(gradient.data(), alpha_, x.data(),
                                          i_dim);
      if constexpr (step == GradientStepType::PAGE) {
        for (size_t i = 0; i < n_dim; i++) prev_gradient[i] = gradient[i];
      }
      // compute gradient
      g_lam(x, gradient);
      if constexpr (step == GradientStepType::PAGE) {
        if (generator() > p) {
          // only do a small update where new gradient is old gradient
          // + difference between gradients
          nlsolver::math::a_minus_b_mult_scalar_add_c(
              gradient.data(), prev_gradient.data(), ratio, gradient.data(),
              i_dim);
        }
      }
      iter++;
    }
  }
};

template <typename Callable, typename scalar_t,
          typename Grad = fin_diff<Callable, scalar_t>>
class ConjugatedGradientDescent {
  Callable &f;
  Grad g;
  const size_t max_iter;
  const scalar_t grad_eps, alpha;

 public:
  explicit ConjugatedGradientDescent<Callable, scalar_t, Grad>(
      Callable &f, Grad g = fin_diff<Callable, scalar_t>(),
      const size_t max_iter = 500, const scalar_t grad_eps = 5e-3,
      const scalar_t alpha = 0.03)
      : f(f), g(g), max_iter(max_iter), grad_eps(grad_eps), alpha(alpha) {}
  // minimize interface
  [[maybe_unused]] solver_status<scalar_t> minimize(std::vector<scalar_t> &x) {
    return this->solve<true>(x);
  }
  // maximize interface
  [[maybe_unused]] solver_status<scalar_t> maximize(std::vector<scalar_t> &x) {
    return this->solve<false>(x);
  }

 private:
  template <const bool minimize = true>
  solver_status<scalar_t> solve(std::vector<scalar_t> &x) {
    const size_t n_dim = x.size();
    int i_dim = static_cast<int>(n_dim);
    std::vector<scalar_t> gradient = std::vector<scalar_t>(n_dim, 0.0);
    // we need additional temporaries for linesearch
    std::vector<scalar_t> search_direction = std::vector<scalar_t>(n_dim, 0.0);
    std::vector<scalar_t> linesearch_temp = std::vector<scalar_t>(n_dim, 0.0);
    scalar_t alpha_ = this->alpha;
    size_t iter = 0, function_calls_used = 0, grad_evals_used = 0;
    constexpr scalar_t f_multiplier = minimize ? -1.0 : 1.0;
    // construct lambda that takes f and enables function evaluation counting
    auto f_lam = [&](decltype(x) &coef) {
      function_calls_used++;
      return this->f(coef);
    };
    auto g_lam = [&](decltype(x) &coef, decltype(gradient) &grad) {
      grad_evals_used++;
      // simple optimization - finite difference gradient is actually stateless
      // so in that case this wrapper is entirely valid
      if constexpr (std::is_same<Grad, fin_diff<Callable, scalar_t>>::value) {
        fin_diff<decltype(f_lam), scalar_t>()(f_lam, coef, grad);
        return;
      }
      /* otherwise we cannot keep track of function evaluations that way
       * and our users will have to implement their own gradient function (that
       * might be smarter, actually) - we can however implement our own counter
       * for gradient evaluations
       */
      this->g(this->f, coef, grad);
      return;
    };
    g_lam(x, gradient);
    // set search direction for linesearch
    nlsolver::math::a_mult_scalar_to_b(gradient.data(), f_multiplier,
                                       search_direction.data(), i_dim);
    while (true) {
      // compute gradient
      const scalar_t grad_norm = math::norm(gradient.data(), i_dim);
      if (iter >= this->max_iter || grad_norm < grad_eps ||
          std::isinf(grad_norm)) {
        // evaluate at current parameters
        scalar_t current_val = f_lam(x);
        return solver_status<scalar_t>(current_val, iter, function_calls_used,
                                       grad_evals_used);
      }
      alpha_ = nlsolver::linesearch::armijo_search(
          f_lam, x, gradient, search_direction, linesearch_temp, this->alpha);
      // update parameters; x[i] += search_direction[i] * alpha_;
      nlsolver::math::a_mult_scalar_add_b(search_direction.data(), alpha_,
                                          x.data(), i_dim);
      // recompute gradient, compute new search direction using conjugation
      // first, compute gradient.dot(gradient) with existing gradient,
      // then compute new gradient and compute the same, then compute their
      // ratio
      scalar_t denominator = math::dot(gradient.data(), gradient.data(), i_dim);
      g_lam(x, gradient);
      // figure out the numerator from new gradient
      scalar_t numerator = math::dot(gradient.data(), gradient.data(), i_dim);
      const scalar_t search_update = numerator / denominator;
      // update search direction
      nlsolver::math::a_mul_scalar(search_direction.data(), search_update,
                                   i_dim);
      nlsolver::math::a_mult_scalar_add_b(gradient.data(), f_multiplier,
                                          search_direction.data(), i_dim);
      iter++;
    }
  }
};
template <typename scalar_t>
void update_inverse_hessian(std::vector<scalar_t> &inv_hessian,
                            const std::vector<scalar_t> &step,
                            const std::vector<scalar_t> &grad_diff,
                            std::vector<scalar_t> &grad_diff_inv_hess,
                            const scalar_t rho) {
  // precompute temporaries needed in the hessian update
  const size_t n_dim = grad_diff.size();
  int i_dim = static_cast<int>(n_dim);
  for (size_t i = 0; i < n_dim; i++) {
    grad_diff_inv_hess[i] =
        math::dot(grad_diff.data(), inv_hessian.data() + (i * n_dim), i_dim);
  }
  scalar_t denom =
      math::dot(grad_diff.data(), grad_diff_inv_hess.data(), i_dim);
  denom = (denom * rho) + 1.0;
  // step is                               | n_dim x 1
  // grad_diff_inv_hess is                 |     1 x n_dim
  // => step * grad_diff_inv_hess is a     | n_dim x n_dim
  // inv_hess_grad_diff is                 | n_dim x 1
  // -> inv_hess_grad_diff * step.t() is a | n_dim x n_dim
  for (size_t j = 0; j < n_dim; j++) {
    for (size_t i = 0; i < n_dim; i++) {
      // do not replace this with -= or the whole thing falls apart
      // because of operator order precedence - e.g. whole rhs would
      // get evaluated before -=, whereas we want to do inv_hessian - first part
      // + second part
      inv_hessian[j * n_dim + i] =
          inv_hessian[j * n_dim + i] -
          // first part
          rho * (step[i] * grad_diff_inv_hess[j] +
                 grad_diff_inv_hess[i] * step[j] +
                 // second part ->  multiply(step[i], denom * step[j])
                 denom * step[i] * step[j]);
    }
    // inline intrinsics would be complicated, but the above can probably be
    // reduced to something a lot faster that way
  }
}
template <typename Callable, typename scalar_t = double,
          typename Grad = fin_diff<Callable, scalar_t>>
class BFGS {
 private:
  Callable &f;
  Grad g;
  // stopping
  const size_t max_iter;
  const scalar_t grad_eps, alpha;

 public:
  // constructor
  explicit BFGS<Callable, scalar_t, Grad>(
      Callable &f, Grad g = fin_diff<Callable, scalar_t>(),
      const size_t max_iter = 100, const scalar_t grad_eps = 5e-3,
      const scalar_t alpha = 1)
      : f(f), g(g), max_iter(max_iter), grad_eps(grad_eps), alpha(alpha) {}

  // minimize interface
  [[maybe_unused]] solver_status<scalar_t> minimize(std::vector<scalar_t> &x) {
    return this->solve<true>(x);
  }
  // maximize helper
  [[maybe_unused]] solver_status<scalar_t> maximize(std::vector<scalar_t> &x) {
    return this->solve<false>(x);
  }

 private:
  template <const bool minimize = true>
  solver_status<scalar_t> solve(std::vector<scalar_t> &x) {
    static_assert(minimize, "BFGS currently only supports minimization");
    const size_t n_dim = x.size();
    int i_dim = static_cast<int>(n_dim);
    std::vector<scalar_t> inverse_hessian =
        std::vector<scalar_t>(n_dim * n_dim);
    std::vector<scalar_t> search_direction = std::vector<scalar_t>(n_dim, 0.0),
                          gradient = std::vector<scalar_t>(n_dim, 0.0),
                          prev_gradient = std::vector<scalar_t>(n_dim, 0.0),
                          grad_update = std::vector<scalar_t>(n_dim, 0.0),
                          s = std::vector<scalar_t>(n_dim, 0.0),
                          linesearch_temp = std::vector<scalar_t>(n_dim, 0.0),
                          grad_diff_inv_hess(n_dim), inv_hess_grad_diff(n_dim);
    // initialize to identity matrix
    for (size_t i = 0; i < n_dim; i++) inverse_hessian[i + (i * n_dim)] = 1.0;
    size_t iter = 0, function_calls_used = 0, grad_evals_used = 0;
    auto f_lam = [&](decltype(x) &coef) {
      function_calls_used++;
      return this->f(coef);
    };
    auto g_lam = [&](decltype(x) &coef, decltype(gradient) &grad) {
      grad_evals_used++;
      // simple optimization - finite difference gradient is actually stateless
      // so in that case this wrapper is entirely valid
      if constexpr (std::is_same<Grad, fin_diff<Callable, scalar_t>>::value) {
        fin_diff<decltype(f_lam), scalar_t>()(f_lam, coef, grad);
        return;
      }
      /* otherwise we cannot keep track of function evaluations that way
       * and our users will have to implement their own gradient function (that
       * might be smarter, actually) - we can however implement our own counter
       * for gradient evaluations
       */
      this->g(this->f, coef, grad);
      return;
    };
    g_lam(x, gradient);
    // constexpr scalar_t f_multiplier = minimize ? -1.0 : 1.0;
    scalar_t prev_grad_norm = 1e9;
    scalar_t current_grad_norm = 1e8;
    while (true) {
      if (iter >= this->max_iter || current_grad_norm < grad_eps ||
          std::abs(current_grad_norm - prev_grad_norm) < grad_eps ||
          std::isinf(current_grad_norm)) {
        // evaluate at current parameters
        scalar_t current_val = f_lam(x);
        return solver_status<scalar_t>(current_val, iter, function_calls_used,
                                       grad_evals_used);
      }
      // update search direction vector using -inverse_hessian * gradient
      for (size_t j = 0; j < n_dim; j++) {
        search_direction[j] = -math::dot(inverse_hessian.data() + (j * n_dim),
                                         gradient.data(), i_dim);
      }
      scalar_t phi = math::dot(gradient.data(), search_direction.data(), i_dim);
      if ((phi > 0) || std::isnan(phi) || current_grad_norm > prev_grad_norm) {
        std::fill(inverse_hessian.begin(), inverse_hessian.end(), 0.0);
        // reset hessian approximation and search_direction
        for (size_t i = 0; i < n_dim; i++) {
          inverse_hessian[i + (i * n_dim)] = 1.0;
          search_direction[i] = -gradient[i];
        }
      }
      prev_gradient = gradient;
      const scalar_t rate = nlsolver::linesearch::more_thuente_search(
          f_lam, x, gradient, search_direction, linesearch_temp, this->alpha,
          g_lam);
      // update parameters
      nlsolver::math::a_mult_scalar_to_b(search_direction.data(), rate,
                                         s.data(), i_dim);
      nlsolver::math::a_plus_b(x.data(), s.data(), i_dim);
      // we also need to compute the gradient at this new point
      // update it by reference
      g_lam(x, gradient);
      prev_grad_norm = current_grad_norm;
      current_grad_norm = nlsolver::math::norm(gradient.data(), i_dim);
      // Update grad difference, rho and inverse hessian
      nlsolver::math::a_minus_b_to_c(gradient.data(), prev_gradient.data(),
                                     grad_update.data(), i_dim);
      scalar_t rho = nlsolver::math::dot(grad_update.data(), s.data(), i_dim);
      rho = 1 / rho;
      // update inverse hessian using Sherman Morrisson
      update_inverse_hessian(inverse_hessian, s, grad_update,
                             grad_diff_inv_hess,  // inv_hess_grad_diff,
                             rho);
      iter++;
    }
  }
};
template <typename Callable, typename scalar_t>
class [[maybe_unused]] Brent {
  Callable &f;
  const scalar_t tol, eps;
  const size_t max_iter;

 public:
  explicit Brent(Callable &f, const scalar_t tol = 1e-12,
                 const scalar_t eps = 1e-12, const size_t max_iter = 200)
      : f(f), tol(tol), eps(eps), max_iter(max_iter) {}
  // minimize interface
  [[maybe_unused]] solver_status<scalar_t> minimize(scalar_t &x,
                                                    const scalar_t lower = -5,
                                                    const scalar_t upper = 5) {
    return this->solve<true>(x, upper, lower);
  }
  // maximize helper
  [[maybe_unused]] solver_status<scalar_t> maximize(scalar_t &x,
                                                    const scalar_t lower = -5,
                                                    const scalar_t upper = 5) {
    return this->solve<false>(x, upper, lower);
  }

 private:
  template <const bool minimize = true>
  solver_status<scalar_t> solve(scalar_t &x_, const scalar_t upper,
                                const scalar_t lower) {
    // lightly adapted from R's C level Brent_fmin
    // values
    size_t f_evals_used = 0;
    constexpr scalar_t f_mult = minimize ? 1.0 : -1.0;
    // functor
    auto f_lam = [&](auto x) {
      const auto val = f_mult * this->f(x);
      f_evals_used++;
      return val;
    };
    /*  c is the squared inverse of the golden ratio */
    const scalar_t c = (3. - std::sqrt(5.)) * .5;
    /* Local variables */
    scalar_t a, b, d, e, p, q, r, u, v, w, x, t2, fu, fv, fw, fx, xm, tol1,
        tol3;
    /*  eps is approximately the square root of the relative machine precision.
     */
    tol1 = eps + 1.; /* the smallest 1.000... > 1 */
    a = lower;
    b = upper;
    v = a + c * (b - a);
    w = v;
    x = v;
    d = 0.; /* -Wall */
    e = 0.;
    fx = f_lam(x);
    fv = fx;
    fw = fx;
    tol3 = tol / 3.;
    /*  main loop starts here ----------------------------------- */
    size_t iter = 0;
    for (; iter < max_iter; ++iter) {
      xm = (a + b) * .5;
      tol1 = eps * fabs(x) + tol3;
      t2 = tol1 * 2.;
      /* check stopping criterion */
      if (std::fabs(x - xm) <= t2 - (b - a) * .5) break;
      p = 0.;
      q = 0.;
      r = 0.;
      if (std::fabs(e) > tol1) { /* fit parabola */
        r = (x - w) * (fx - fv);
        q = (x - v) * (fx - fw);
        p = (x - v) * q - (x - w) * r;
        q = (q - r) * 2.;
        if (q > 0.) {
          p = -p;
        } else {
          q = -q;
        }
        r = e;
        e = d;
      }
      if (std::fabs(p) >= std::fabs(q * .5 * r) || p <= q * (a - x) ||
          p >= q * (b - x)) { /* a golden-section step */
        if (x < xm) {
          e = b - x;
        } else {
          e = a - x;
        }
        d = c * e;
      } else { /* a parabolic-interpolation step */
        d = p / q;
        u = x + d;
        /* f must not be evaluated too close to ax or bx */
        if (u - a < t2 || b - u < t2) {
          d = tol1;
          if (x >= xm) {
            d = -d;
          }
        }
      }
      /* f must not be evaluated too close to x */
      if (std::fabs(d) >= tol1) {
        u = x + d;
      } else if (d > 0.) {
        u = x + tol1;
      } else {
        u = x - tol1;
      }
      fu = f_lam(u);
      /*  update  a, b, v, w, and x */
      if (fu <= fx) {
        if (u < x)
          b = x;
        else
          a = x;
        v = w;
        w = x;
        x = u;
        fv = fw;
        fw = fx;
        fx = fu;
      } else {
        if (u < x) {
          a = u;
        } else {
          b = u;
        }
        if (fu <= fw || w == x) {
          v = w;
          fv = fw;
          w = u;
          fw = fu;
        } else if (fu <= fv || v == x || v == w) {
          v = u;
          fv = fu;
        }
      }
    }
    x_ = fx;
    return solver_status<scalar_t>(x, iter, f_evals_used);
  }
};
template <typename Callable, typename scalar_t,
          typename Grad = fin_diff<Callable, scalar_t>,
          typename Hess = fin_diff_h<Callable, scalar_t>>
class [[maybe_unused]] LevenbergMarquardt {
 private:
  Callable &f;
  Grad g;
  Hess h;
  scalar_t lambda;
  const scalar_t upward_mult, downward_mult;
  const size_t max_iter;
  const scalar_t f_delta;

 public:
  // constructor
  explicit LevenbergMarquardt<Callable, scalar_t, Grad, Hess>(
      Callable &f, const scalar_t lambda = 10, const scalar_t upward_mult = 10,
      const scalar_t downward_mult = 10, const size_t max_iter = 100,
      const scalar_t f_delta = 1e-12, Grad g = fin_diff<Callable, scalar_t>(),
      Hess h = fin_diff_h<Callable, scalar_t>())
      : f(f),
        g(g),
        h(h),
        lambda(lambda),
        upward_mult(upward_mult),
        downward_mult(downward_mult),
        max_iter(max_iter),
        f_delta(f_delta) {}
  // minimize interface
  [[maybe_unused]] solver_status<scalar_t> minimize(std::vector<scalar_t> &x) {
    return this->solve<true>(x);
  }
  // maximize helper
  [[maybe_unused]] solver_status<scalar_t> maximize(std::vector<scalar_t> &x) {
    return this->solve<false>(x);
  }

 private:
  template <const bool minimize = true>
  solver_status<scalar_t> solve(std::vector<scalar_t> &x) {
    static_assert(minimize,
                  "LevenbergMarquardt currently only supports minimization");
    const size_t n = x.size();
    size_t iter = 0;
    // scalar_t current_f_value;
    size_t function_calls_used = 0, grad_evals_used = 0, hess_evals_used = 0;
    std::vector<scalar_t> gradient(n, 0.0);
    std::vector<scalar_t> hessian(n * n, 0.0);
    // construct lambda that takes f and enables function evaluation counting
    auto f_lam = [&](decltype(x) &coef) {
      function_calls_used++;
      return this->f(coef);
    };
    auto g_lam = [&](decltype(x) &coef, decltype(gradient) &grad) {
      grad_evals_used++;
      // simple optimization - finite difference gradient is actually stateless
      // so in that case this wrapper is entirely valid
      if constexpr (std::is_same<Grad, fin_diff<Callable, scalar_t>>::value) {
        fin_diff<decltype(f_lam), scalar_t>()(f_lam, coef, grad);
        return;
      }
      /* otherwise we cannot keep track of function evaluations that way
       * and our users will have to implement their own gradient function (that
       * might be smarter, actually) - we can however implement our own counter
       * for gradient evaluations
       */
      this->g(this->f, coef, grad);
      return;
    };
    auto h_lam = [&](decltype(x) &coef, decltype(hessian) &hess_) {
      hess_evals_used++;
      // simple optimization - finite difference gradient is actually stateless
      // so in that case this wrapper is entirely valid
      if constexpr (std::is_same<Hess, fin_diff_h<Callable, scalar_t>>::value) {
        fin_diff_h<decltype(f_lam), scalar_t>()(f_lam, coef, hess_);
        return;
      }
      /* otherwise we cannot keep track of function evaluations that way
       * and our users will have to implement their own gradient function (that
       * might be smarter, actually) - we can however implement our own counter
       * for gradient evaluations
       */
      this->h(this->f, coef, hess_);
      return;
    };
    g_lam(x, gradient);
    h_lam(x, hessian);
    scalar_t previous_f_value = 0.0;
    scalar_t current_f_value = f_lam(x);
    std::vector<scalar_t> update(x.size());
    while (true) {
      // stopping conditions
      const scalar_t current_f_delta =
          std::abs(previous_f_value - current_f_value);
      if (iter >= max_iter || current_f_delta < this->f_delta ||
          std::isnan(previous_f_value)) {
        return solver_status<scalar_t>(current_f_value, iter,
                                       function_calls_used, grad_evals_used,
                                       hess_evals_used);
      }
      // add current lambda to diagonal elements of H, compute H^-1 * G
      for (size_t i = 0; i < n; i++) {
        hessian[i * n + i] += lambda;
      }
      // compute update using gradient and hessian
      nlsolver::math::get_update_with_hessian(update, hessian, gradient);
      for (size_t i = 0; i < x.size(); i++) x[i] -= update[i];
      previous_f_value = current_f_value;
      current_f_value = f_lam(x);
      iter++;
      // update hessian and gradient
      g_lam(x, gradient);
      h_lam(x, hessian);
      lambda = current_f_value < previous_f_value ? lambda / downward_mult
                                                  : lambda * upward_mult;
    }
  }
};
template <typename Callable, typename RNG, typename scalar_t = double>
class NelderMeadPSO {
 private:
  // user supplied
  RNG &generator;
  Callable &f;
  // initialized once
  const scalar_t alpha, gamma, rho, sigma, inertia, cognitive_coef, social_coef;
  scalar_t eps;
  // used during optimization
  std::vector<std::vector<scalar_t>> particle_positions, particle_velocities;
  std::vector<scalar_t> particle_current_values;
  const size_t max_iter, no_change_best_iter;
  size_t function_calls_used = 0;

 public:
  // constructor
  [[maybe_unused]] NelderMeadPSO(
      Callable &f, RNG &generator, const scalar_t alpha = 1,
      const scalar_t gamma = 2, const scalar_t rho = 0.5,
      const scalar_t sigma = 0.5, const scalar_t inertia = 0.8,
      const scalar_t cognitive_coef = 1.8, const scalar_t social_coef = 1.8,
      const scalar_t eps = 1e-6, const size_t max_iter = 1000,
      const size_t no_change_best_iter = 20)
      : generator(generator),
        f(f),
        alpha(alpha),
        gamma(gamma),
        rho(rho),
        sigma(sigma),
        inertia(inertia),
        cognitive_coef(cognitive_coef),
        social_coef(social_coef),
        eps(eps),
        max_iter(max_iter),
        no_change_best_iter(no_change_best_iter) {}
  // minimize interface
  [[maybe_unused]] solver_status<scalar_t> minimize(std::vector<scalar_t> &x) {
    const size_t n_dim = x.size();
    // compute implied upper and lower bounds
    std::vector<scalar_t> lower(n_dim);
    std::vector<scalar_t> upper(n_dim);
    for (size_t i = 0; i < n_dim; i++) {
      scalar_t temp = std::abs(2.5 * x[i]);
      lower[i] = -temp;
      upper[i] = temp;
    }
    return this->solve<true, false>(x, upper, lower);
  }
  // maximize interface
  [[maybe_unused]] solver_status<scalar_t> maximize(std::vector<scalar_t> &x) {
    const size_t n_dim = x.size();
    // compute implied upper and lower bounds
    std::vector<scalar_t> lower(n_dim);
    std::vector<scalar_t> upper(n_dim);
    for (size_t i = 0; i < n_dim; i++) {
      scalar_t temp = std::abs(2.5 * x[i]);
      lower[i] = -temp;
      upper[i] = temp;
    }
    return this->solve<false, false>(x, upper, lower);
  }
  // minimize interface
  [[maybe_unused]] solver_status<scalar_t> minimize(
      std::vector<scalar_t> &x, const std::vector<scalar_t> &lower,
      const std::vector<scalar_t> &upper) {
    return this->solve<true, true>(x, upper, lower);
  }
  // maximize helper
  [[maybe_unused]] solver_status<scalar_t> maximize(
      std::vector<scalar_t> &x, const std::vector<scalar_t> &lower,
      const std::vector<scalar_t> &upper) {
    return this->solve<false, true>(x, upper, lower);
  }

 private:
  template <const bool minimize = true, const bool bound = false>
  solver_status<scalar_t> solve(std::vector<scalar_t> &x,
                                const std::vector<scalar_t> &upper,
                                const std::vector<scalar_t> &lower) {
    const size_t n_dim = x.size();
    if (n_dim < 2) {
      std::cout
          << "You are trying to optimize a one dimensional function "
          << "you should probably be using vanilla NelderMead (or vanilla PSO)"
          << " - our implementation does not support this in the "
             "NelderMead-PSO hybrid."
          << std::endl;
      // return some invalid solver state
      return solver_status<scalar_t>(999999, 0, 0);
    }
    // initialize solver state
    const size_t n_simplex_particles = n_dim + 1, n_pso_particles = 2 * n_dim,
                 n_particles = n_pso_particles + n_simplex_particles;
    init_solver_state<minimize>(x, upper, lower, n_simplex_particles,
                                n_pso_particles, n_dim);
    // temporaries for the simplex
    std::vector<scalar_t> centroid(n_dim), temp_reflect(n_dim),
        temp_expand(n_dim), temp_contract(n_dim);
    // create an index for the current particle order (best to worst)
    std::vector<size_t> current_order(n_particles);
    std::iota(current_order.begin(), current_order.end(), 0);
    size_t iter = 0;
    scalar_t best_val = this->particle_current_values[0];
    size_t best_val_no_change = 0;
    while (true) {
      // sort particles from best to worst
      std::sort(current_order.begin(), current_order.end(),
                [&](size_t left_id, size_t right_id) -> bool {
                  return this->particle_current_values[left_id] <
                         this->particle_current_values[right_id];
                });
      // record if best value was updated
      const bool best_val_stayed_same =
          best_val == this->particle_current_values[current_order[0]];
      // this rolls together - increment by one or reset
      best_val_no_change += best_val_stayed_same;
      best_val_no_change *= best_val_stayed_same;
      // stopping criteria
      if (iter >= this->max_iter ||
          best_val_no_change >= this->no_change_best_iter ||
          // this should be applied only over the simplex particles
          this->simplex_std_err(current_order, n_simplex_particles) <
              this->eps) {
        x = this->particle_positions[current_order[0]];
        // best scores, iteration number and function calls used total
        return solver_status<scalar_t>(
            this->particle_current_values[current_order[0]], iter,
            this->function_calls_used);
      }
      // use top N+1 particles to form simplex and apply the simplex update
      apply_simplex<minimize, bound>(centroid, temp_reflect, temp_expand,
                                     temp_contract, current_order,
                                     n_simplex_particles, n_dim, upper, lower);
      // update the rest of the particles - i.e. the non-simplex ones
      // using the regular PSO update
      apply_pso<minimize, bound>(current_order, n_simplex_particles,
                                 n_particles, n_dim, upper, lower);
      iter++;
    }
  }
  template <const bool minimize = true>
  void init_solver_state(const std::vector<scalar_t> &x,
                         const std::vector<scalar_t> &upper,
                         const std::vector<scalar_t> &lower,
                         const size_t nm_particles, const size_t pso_particles,
                         const size_t n_dim) {
    const size_t n_particles = nm_particles + pso_particles;
    this->particle_positions = std::vector<std::vector<scalar_t>>(n_particles);
    this->particle_velocities = std::vector<std::vector<scalar_t>>(n_particles);
    this->particle_current_values = std::vector<scalar_t>(n_particles);
    this->function_calls_used = 0;
    size_t i = 0;
    for (; i < n_particles; i++) {
      this->particle_positions[i] = std::vector<scalar_t>(n_dim);
      this->particle_velocities[i] = std::vector<scalar_t>(n_dim, 0.0);
    }
    // create particles - first x.size() + 1 particles should be initialized
    // as in NM; this follows Gao and Han, see:
    // 'Proper initialization is crucial for the Nelder–Mead simplex search.'
    // (2019), Wessing, S.  Optimization Letters 13, p. 847–856
    // (also at https://link.springer.com/article/10.1007/s11590-018-1284-4)
    particle_positions[0] = x;
    for (i = 1; i < nm_particles; i++) {
      scalar_t x_inf_norm = max_abs_vec(x);
      // if smaller than 1, set to 1
      scalar_t a = x_inf_norm < 1.0 ? 1.0 : x_inf_norm;
      // if larger than 10, set to 10
      scalar_t scale = a < 10 ? a : 10;
      for (i = 1; i < nm_particles; i++) {
        particle_positions[i] = x;
        particle_positions[i][i] = x[i] + scale;
      }
      // update first simplex point
      auto n = static_cast<scalar_t>(x.size());
      for (i = 0; i < x.size(); i++) {
        particle_positions[0][i] = x[i] + ((1.0 - sqrt(n + 1.0)) / n * scale);
      }
    }
    // the rest according to PSO
    for (i = nm_particles; i < n_particles; i++) {
      for (size_t j = 0; j < n_dim; j++) {
        // update velocities and positions
        scalar_t temp = std::abs(upper[j] - lower[j]);
        this->particle_positions[i][j] =
            lower[j] + ((upper[j] - lower[j]) * generator());
        this->particle_velocities[i][j] = -temp + (generator() * temp);
      }
    }
    constexpr scalar_t f_multiplier = minimize ? 1.0 : -1.0;
    for (i = 0; i < n_particles; i++) {
      this->particle_current_values[i] =
          f_multiplier * f(particle_positions[i]);
      this->function_calls_used++;
    }
  }
  template <const bool minimize, const bool bound>
  void apply_simplex(std::vector<scalar_t> &centroid,
                     std::vector<scalar_t> &temp_reflect,
                     std::vector<scalar_t> &temp_expand,
                     std::vector<scalar_t> &temp_contract,
                     std::vector<size_t> &current_order,
                     const size_t nm_particles, const size_t n_dim,
                     const std::vector<scalar_t> &upper,
                     const std::vector<scalar_t> &lower) {
    const scalar_t best_score = this->particle_current_values[current_order[0]];
    const size_t worst_id = current_order[nm_particles - 1];
    const size_t second_worst_id = current_order[nm_particles - 2];
    // update centroid of all points except for the worst one
    this->update_centroid(centroid, current_order, nm_particles - 1);
    // reflect worst point
    simplex_transform<scalar_t, true, bound>(this->particle_positions[worst_id],
                                             centroid, temp_reflect,
                                             this->alpha, upper, lower);
    // set constant multiplier for minimization
    constexpr scalar_t f_multiplier = minimize ? 1.0 : -1.0;
    // score reflected point
    const scalar_t ref_score = f_multiplier * f(temp_reflect);
    this->function_calls_used++;
    // if reflected point is better than second worst, but not better than best
    if (ref_score >= best_score &&
        ref_score < this->particle_current_values[second_worst_id]) {
      this->particle_positions[worst_id] = temp_reflect;
      this->particle_current_values[worst_id] = ref_score;
      // otherwise if this is the best score so far, expand
    } else if (ref_score < best_score) {
      simplex_transform<scalar_t, false, bound>(
          temp_reflect, centroid, temp_expand, this->gamma, upper, lower);
      // obtain score for expanded point
      const scalar_t exp_score = f_multiplier * f(temp_expand);
      this->function_calls_used++;
      // if this is better than the expanded point score, replace the worst
      // point with the expanded point, otherwise replace it with
      // the reflected point
      std::vector<scalar_t> &replacement =
          exp_score < ref_score ? temp_expand : temp_reflect;
      this->particle_positions[worst_id] = replacement;
      this->particle_current_values[worst_id] =
          exp_score < ref_score ? exp_score : ref_score;
      // otherwise we have a point  worse than the 'second worst'
    } else {
      // contract outside - here we overwrite the 'temp_expand' and it
      // functionally becomes 'temp_contract'
      const scalar_t worst_score = this->particle_current_values[worst_id];
      simplex_transform<scalar_t, false, bound>(
          ref_score < worst_score
              ? temp_reflect
              :
              // or point is the worst point so far - contract inside
              this->particle_positions[worst_id],
          centroid, temp_contract, this->rho, upper, lower);
      const scalar_t cont_score = f_multiplier * f(temp_contract);
      this->function_calls_used++;
      // if this contraction is better than the reflected point or worst point
      if (cont_score < std::min(ref_score, worst_score)) {
        // replace worst point with contracted point
        this->particle_positions[worst_id] = temp_contract;
        this->particle_current_values[worst_id] = cont_score;
        // otherwise shrink
      } else {
        // if we had not violated the bounds before shrinking, shrinking
        // will not cause new violations - hence no bounds applied here
        shrink(current_order, this->sigma, nm_particles, n_dim);
        // only in this case do we have to score again
        for (size_t i = 1; i < nm_particles; i++) {
          this->particle_current_values[current_order[i]] =
              f_multiplier * f(this->particle_positions[current_order[i]]);
        }
        this->function_calls_used += nm_particles - 1;
        // re-sort, as PSO step is called after this
        std::sort(current_order.begin(), current_order.end(),
                  [&](size_t left_id, size_t right_id) -> bool {
                    return this->particle_current_values[left_id] <
                           this->particle_current_values[right_id];
                  });
      }
    }
  }
  template <const bool minimize, const bool bound>
  void apply_pso(const std::vector<size_t> &current_order,
                 const size_t n_simplex_particles,
                 const size_t n_total_particles, const size_t n_dim,
                 const std::vector<scalar_t> &upper,
                 const std::vector<scalar_t> &lower) {
    constexpr scalar_t f_multiplier = minimize ? 1.0 : -1.0;
    bool order_flip = false;
    size_t best_in_pair = current_order[n_simplex_particles];
    const std::vector<scalar_t> &best =
        this->particle_positions[current_order[0]];
    for (size_t i = n_simplex_particles; i < n_total_particles; i++) {
      const size_t id = current_order[i];
      if (order_flip) {
        best_in_pair = current_order[i + 1];
      }
      order_flip = static_cast<bool>((i - n_simplex_particles) % 2);
      // get references to current particle, current velocity and pairwise best
      // particle
      std::vector<scalar_t> &particle = particle_positions[id],
                            velocity = this->particle_velocities[id],
                            pairwise_best = particle_positions[best_in_pair];
      for (size_t j = 0; j < n_dim; j++) {
        // generate random movements
        const scalar_t r_p = generator(), r_g = generator();
        // update current velocity for current particle - inertia update
        // TODO(JSzitas): SIMD Candidate
        scalar_t temp =
            (this->inertia * velocity[j]) +
            // cognitive update - this should be based on better
            // particle of each 2 particle pairs
            this->cognitive_coef * r_p * (pairwise_best[j] - particle[j]) +
            // social update (moving more if further away from 'best' position)
            this->social_coef * r_g * (best[j] - particle[j]);
        if constexpr (bound) {
          temp = std::clamp(temp, lower[i], upper[i]);
        }
        velocity[j] = temp;
        particle[j] += temp;
      }
      // rerun function evaluation
      this->particle_current_values[id] = f_multiplier * f(particle);
      this->function_calls_used++;
    }
  }
  void update_centroid(std::vector<scalar_t> &centroid,
                       const std::vector<size_t> &current_order,
                       const size_t last_point) {
    // reset centroid - fill with 0
    std::fill(centroid.begin(), centroid.end(), 0.0);
    // iterate through 0 to last_point - 1 - last point taken to be the Nth
    // best point in an N+1 dimensional simplex
    size_t i = 0;
    for (; i < last_point; i++) {
      const std::vector<scalar_t> &particle =
          this->particle_positions[current_order[i]];
      // TODO(JSzitas): SIMD Candidate
      for (size_t j = 0; j < centroid.size(); j++) {
        centroid[j] += particle[j];
      }
    }
    for (auto &val : centroid) val /= static_cast<scalar_t>(i);
  }
  void shrink(const std::vector<size_t> &current_order, const scalar_t sigma_,
              const size_t nm_particles, const size_t n_dim) {
    // take a reference to the best vector
    const std::vector<scalar_t> &best =
        this->particle_positions[current_order[0]];
    for (size_t i = 1; i < nm_particles; i++) {
      // update all items in current vector using the best vector -
      // hopefully the contiguous data here can help a bit with cache
      // locality
      std::vector<scalar_t> &current =
          this->particle_positions[current_order[i]];
      // TODO(JSzitas): SIMD Candidate
      for (size_t j = 0; j < n_dim; j++) {
        current[j] = best[j] + sigma_ * (current[j] - best[j]);
      }
    }
  }
  scalar_t simplex_std_err(const std::vector<size_t> &current_order,
                           const size_t nm_particles) {
    size_t i = 0;
    scalar_t mean_val = 0, result = 0;
    for (; i < nm_particles; i++) {
      mean_val += this->particle_current_values[current_order[i]];
    }
    mean_val /= static_cast<scalar_t>(i);
    i = 0;
    for (; i < nm_particles; i++) {
      result +=
          pow(this->particle_current_values[current_order[i]] - mean_val, 2);
    }
    result /= (scalar_t)(i - 1);
    return sqrt(result);
  }
};
}  // namespace nlsolver

namespace nlsolver::rootfinder {
template <typename Callable, typename scalar_t>
[[maybe_unused]] solver_status<scalar_t> bisection(
    Callable &f, scalar_t &x, const scalar_t lower = -100,
    const scalar_t upper = 100, const scalar_t eps = 1e-6,
    const size_t max_iter = 200) {
  size_t f_evals_used = 0;
  auto f_lam = [&](auto x) {
    auto val = f(x);
    f_evals_used++;
    return val;
  };
  scalar_t a = lower, b = upper;
  if (a > b) {
    std::swap(a, b);
  }
  if (f_lam(a) * f_lam(b) >= 0) {
    std::cout << "Initial values do not bracket the root; exiting."
              << std::endl;
    return solver_status<scalar_t>(0, 0, f_evals_used);
  }
  scalar_t mid, val;
  size_t iter = 0;
  while (true) {
    // define midpoint
    mid = (a + b) / 2;
    val = f_lam(mid);
    if (std::abs(val) < eps || iter > max_iter) {
      //; solution found
      x = mid;
      return solver_status<scalar_t>(val, iter, f_evals_used);
    }
    if (val > 0) {
      b = mid;
    } else if (val < 0) {
      a = mid;
    }
    iter++;
  }
}
template <typename Callable, typename scalar_t>
[[maybe_unused]] solver_status<scalar_t> false_position(
    Callable &f, scalar_t &x, const scalar_t lower = -100,
    const scalar_t upper = 100, const scalar_t eps = 1e-6,
    const size_t max_iter = 200) {
  size_t f_evals_used = 0;
  auto f_lam = [&](auto x) {
    auto val = f(x);
    f_evals_used++;
    return val;
  };
  scalar_t val_a = f_lam(lower), val_b = f_lam(upper);
  if (val_a * val_b >= 0) {
    std::cout << "Initial values do not bracket the root; exiting."
              << std::endl;
    return solver_status<scalar_t>(0, 0, f_evals_used);
  }
  scalar_t a = lower, b = upper, mid, val;
  size_t iter = 0;
  while (true) {
    // define midpoint
    mid = a + ((b - a) * val_a) / (val_a - val_b);
    val = f_lam(mid);
    if (std::abs(val) < eps || iter > max_iter) {
      //; solution found
      x = mid;
      return solver_status<scalar_t>(val, iter, f_evals_used);
    }
    if (val < 0) {
      a = mid;
      val_a = val;
    } else if (val > 0) {
      b = mid;
      val_b = mid;
    }
    iter++;
  }
}

template <typename Callable, typename scalar_t>
[[maybe_unused]] solver_status<scalar_t> brent(Callable &f, scalar_t &x,
                                               const scalar_t lower,
                                               const scalar_t upper,
                                               const scalar_t tol = 1e-12,
                                               const size_t max_iter = 200) {
  size_t f_evals_used = 0;
  // functor
  auto f_lam = [&](auto x) {
    const auto val = f(x);
    f_evals_used++;
    return val;
  };
  scalar_t a = lower, b = upper;
  scalar_t val_a = f_lam(a), val_b = f_lam(b);
  if (val_a * val_b >= 0) {
    std::cout << "Initial values do not bracket the root; exiting."
              << std::endl;
    return solver_status<scalar_t>(0, 0, f_evals_used);
  }
  scalar_t c = a, val_c = val_a, s, val_s, d;
  size_t iter = 0;
  bool flag = true;
  while (true) {
    // inverse quadratic interpolation
    if ((val_a != val_c) && (val_b != val_c)) {
      s = ((a * val_b * val_c) / ((val_a - val_b) * (val_a - val_c))) +
          ((b * val_a * val_c) / ((val_b - val_a) * (val_b - val_c))) +
          ((c * val_a * val_b) / ((val_c - val_a) * (val_c - val_b)));
    } else {
      // otherwise do the secant method
      s = b - val_b * ((b - a) / (val_b - val_a));
    }
    // bisection
    if (!(s > (((3 * a) + b) / 4) && s < b) ||
        (flag && (std::abs(s - b) >= (std::abs(b - c) / 2))) ||
        (!flag && std::abs(s - b) >= (std::abs(c - d) / 2)) ||
        (flag && std::abs(b - c) < tol) || (!flag && std::abs(c - d) < tol)) {
      s = (a + b) / 2;
      flag = true;
    } else {
      flag = false;
    }
    val_s = f_lam(s);
    d = c;
    c = b;
    val_c = val_b;
    if ((val_a * val_s) < 0) {
      b = s;
      val_b = val_s;
    } else {
      a = s;
      val_a = val_s;
    }
    if (std::abs(val_a) < std::abs(val_b)) {
      std::swap(a, b);
      std::swap(val_a, val_b);
    }
    if (std::abs(val_b) < tol || std::abs(val_s) < tol ||
        std::abs(b - a) < tol || iter >= max_iter) {
      x = b;
      return solver_status<scalar_t>(val_b, iter, f_evals_used);
    }
    iter++;
  }
}

template <typename Callable, typename scalar_t>
[[maybe_unused]] solver_status<scalar_t> ridders(Callable &f, scalar_t &x,
                                                 const scalar_t lower,
                                                 const scalar_t upper,
                                                 const scalar_t tol = 1e-12,
                                                 const scalar_t eps = 1e-12,
                                                 const size_t max_iter = 5) {
  size_t f_evals_used = 0;
  // functor
  auto f_lam = [&](auto x) {
    const auto val = f(x);
    f_evals_used++;
    return val;
  };
  scalar_t a = lower, b = upper;
  scalar_t val_a = f_lam(a), val_b = f_lam(b);
  if (val_a * val_b >= 0) {
    std::cout << "Initial values do not bracket the root; exiting."
              << std::endl;
    return solver_status<scalar_t>(0, 0, f_evals_used);
  }
  size_t iter = 0;
  scalar_t mid, val_mid, new_mid;
  while (true) {
    // form an evaluate at midpoint
    // evaluate at midpoint
    mid = (a + b) / 2, val_mid = f_lam(mid);
    // new iterate
    new_mid =
        mid + (mid - a) * (std::copysign(1., val_a - val_b) * val_mid /
                           std::sqrt(std::pow(val_mid, 2) - (val_a * val_b)));
    // check if equal zero, if yes return
    scalar_t val_new_mid = f_lam(new_mid);
    // check tolerances
    if (std::min(std::abs(new_mid - a), std::abs(new_mid - b)) < tol ||
        std::abs(val_new_mid) < eps || iter >= max_iter) {
      x = new_mid;
      return solver_status<scalar_t>(val_new_mid, iter, f_evals_used);
    }
    // not matching signs
    if ((val_mid * val_new_mid) < 0) {
      a = mid;
      val_a = val_mid;
      b = new_mid;
      val_b = val_new_mid;
      // likewise, not matching signs
    } else if ((val_a * val_new_mid) < 0) {
      a = new_mid;
      val_a = val_new_mid;
    } else {
      b = new_mid;
      val_b = val_new_mid;
    }
    iter++;
  }
}

namespace nlsolver::internal::circulant {
template <typename T, const size_t size>
struct fixed_circulant {
 private:
  std::array<T, size> data = std::array<T, size>();
  size_t circle_index = 0;

 public:
  T &operator[](const size_t i) {
    return this->data[(this->circle_index + i) % size];
  }
  void push_back(const T item) {
    this->data[this->circle_index] = item;
    this->circle_index = (this->circle_index + 1) % size;
  }
  auto last() { return this->data[this->circle_index]; }
};
};  // namespace nlsolver::internal::circulant
template <typename Callable, typename scalar_t>
[[maybe_unused]] solver_status<scalar_t> tiruneh(
    Callable &f, scalar_t &x,
    const std::array<scalar_t, 3> x_k = {-100., 0., 100.},
    const scalar_t eps = 1e-6, const scalar_t tol = 1e-12,
    const size_t max_iter = 10) {
  size_t f_evals_used = 0;
  // functor
  auto f_lam = [&](auto x) {
    const auto val = f(x);
    f_evals_used++;
    return val;
  };
  // initialize circulant
  nlsolver::internal::circulant::fixed_circulant<scalar_t, 3> k, f_k;
  for (auto &val : x_k) {
    k.push_back(val);
  }
  for (size_t i = 0; i < 3; i++) {
    f_k.push_back(f_lam(k[i]));
  }
  size_t iter = 0;
  scalar_t temp = 0.;
  while (true) {
    if (std::abs(f_k.last()) < tol || iter > max_iter ||
        std::abs(f_k[0] - f_k[1]) < eps) {
      x = k.last();
      return solver_status<scalar_t>(f_k.last(), iter, f_evals_used);
    }
    // iteration and replacement
    temp = k[2] - (f_k[2] * (f_k[0] - f_k[1])) /
                      (((f_k[0] - f_k[2]) / (k[0] - k[2]) * (f_k[0] - f_k[1])) -
                       f_k[0] * (((f_k[0] - f_k[2]) / (k[0] - k[2])) -
                                 ((f_k[1] - f_k[2]) / (k[1] - k[2]))));
    // update
    k.push_back(temp);
    f_k.push_back(f_lam(temp));
    iter++;
  }
}
template <typename Callable, typename scalar_t>
[[maybe_unused]] solver_status<scalar_t> itp(
    Callable &f, scalar_t &x, const scalar_t lower, const scalar_t upper,
    const scalar_t kappa1 = 0.3, const scalar_t kappa2 = 2.1,
    const scalar_t n0 = 1, const scalar_t tol = 1e-12,
    const scalar_t eps = 1e-12, const size_t max_iter = 200) {
  size_t f_evals_used = 0;
  // functor
  auto f_lam = [&](auto x) {
    const auto val = f(x);
    f_evals_used++;
    return val;
  };
  scalar_t a = lower, b = upper;
  // standard bracketing condition
  scalar_t val_a = f_lam(a), val_b = f_lam(b);
  if (val_a * val_b >= 0) {
    std::cout << "Initial values do not bracket the root; exiting."
              << std::endl;
    return solver_status<scalar_t>(0, 0, f_evals_used);
  }
  size_t iter = 0;
  const scalar_t two_eps = 2 * eps;
  const scalar_t n_max = std::log2((b - a) / two_eps) + n0;
  scalar_t mid, r, delta, b_min_a, interp, sigma, temp, val_itp = 100000.0;
  while (true) {
    b_min_a = b - a;
    if (b_min_a < two_eps || iter >= max_iter) {
      x = (a + b) / 2.;
      return solver_status<scalar_t>(val_itp, iter, f_evals_used);
    }
    mid = (a + b) / 2;
    r = eps * std::pow(2, n_max - 1) - (b_min_a / 2);
    delta = kappa1 * std::pow(b_min_a, kappa2);
    // interpolate
    interp = (val_b * a - val_a * b) / (val_b - val_a);
    // truncate
    temp = mid - interp;
    sigma = (temp) > 0;
    const bool project = temp <= r;
    if (delta <= std::abs(temp)) {
      interp += sigma * delta;
    } else {
      interp = mid;
    }
    // project
    if (project) {
      temp = interp;
    } else {
      temp = mid - sigma * r;
    }
    // update
    val_itp = f_lam(temp);
    if (val_itp > 0) {
      b = temp;
      val_b = val_itp;
    } else if (val_itp < 0) {
      a = temp;
      val_a = val_itp;
    } else {
      x = temp;
      return solver_status<scalar_t>(val_itp, iter, f_evals_used);
    }
    iter++;
  }
}

template <typename Callable, typename scalar_t>
[[maybe_unused]] solver_status<scalar_t> chandrupatla(
    Callable &f, scalar_t &x, const scalar_t lower, const scalar_t upper,
    const scalar_t eps_m = 1e-10, const scalar_t eps_a = 2e-10,
    const size_t max_iter = 200) {
  size_t f_evals_used = 0;
  // functor
  auto f_lam = [&](auto x) {
    const auto val = f(x);
    f_evals_used++;
    return val;
  };
  scalar_t a = lower, b = upper, c = upper;
  // standard bracketing condition
  scalar_t val_a = f_lam(a), val_b = f_lam(b), val_c = upper;
  if (val_a * val_b >= 0) {
    std::cout << "Initial values do not bracket the root; exiting."
              << std::endl;
    return solver_status<scalar_t>(0, 0, f_evals_used);
  }
  size_t iter = 0;
  scalar_t t = 0.5, x_t, val_t, x_m, val_m;
  while (true) {
    // use t to linearly interpolate between a and b,
    // and evaluate this function as our newest estimate xt
    x_t = b + t * (a - b);
    val_t = f_lam(x_t);
    // std::cout << "x_t: " << x_t << ", val t: "<< val_t << std::endl;
    //  not matching signs
    if ((val_t * val_b) < 0) {
      c = a;
      a = b;
      val_c = val_a;
      val_a = val_b;
    } else {
      c = b;
      val_c = val_b;
    }
    b = x_t;
    val_b = val_t;
    // set xm so that f(xm) is the minimum magnitude of f(a) and f(b)
    const bool b_smaller_a = std::abs(val_b) < std::abs(val_a);
    x_m = b_smaller_a ? b : a;
    val_m = b_smaller_a ? val_b : val_a;
    // std::cout << "x_m: " << x_m << ", val m: "<< val_m << std::endl;
    if (std::abs(val_m) < eps_a || iter > max_iter) {
      x = x_m;
      return solver_status<scalar_t>(val_m, iter, f_evals_used);
    }
    // Figure out values xi and phi to determine which method we should use next
    const scalar_t tol = 2 * eps_m * std::abs(x_m) + eps_a;
    const scalar_t t_lim = tol / std::abs(a - c);
    if (t_lim > 0.5) {
      // std::cout << "Hitting t lim" << std::endl;
      x = x_m;
      return solver_status<scalar_t>(val_m, iter, f_evals_used);
    }
    const scalar_t xi = (a - b) / (c - b);
    const scalar_t phi = (val_a - val_b) / (val_c - val_b);
    if ((std::pow(phi, 2) < xi) && (std::pow(1 - phi, 2) < (1 - xi))) {
      // std::cout << "Doing inverse quadratic"<< std::endl;
      t = val_a / (val_b - val_a) * val_c / (val_b - val_c) +
          (c - a) / (b - a) * val_a / (val_c - val_a) * val_b / (val_c - val_b);
    } else {
      t = 0.5;
    }
    // limit to the range (tlim, 1-tlim)
    t = std::min(1. - t_lim, std::max(t_lim, t));
    iter++;
  }
}
};  // namespace nlsolver::rootfinder

namespace nlsolver::experimental {
// TODO(JSzitas): WIP
template <typename Callable, typename RNG, typename scalar_t = double>
class [[maybe_unused]] CMAES {
 private:
  Callable &f;
  RNG &generator;
  const scalar_t m_step;
  const size_t max_iter;
  const scalar_t condition, x_delta, f_delta;

 public:
  // constructor
  [[maybe_unused]] CMAES<Callable, RNG, scalar_t>(
      Callable &f, RNG &generator, const scalar_t m_step,
      const size_t max_iter = 1000, const scalar_t condition = 1e14,
      const scalar_t x_delta = 1e-7, const scalar_t f_delta = 1e-9)
      : f(f),
        generator(generator),
        m_step(m_step),
        max_iter(max_iter),
        condition(condition),
        x_delta(x_delta),
        f_delta(f_delta) {}
  // minimize interface
  [[maybe_unused]] solver_status<scalar_t> minimize(std::vector<scalar_t> &x) {
    return this->solve<true>(x);
  }
  // maximize helper
  [[maybe_unused]] solver_status<scalar_t> maximize(std::vector<scalar_t> &x) {
    return this->solve<false>(x);
  }

 private:
  template <const bool minimize = true>
  solver_status<scalar_t> solve(std::vector<scalar_t> &x) {
    const size_t n = x.size();
    size_t la = std::ceil(4 + round(3 * log(n)));
    const size_t mu = std::floor(la / 2);
    std::vector<scalar_t> w(mu);
    for (size_t i = 0; i < mu; i++) {
      w[i] = std::log(static_cast<scalar_t>(mu) + 1 / 2) -
             log(static_cast<double>(i) + 1);
    }
    scalar_t w_sum = 0.0;
    for (auto &weight : w) w_sum += weight;
    // normalize weights
    for (auto &weight : w) weight /= w_sum;
    scalar_t mu_eff = 0.0;
    for (auto &weight : w) mu_eff += weight * weight;
    mu_eff = 1 / mu_eff;
    // increase to 16 samples for very small populations
    la = std::max<size_t>(size_t(16), la);
    const scalar_t cc = (4. + mu_eff / n) / (n + 4. + 2. * mu_eff / n);
    const scalar_t cs = (mu_eff + 2.) / (n + mu_eff + 5.);
    const scalar_t c1 = 2. / (pow(n + 1.3, 2.) + mu_eff);
    const scalar_t cmu = std::min(
        1. - c1, 2. * (mu_eff - 2. + 1. / mu_eff) / (pow(n + 2, 2.) + mu_eff));
    const scalar_t ds =
        1. + cs + 2. * std::max(0., sqrt((mu_eff - 1.) / (n + 1.)) - 1.);
    const scalar_t chi = sqrt(n) * (1. - 1. / (4. * n) + 1. / (21. * n * n));
    const scalar_t hsig_thr = (1.4 + 2 / (n + 1.)) * chi;
    std::vector<scalar_t> pc(n, 0.0), ps(n, 0.0);
    // these are technically matrices, but we probably only need to allocate C
    // here
    std::vector<scalar_t> B(n * n, 0.0), D(n * n, 0.0), C(n * n, 0.0);
    for (size_t i = 0; i < n; i++) {
      B[i + i * n] = 1.0;
      C[i + i * n] = 1.0;
      D[i + i * n] = 1.0;
    }
    scalar_t sigma = m_step;
    std::vector<scalar_t> x_mean = x;
    std::vector<scalar_t> z_mean(n, 0.0);
    std::vector<scalar_t> costs(la);
    scalar_t current_value = this->f(x);
    size_t eigen_last_eval = 0;
    size_t eigen_next_eval =
        std::max<size_t>(1, static_cast<size_t>(1 / (10 * n * (c1 + cmu))));
    std::vector<scalar_t> particles_z(n * la);
    // this is a std::vector<std::vector<scalar_t>> only because of how the
    // interface to f is
    std::vector<scalar_t> particles_x(n * la);
    // take iterator type and use it in the lambda
    size_t f_evals = 0;
    using ItType = decltype(particles_x.begin());
    std::vector<scalar_t> temp_eval_vec = x;
    constexpr scalar_t f_multiplier = minimize ? -1.0 : 1.0;
    // set up a lambda to adapt the interface, and to allow us to keep track of
    // function evaluations
    auto f_lam = [&](ItType first, ItType last) {
      size_t p = 0;
      // copy values over to temporary
      for (; first != last; ++first) {
        temp_eval_vec[p++] = *first;
      }
      f_evals++;
      // run evaluation on temporary
      return f_multiplier * this->f(temp_eval_vec);
    };
    size_t iter = 0;
    // initialize QRSolver
    auto qr = tinyqr::QRSolver<scalar_t>(n);
    // allocate BD and initialize it to identity matrix
    std::vector<scalar_t> BD(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
      BD[i * n + i] = 1.0;
    }
    const scalar_t cs_update = std::sqrt(cs * (2. - cs) * mu_eff);
    const scalar_t cc_update = std::sqrt(cc * (2. - cc) * mu_eff);
    const scalar_t c_update = (1 - c1 - cmu);
    const scalar_t cc_c_update = cc * (2 - cc);
    // temporary storage - useful throughout
    std::vector<scalar_t> temp_vec(std::max(n, la), 0.0);
    // run main loop
    while (true) {
      // update Z particles
      rnorm(this->generator, particles_z.begin(), particles_z.end());
      for (int k = 0; k < la; ++k) {
        // generate particles using current means, sigma and projection
        scalar_t temp = 0.0;
        for (size_t j = 0; j < n; j++) {
          // the BD * particles_z is just a matrix vector product where
          // particles_z can just be a pointer and offset this should be pretty
          // easy to rewrite then in terms of e.g. a_plus_scalar_times_b as BD *
          // particles_z is just a vector in the end of size nx1
          // TODO(JSzitas): rewrite matrix product
          // row of BD with value in particles
          // scalar_t temp = 0.0;
          for (size_t i = 0; i < n; i++) {
            temp_vec[k] += BD[i * n + j] * particles_z[i * n + j];
          }
          particles_x[k * n + j] = x_mean[j] + sigma;  //* //temp;
        }
        // arx.col(k) = x_mean + sigma * B*D*arz.col(k);
        costs[k] =
            f_lam(particles_x.begin() + k * n, particles_x.end() + k * n);
      }
      std::vector<size_t> indices = index_partial_sort(costs, mu);
      // reset x_mean and z_mean
      std::fill(x_mean.begin(), x_mean.end(), 0.0);
      std::fill(z_mean.begin(), z_mean.end(), 0.0);
      for (int k = 0; k < mu; k++) {
        for (size_t j = 0; j < n; j++) {
          z_mean[j] += w[k] * particles_z[indices[k] * n + j];
          x_mean[j] += w[k] * particles_x[indices[k] * n + j];
        }
        // z_mean += w[k] * particles_z arz.col(indices[k]);
        // x_mean += w[k]*arx.col(indices[k]);
      }
      // Update evolution paths using projection
      // this is the only place where we need B, not BD, and the result is a
      // vector
      // ps = (1. - cs)*ps + mu_eff_update * B * z_mean;
      const scalar_t cs_mult = (1 - cs);
      for (size_t i = 0; i < n; i++) {
        // TODO: SIMD candidate
        ps[i] =
            cs_mult * ps[i] + cs_update * qr.sqrt_eigenvalues(i) * z_mean[i];
      }
      // this is nasty, surely we can make it a bit nicer
      // note that hsig basically works as a condition here; this might be one
      // of the few times when using an if statement could be a significant
      // optimization, as the switch leads to a lot of multiplication that gets
      // carried out of it is == zero, which might be entirely redundant if this
      // did not rely on iter, line 3804 could be constexpr, but I am not
      // convinced its worthwhile
      const bool hsig =
          (nlsolver::math::norm(ps.data(), n) /
           std::sqrt(std::pow(1 - (1. - cs), (2 * (iter + 1))))) < hsig_thr;
      const scalar_t cc_mult = (1. - cc);
      for (size_t i = 0; i < n; i++) {
        pc *= cc_mult;
      }
      // default C update is on a constant schedule
      C = c_update * C;
      // only under hsig do we actually do anything dynamic => and that only
      // happens for all **potential** threads in lockstep anyway at this level
      if (hsig) {
        pc += cc_update * (B * D * z_mean);
        C += c1 * (pc * pc.transpose());
      } else {
        C += c1 * (pc * pc.transpose() + cc_c_update * C);
      }
      for (int k = 0; k < mu; ++k) {
        // this should not get reallocated all the time, this is matrix * matrix
        // * some vector, effectively reducing us to a vector, but B*D should be
        // trivial to calculate ahead of time plus the D matrix should be a
        // diagonal matrix (eigenvalues only) so why use full matrix multiplies
        // TODO(JSzitas):
        // we can actually cache this from computing the projection from before
        // TVector temp = BD*arz.col(k);
        // the product temp * temp' can probably be avoided altogether seeing we
        // are writing to a matrix
        // C += cmu * w[k] * temp * temp.transpose();
      }
      // ps norm can be precomputed once, it is also used on line 3950
      sigma *= exp((cs / ds) * ((ps).norm() / chi - 1.));
      // run new evaluations
      if ((iter - eigen_last_eval) == eigen_next_eval) {
        // Update B and D by running eigensolver
        eigen_last_eval = iter;
        // call QR solver's solve method
        qr.solve(C);
        // re-compute new BD without explicitly getting D and B
        // there is a fast way to do it leveraging t(C) = t(B) * t(A) for matrix
        // identities, since A is diagonal and we can do t(t(B)) to leverage
        // cache locality
        for (size_t i = 0; i < n; i++) {
          for (size_t j = 0; j < n; j++) {
            BD[i * n + j] =
                qr.sqrt_eigenvalues(j) *
                qr.eigenvectors()[i * n + j];  // column of D times values in B
          }
        }
      }
      // condition number, useful for convergence - ratio of largest and
      // smallest eigenvalue
      const scalar_t current_condition = qr.eigenvalue_condition_number();
      // Super::m_current.condition = D.diagonal().maxCoeff() /
      // D.diagonal().minCoeff();
      //  compute difference in best value and difference in coefficients,
      //  useful for checking convergence
      const scalar_t current_f_delta =
          std::abs(costs[indices[0]] - current_value);
      const scalar_t current_x_delta =
          nlsolver::math::norm_diff(x.data(), x_mean.data(), n);
      x = x_mean;
      current_value = costs[indices[0]];
      // if all good solutions are becoming too similar, increase variance
      if (std::abs(costs[indices[0]] - costs[indices[mu - 1]]) < 1e-6) {
        sigma *= exp(0.2 + cs / ds);
      }
      if (iter >= this->max_iter || current_condition > this->condition ||
          current_f_delta < this->f_delta || current_x_delta < this->x_delta) {
        x = x_mean;
        m_step = sigma;
        return solver_status<scalar_t>(current_value, iter, f_evals);
      }
      iter++;
    }
  }
};
};  // namespace nlsolver::experimental

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif  // NLSOLVER_H_
