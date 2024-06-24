// Tiny QR solver, header only library
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (C) 2023- Juraj Szitas
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

#ifndef TINYQR_H_
#define TINYQR_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

// vectorised math macros
#if !defined(NO_MANUAL_VECTORIZATION) && defined(__GNUC__) && \
    (__GNUC__ > 6) && defined(__AVX512F__)
#define USE_AVX512
#endif
#if !defined(NO_MANUAL_VECTORIZATION) && defined(__AVX__) && \
    defined(__SSE__) && defined(__SSE2__) && defined(__SSE3__)
#define USE_AVX
#endif

#if defined(USE_AVX) || defined(USE_AVX512)
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__)
#include <immintrin.h>  //<x86intrin.h>
#endif
#endif

namespace tinyqr::internal {
// since we are using 1/sqrt(x) here - default
template <typename scalar_t>
inline __attribute__((always_inline)) scalar_t inv_sqrt(scalar_t x) {
  return 1.0 / std::sqrt(x);
}
// fast inverse square root - might buy us a tiny bit, and I have been looking
// for forever to use this :)
/*
template <>
[[maybe_unused]] inline __attribute__((always_inline)) float inv_sqrt(float x) {
  long i = *reinterpret_cast<long *>(&x);       // evil floating point bit level
hacking // NOLINT [runtime/int] i = 0x5f3759df - (i >> 1);  // what the fuck?
  return *reinterpret_cast<float *>(&i);
}*/
template <typename scalar_t>
inline __attribute__((always_inline)) std::tuple<scalar_t, scalar_t>
givens_rotation(const scalar_t a, const scalar_t b) {
  if (std::abs(b) > std::abs(a)) {
    const scalar_t r = a / b;
    const auto s = static_cast<scalar_t>(inv_sqrt(std::pow(r, 2) + 1.0));
    return {s * r, s};
  }
  const scalar_t r = b / a;
  const auto c = static_cast<scalar_t>(inv_sqrt(std::pow(r, 2) + 1.0));
  return {c, c * r};
}

template <typename scalar_t>
[[maybe_unused]] inline __attribute__((always_inline)) void tA_matmul_B_to_C(
    std::vector<scalar_t> &__restrict__ A,
    std::vector<scalar_t> &__restrict__ B,
    std::vector<scalar_t> &__restrict__ C, const size_t ncol) {
  for (size_t k = 0; k < ncol; k++) {
    for (size_t l = 0; l < ncol; l++) {
      scalar_t accumulator = 0.0;
      for (size_t m = 0; m < ncol; m++) {
        accumulator += A[k * ncol + m] * B[l * ncol + m];
      }
      C[l * ncol + k] = accumulator;
    }
  }
}
// transpose a square matrix in place
template <typename scalar_t>
inline __attribute__((always_inline)) void transpose_square(
    std::vector<scalar_t> &X, const size_t p) {
  for (size_t i = 0; i < p; i++) {
    for (size_t j = i + 1; j < p; j++) {
      std::swap(X[(j * p) + i], X[(i * p) + j]);
    }
  }
}
// TODO(JSzitas): All of the following code supports SIMD
// and this impl should make it a lot easier
template <typename scalar_t>
inline __attribute__((always_inline)) void rotate_matrix(
    scalar_t *__restrict__ lower, scalar_t *__restrict__ upper,
    const scalar_t c, const scalar_t s, size_t p) {
  for (; p > 0; --p) {
    const scalar_t temp_1 = *lower;
    const scalar_t temp_2 = *upper;
    *lower = c * temp_1 + s * temp_2;
    *upper = -s * temp_1 + c * temp_2;
    ++lower;
    ++upper;
  }
}

#ifdef USE_AVX
template <>
inline __attribute__((always_inline)) void rotate_matrix(
    float *__restrict__ lower, float *__restrict__ upper, const float c,
    const float s, size_t p) {
  if (p > 7) {
    const __m256 c_ = _mm256_set1_ps(c);
    const __m256 s_ = _mm256_set1_ps(s);
    __m256 res = _mm256_setzero_ps();
    for (; p > 7; p -= 8) {
      // set current register values
      const auto lower_ = _mm256_loadu_ps(lower);
      const auto upper_ = _mm256_loadu_ps(upper);
      // updates on lower
      res = _mm256_add_ps(_mm256_mul_ps(c_, lower_), _mm256_mul_ps(s_, upper_));
      // store in lower
      _mm256_storeu_ps(lower, res);
      // updates on upper
      res = _mm256_sub_ps(_mm256_mul_ps(c_, upper_), _mm256_mul_ps(s_, lower_));
      // store in upper
      _mm256_storeu_ps(upper, res);
      lower += 8;
      upper += 8;
    }
  }
  for (; p > 0; --p) {
    const float temp_1 = *lower;
    const float temp_2 = *upper;
    *lower = c * temp_1 + s * temp_2;
    *upper = -s * temp_1 + c * temp_2;
    ++lower;
    ++upper;
  }
}
#endif
#ifdef USE_AVX_512
template <>
inline __attribute__((always_inline)) void rotate_matrix(
    float *__restrict__ lower, float *__restrict__ upper, const float c,
    const float s, size_t p) {
  if (p > 15) {
    const __m512 c_ = _mm512_set1_ps(c);
    const __m512 s_ = _mm512_set1_ps(s);
    __m256 res = _mm512_setzero_ps();
    for (; p > 15; p -= 16) {
      // set current register values
      const auto lower_ = _mm512_loadu_ps(lower);
      const auto upper_ = _mm512_loadu_ps(upper);
      // updates on lower
      res = _mm512_add_ps(_mm512_mul_ps(c_, lower_), _mm512_mul_ps(s_, upper_));
      // store in lower
      _mm512_storeu_ps(lower, res);
      // updates on upper
      res = _mm512_sub_ps(_mm512_mul_ps(c_, upper_), _mm512_mul_ps(s_, lower_));
      // store in upper
      _mm512_storeu_ps(upper, res);
      lower += 16;
      upper += 16;
    }
  }
  for (; p > 0; --p) {
    const float temp_1 = *lower;
    const float temp_2 = *upper;
    *lower = c * temp_1 + s * temp_2;
    *upper = -s * temp_1 + c * temp_2;
    ++lower;
    ++upper;
  }
}
#endif

template <typename scalar_t>
inline __attribute__((always_inline)) std::vector<scalar_t> make_identity(
    const size_t n) {
  std::vector<scalar_t> result(n * n, static_cast<scalar_t>(0.0));
  for (size_t i = 0; i < n; i++) result[i * n + i] = static_cast<scalar_t>(1.0);
  return result;
}
template <typename scalar_t, const bool report_success = true,
          const size_t tune = 1000>
[[maybe_unused]] void validate_qr(const std::vector<scalar_t> &__restrict__ X,
                                  const std::vector<scalar_t> &__restrict__ Q,
                                  const std::vector<scalar_t> &__restrict__ R,
                                  const size_t n, const size_t p) {
  // constant factor here added since epsilon is too small otherwise
  constexpr auto eps =
      std::numeric_limits<scalar_t>::epsilon() * static_cast<scalar_t>(tune);
  // this trick is done since some third party limited precision floats
  // do not provide an impl of the constexpr numeric limits epsilon function
  // const auto eps = static_cast<scalar_t>(eps_);
  // Matrix multiplication QR
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < p; ++j) {
      auto tmp = scalar_t(0.0);
      for (size_t k = 0; k < p; ++k) {
        tmp += Q[k * n + i] * R[j * p + k];
      }
      // Compare to original matrix X
      if (std::abs(X[j * n + i] - tmp) > eps) {
        std::cout << "Error in {validate_qr}, " << tmp << " != " << X[i * p + j]
                  << " diff: " << std::abs(X[j * n + i] - tmp)
                  << " eps: " << eps << "\n";
        std::cout << "Failed to recreate input from QR matrices for size " << n
                  << ", " << p << "\n";
        return;
      }
    }
  }
  if constexpr (report_success) {
    std::cout << "Validation of QR successful for size " << n << ", " << p
              << std::endl;
  }
}
template <typename scalar_t, const bool cleanup = false>
void qr_impl(std::vector<scalar_t> &__restrict__ Q,
             std::vector<scalar_t> &__restrict__ R, const size_t n,
             const size_t p, const scalar_t tol) {
  // the key to optimizing this is probably to take R as R transposed - most
  // likely a lot of work is done just in the k loops, which is probably a good
  // place to optimize
  for (size_t j = 0; j < p; j++) {
    for (size_t i = n - 1; i > j; --i) {
      // using tuples and structured bindings should make this fairly ok
      // performance wise
      // check if R[j * n + i] - is not zero; if it is we can skip this
      // iteration
      // if (std::abs(R[i * p + j]) <= std::numeric_limits<scalar_t>::min())
      // continue;
      const auto [c, s] = givens_rotation(R[(i - 1) * p + j], R[i * p + j]);
      // you can make the matrix multiplication implicit, as the givens rotation
      // only impacts a moving 2x2 block
      // R is transposed
      rotate_matrix(R.data() + (i - 1) * p, R.data() + i * p, c, s, p);
      rotate_matrix(Q.data() + (i - 1) * n, Q.data() + i * n, c, s, n);
    }
  }
  // clean up R - particularly under the diagonal - only useful if you are
  // interested in the actual decomposition
  if constexpr (cleanup) {
    for (auto &val : R) {
      val = std::abs(val) < tol ? 0.0 : val;
    }
  }
}
}  // namespace tinyqr::internal
namespace tinyqr {
template <typename scalar_t>
struct QR {
  std::vector<scalar_t> Q;
  std::vector<scalar_t> R;
};
template <typename scalar_t>
[[maybe_unused]] QR<scalar_t> qr_decomposition(const std::vector<scalar_t> &X,
                                               const size_t n, const size_t p,
                                               const scalar_t tol = 1e-8) {
  // initialize Q and R
  std::vector<scalar_t> Q = tinyqr::internal::make_identity<scalar_t>(n);
  // initialize R as transposed
  std::vector<scalar_t> R(X.size(), static_cast<scalar_t>(0.0));
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < p; j++) {
      R[i * p + j] = X[j * n + i];
    }
  }
  tinyqr::internal::qr_impl<scalar_t, true>(Q, R, n, p, tol);
  // we do not actually need to manipulate more than pxp block of R
  tinyqr::internal::transpose_square(R, p);
  Q.resize(n * p);
  R.resize(p * p);
  return {Q, R};
}
template <typename scalar_t>
struct eigendecomposition {
  std::vector<scalar_t> eigenvals;
  std::vector<scalar_t> eigenvecs;
};

template <typename scalar_t>
[[maybe_unused]] eigendecomposition<scalar_t> qr_algorithm(
    const std::vector<scalar_t> &A, const size_t max_iter = 25,
    const scalar_t tol = 1e-8) {
  auto Ak = A;
  // A must be square
  const size_t n = std::sqrt(A.size());
  std::vector<scalar_t> QQ(n * n, 0.0);
  for (size_t i = 0; i < n; i++) QQ[i * n + i] = 1.0;
  // initialize Q and R
  std::vector<scalar_t> Q(n * n, 0.0);
  for (size_t i = 0; i < n; i++) Q[i * n + i] = 1.0;
  std::vector<scalar_t> R(n * n);  // = Ak;
  std::vector<scalar_t> temp(Q.size());
  for (size_t i = 0; i < max_iter; i++) {
    // reset Q and R, G gets reset inside qr_impl
    for (size_t j = 0; j < n; j++) {
      for (size_t k = 0; k < n; k++) {
        // probably a decent way to reset to a diagonal matrix
        Q[j * n + k] = static_cast<scalar_t>(k == j);
        R[j * n + k] = Ak[j * n + k];
      }
    }
    // call QR decomposition
    tinyqr::internal::qr_impl<scalar_t, false>(Q, R, n, n, tol);
    // note QR decomposition returns Rt, not R!!!
    tinyqr::internal::tA_matmul_B_to_C<scalar_t>(R, Q, Ak, n);
    // overwrite QQ in place
    size_t p = 0;
    for (size_t j = 0; j < n; j++) {
      for (size_t k = 0; k < n; k++) {
        temp[p] = 0;
        for (size_t l = 0; l < n; l++) {
          temp[p] += QQ[l * n + k] * Q[j * n + l];
        }
        p++;
      }
    }
    // write to A directly
    for (size_t k = 0; k < QQ.size(); k++) {
      QQ[k] = temp[k];
    }
  }
  // diagonal elements of Ak are eigenvalues - we can just shuffle elements of A
  // and resize
  for (size_t i = 1; i < n; i++) {
    Ak[i] = Ak[i * n + i];
  }
  Ak.resize(n);
  return {Ak, QQ};
}
template <typename scalar_t>
class [[maybe_unused]] QRSolver {
  const size_t n;
  std::vector<scalar_t> Ak, QQ, Q, R, temp, eigval;

 public:
  [[maybe_unused]] explicit QRSolver<scalar_t>(const size_t n) : n(n) {
    this->Ak = std::vector<scalar_t>(n * n);
    this->QQ = std::vector<scalar_t>(n * n, 0.0);
    for (size_t i = 0; i < n; i++) this->QQ[i * n + i] = 1.0;
    // initialize Q and R
    this->Q = std::vector<scalar_t>(n * n, 0.0);
    for (size_t i = 0; i < n; i++) this->Q[i * n + i] = 1.0;
    this->R = std::vector<scalar_t>(n * n);
    this->temp = std::vector<scalar_t>(n * n);
    this->eigval = std::vector<scalar_t>(n);
  }
  [[maybe_unused]] void solve(const std::vector<scalar_t> &A,
                              const size_t max_iter = 25,
                              const scalar_t tol = 1e-8) {
    this->Ak = A;
    // in case we need to reset QQ
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++) {
        this->QQ[i * n + j] = static_cast<scalar_t>(i == j);
      }
    }
    for (size_t i = 0; i < max_iter; i++) {
      // reset Q and R, G gets reset inside qr_impl
      for (size_t j = 0; j < n; j++) {
        for (size_t k = 0; k < n; k++) {
          // probably a decent way to reset to a diagonal matrix
          Q[j * n + k] = static_cast<scalar_t>(k == j);
          R[j * n + k] = Ak[j * n + k];
        }
      }
      // call QR decomposition
      tinyqr::internal::qr_impl<scalar_t, false>(Q, R, n, n, tol);
      // note QR decomposition returns Qt, not Q!!!
      tinyqr::internal::tA_matmul_B_to_C<scalar_t>(R, Q, Ak, n);
      // overwrite QQ in place
      size_t p = 0;
      for (size_t j = 0; j < n; j++) {
        for (size_t k = 0; k < n; k++) {
          temp[p] = 0;
          for (size_t l = 0; l < n; l++) {
            temp[p] += QQ[l * n + k] * Q[j * n + l];
          }
          p++;
        }
      }
      // write to A colwise - i.e. directly
      for (size_t k = 0; k < QQ.size(); k++) {
        QQ[k] = temp[k];
      }
    }
    for (size_t i = 0; i < n; i++) {
      eigval[i] = Ak[i * n + i];
    }
  }
  [[maybe_unused]] const std::vector<scalar_t> &eigenvalues() const {
    return eigval;
  }
  [[maybe_unused]] const std::vector<scalar_t> &eigenvectors() const {
    return this->QQ;
  }
};
// Function for back substitution using QR decomposition result
// this does not require any temporaries
template <typename scalar_t>
std::vector<scalar_t> back_solve(const std::vector<scalar_t> &__restrict__ Q,
                                 const std::vector<scalar_t> &__restrict__ R,
                                 const std::vector<scalar_t> &__restrict__ y,
                                 const size_t nrow, const size_t ncol) {
  std::vector<scalar_t> result(ncol, 0.0);
  for (size_t i = ncol; i-- > 0;) {
    scalar_t temp = 0.0;
    // this might benefit from transposes
    for (size_t j = i + 1; j < ncol; ++j) {
      temp += R[j * ncol + i] * result[j];
    }
    scalar_t y_tmp = 0;
    for (size_t j = 0; j < nrow; ++j) {
      // product Q'y need not be computed ahead of time; we can lazily compute
      // coefficient by coefficient, requiring only one temporary stack
      // allocated variable
      y_tmp += Q[i * nrow + j] * y[j];
    }
    result[i] = (y_tmp - temp) / R[i * ncol + i];
  }
  return result;
}
template <typename scalar_t>
[[maybe_unused]] std::vector<scalar_t> lm(
    const std::vector<scalar_t> &__restrict__ X,
    const std::vector<scalar_t> &__restrict__ y, const scalar_t tol = 1e-12) {
  const size_t nrow = y.size();
  const size_t ncol = X.size() / nrow;
  // compute QR decomposition
  const auto qr = tinyqr::qr_decomposition(X, nrow, ncol, tol);
  // solve Rx = Q'y
  return back_solve(qr.Q, qr.R, y, nrow, ncol);
}
}  // namespace tinyqr
#endif  // TINYQR_H_"
