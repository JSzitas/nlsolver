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

#ifndef TEST_FUNCTIONS_H_
#define TEST_FUNCTIONS_H_

#include <array>
#include <cmath>
#include <vector>

namespace nlsolver::test_functions {
template <typename T>
struct Sphere {
  T operator()(const std::vector<T> &x) { return x[0] * x[0] + x[1] * x[1]; }
  const std::vector<T> minimum() const { return {0.0, 0.0}; }
};
template <typename T>
struct Rosenbrock {
  T operator()(const std::vector<T> &x) {
    return 100.0 * std::pow((x[0] * x[0] - x[1]), 2.0) +
           std::pow((x[0] - 1), 2.0);
  }

  const std::vector<T> minimum() const { return {1.0, 1.0}; }
};
template <typename T>
struct Rastrigin {
  T operator()(const std::vector<T> &x) {
    return 2 * 10 + (x[0] * x[0] - 10 * std::cos(2 * M_PI * x[0])) +
           (x[1] * x[1] - 10 * std::cos(2 * M_PI * x[1]));
  }

  const std::vector<T> minimum() const { return {0.0, 0.0}; }
};
template <typename T>
struct Ackley {
  T operator()(const std::vector<T> &x) {
    T a = -20 * std::exp(-0.2 * std::sqrt(0.5 * (x[0] * x[0] + x[1] * x[1])));
    T b = -std::exp(0.5 *
                    (std::cos(2 * M_PI * x[0]) + std::cos(2 * M_PI * x[1])));
    return a + b + std::exp(1) + 20;
  }

  const std::vector<T> minimum() const { return {0.0, 0.0}; }
};
template <typename T>
struct Beale {
  T operator()(const std::vector<T> &x) {
    return std::pow(1.5 - x[0] + x[0] * x[1], 2) +
           std::pow(2.25 - x[0] + x[0] * x[1] * x[1], 2) +
           std::pow(2.625 - x[0] + x[0] * x[1] * x[1] * x[1], 2);
  }

  const std::vector<T> minimum() const { return {3.0, 0.5}; }
};
template <typename T>
struct Goldstein_Price {
  T operator()(const std::vector<T> &x) {
    T a = 1 + std::pow(x[0] + x[1] + 1, 2) *
                  (19 - 14 * x[0] + 3 * x[0] * x[0] - 14 * x[1] +
                   6 * x[0] * x[1] + 3 * x[1] * x[1]);
    T b = 30 + std::pow(2 * x[0] - 3 * x[1], 2) *
                   (18 - 32 * x[0] + 12 * x[0] * x[0] + 48 * x[1] -
                    36 * x[0] * x[1] + 27 * x[1] * x[1]);
    return a * b;
  }

  const std::vector<T> minimum() const { return {0.0, -1.0}; }
};
template <typename T>
struct Himmelblau {
  T operator()(const std::vector<T> &x) {
    return std::pow(x[0] * x[0] + x[1] - 11, 2) +
           std::pow(x[0] + x[1] * x[1] - 7, 2);
  }
  const std::vector<T> minimum(const size_t index = 0) const {
    const std::array<std::vector<T>, 4> minima = {
        {3.0, 2.0},
        {-2.805118, 3.131312},
        {-3.779310, -3.283186},
        {3.584428, -1.848126},
    };
    return minima[index];
  }
};
template <typename T>
struct ThreeHumpCamel {
  T operator()(const std::vector<T> &x) {
    return 2 * x[0] * x[0] - 1.05 * std::pow(x[0], 4) + std::pow(x[0], 6) / 6 +
           x[0] * x[1] + x[1] * x[1];
  }
  const std::vector<T> minimum() const { return {0.0, 0.0}; }
};
template <typename T>
struct CrossInTray {
  T operator()(const std::vector<T> &x) {
    return -0.0001 *
           std::pow(std::abs(std::sin(x[0]) * std::sin(x[1]) *
                             std::exp(std::abs(
                                 100 - std::sqrt(x[0] * x[0] + x[1] * x[1]) /
                                           M_PI))) +
                        1,
                    0.1);
  }
  const std::vector<T> minimum(const size_t index = 0) const {
    const std::array<std::vector<T>, 4> minima = {
        {1.34941, -1.34941},
        {1.34941, 1.34941},
        {-1.34941, 1.34941},
        {-1.34941, -1.34941},
    };
    return minima[index];
  }
};
template <typename T>
struct Eggholder {
  T operator()(const std::vector<T> &x) {
    return -(x[1] + 47) *
               std::sin(std::sqrt(std::abs(x[0] / 2 + (x[1] + 47)))) -
           x[0] * std::sin(std::sqrt(std::abs(x[0] - (x[1] + 47))));
  }
  const std::vector<T> minimum() const { return {512.0, 404.2319}; }
};
template <typename T>
struct HolderTable {
  T operator()(const std::vector<T> &x) {
    return -std::abs(
        std::sin(x[0]) * std::cos(x[1]) *
        std::exp(std::abs(1 - std::sqrt(x[0] * x[0] + x[1] * x[1]) / M_PI)));
  }
  const std::vector<T> minimum(const size_t index = 0) const {
    const std::array<std::vector<T>, 4> minima = {
        {8.05502, 9.66459},
        {-8.05502, 9.66459},
        {8.05502, -9.66459},
        {-8.05502, -9.66459},
    };
    return minima[index];
  }
};
template <typename T>
struct McCormick {
  T operator()(const std::vector<T> &x) {
    return std::sin(x[0] + x[1]) + std::pow(x[0] - x[1], 2) - 1.5 * x[0] +
           2.5 * x[1] + 1;
  }
  const std::vector<T> minimum() const { return {-0.54719, -1.54719}; }
};
template <typename T>
struct SchafferN2 {
  T operator()(const std::vector<T> &x) {
    return 0.5 + (std::pow(std::sin(x[0] * x[0] - x[1] * x[1]), 2) - 0.5) /
                     std::pow(1 + 0.001 * (x[0] * x[0] + x[1] * x[1]), 2);
  }
  const std::vector<T> minimum() const { return {0.0, 0.0}; }
};
template <typename T>
struct SchafferN4 {
  T operator()(const std::vector<T> &x) {
    return 0.5 +
           (std::pow(std::cos(std::sin(std::abs(x[0] * x[0] - x[1] * x[1]))),
                     2) -
            0.5) /
               std::pow(1 + 0.001 * (x[0] * x[0] + x[1] * x[1]), 2);
  }
  const std::vector<T> minimum(const size_t index = 0) const {
    const std::array<std::vector<T>, 4> minima = {
        {0.0, 1.25313},
        {0.0, -1.25313},
        {1.25313, 0.0},
        {-1.25313, 0.0},
    };
    return minima[index];
  }
};
template <typename T>
struct StyblinskiTang {
  T operator()(const std::vector<T> &x) {
    T sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
      sum += std::pow(x[i], 4) - 16 * std::pow(x[i], 2) + 5 * x[i];
    }
    return sum / 2.0;
  }
  const std::vector<T> minimum() const { return {-2.903534, -2.903534}; }
};
template <typename T, const size_t n_dim = 4, const size_t n_max = 10>
struct Shekel {
  T operator()(const std::vector<T> &x) {
    const std::array<T, n_dim *n_max> a = {
        4, 4, 4, 4, 1, 1, 1, 1, 8, 8, 8, 8, 6, 6, 6, 6, 3, 7,   3, 7,
        2, 9, 2, 9, 5, 5, 3, 3, 8, 1, 8, 1, 6, 2, 6, 2, 7, 3.6, 7, 3.2};
    constexpr std::array<T, n_max> c = {0.1, 0.2, 0.2, 0.4, 0.4,
                                        0.6, 0.3, 0.7, 0.5, 0.5};
    T sum = 0.0;
    for (size_t i = 0; i < n_max; ++i) {
      T inner_sum = 0.0;
      for (size_t j = 0; j < n_dim; ++j) {
        inner_sum += std::pow(x[j] - a[i * n_dim + j], 2);
      }
      sum += 1.0 / (inner_sum + c[i]);
    }
    return -sum;
  }
  const std::vector<T> minimum() const { return {4.0, 4.0, 4.0, 4.0}; }
};
template <typename T>
struct Booth {
  T operator()(const std::vector<T> &x) {
    return std::pow(x[0] + 2 * x[1] - 7, 2) + std::pow(2 * x[0] + x[1] - 5, 2);
  }
  const std::vector<T> minimum() const { return {1.0, 3.0}; }
};
template <typename T>
struct BukinN6 {
  T operator()(const std::vector<T> &x) {
    return 100 * std::sqrt(std::abs(x[1] - 0.01 * x[0] * x[0])) +
           0.01 * std::abs(x[0] + 10);
  }
  const std::vector<T> minimum() const { return {-10.0, 1.0}; }
};
template <typename T>
struct Matyas {
  T operator()(const std::vector<T> &x) {
    return 0.26 * (x[0] * x[0] + x[1] * x[1]) - 0.48 * x[0] * x[1];
  }
  const std::vector<T> minimum() const { return {0.0, 0.0}; }
};
template <typename T>
struct LeviN13 {
  T operator()(const std::vector<T> &x) {
    return std::pow(std::sin(3 * M_PI * x[0]), 2) +
           std::pow((x[0] - 1), 2) *
               (1 + std::pow(std::sin(3 * M_PI * x[1]), 2)) +
           std::pow((x[1] - 1), 2) *
               (1 + std::pow(std::sin(2 * M_PI * x[1]), 2));
  }
  const std::vector<T> minimum() const { return {1.0, 1.0}; }
};
};  // namespace nlsolver::test_functions

#endif  // TEST_FUNCTIONS_H_
