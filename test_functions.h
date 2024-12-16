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
#include <cstdint>  // NOLINT
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <variant>
#include <vector>

#include "./nlsolver.h"
struct mt {
 private:
  std::mt19937 generator;
  std::uniform_real_distribution<double> distribution;

 public:
  explicit mt(const int seed = 42) : generator(42), distribution(0.0, 1.0) {}
  double operator()() { return this->distribution(this->generator); }
};

namespace nlsolver::test_functions {
template <typename T>
struct Sphere {
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 1; }
  T operator()(const std::vector<T> &x) { return x[0] * x[0] + x[1] * x[1]; }
  const std::vector<T> minimum() const { return {0.0, 0.0}; }
};

template <typename T>
struct Rosenbrock {
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 1; }
  T operator()(const std::vector<T> &x) {
    return 100.0 * std::pow((x[0] * x[0] - x[1]), 2.0) +
           std::pow((x[0] - 1), 2.0);
  }
  const std::vector<T> minimum() const { return {1.0, 1.0}; }
};

template <typename T>
struct Rastrigin {
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 1; }
  T operator()(const std::vector<T> &x) {
    return 2 * 10 + (x[0] * x[0] - 10 * std::cos(2 * M_PI * x[0])) +
           (x[1] * x[1] - 10 * std::cos(2 * M_PI * x[1]));
  }
  const std::vector<T> minimum() const { return {0.0, 0.0}; }
};

template <typename T>
struct Ackley {
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 1; }
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
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 1; }
  T operator()(const std::vector<T> &x) {
    return std::pow(1.5 - x[0] + x[0] * x[1], 2) +
           std::pow(2.25 - x[0] + x[0] * x[1] * x[1], 2) +
           std::pow(2.625 - x[0] + x[0] * x[1] * x[1] * x[1], 2);
  }
  const std::vector<T> minimum() const { return {3.0, 0.5}; }
};

template <typename T>
struct Goldstein_Price {
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 1; }
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
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 4; }
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
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 1; }
  T operator()(const std::vector<T> &x) {
    return 2 * x[0] * x[0] - 1.05 * std::pow(x[0], 4) + std::pow(x[0], 6) / 6 +
           x[0] * x[1] + x[1] * x[1];
  }
  const std::vector<T> minimum() const { return {0.0, 0.0}; }
};
template <typename T>
struct CrossInTray {
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 4; }
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
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 1; }
  T operator()(const std::vector<T> &x) {
    return -(x[1] + 47) *
               std::sin(std::sqrt(std::abs(x[0] / 2 + (x[1] + 47)))) -
           x[0] * std::sin(std::sqrt(std::abs(x[0] - (x[1] + 47))));
  }
  const std::vector<T> minimum() const { return {512.0, 404.2319}; }
};
template <typename T>
struct HolderTable {
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 4; }
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
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 1; }
  T operator()(const std::vector<T> &x) {
    return std::sin(x[0] + x[1]) + std::pow(x[0] - x[1], 2) - 1.5 * x[0] +
           2.5 * x[1] + 1;
  }
  const std::vector<T> minimum() const { return {-0.54719, -1.54719}; }
};
template <typename T>
struct SchafferN2 {
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 1; }
  T operator()(const std::vector<T> &x) {
    return 0.5 + (std::pow(std::sin(x[0] * x[0] - x[1] * x[1]), 2) - 0.5) /
                     std::pow(1 + 0.001 * (x[0] * x[0] + x[1] * x[1]), 2);
  }
  const std::vector<T> minimum() const { return {0.0, 0.0}; }
};
template <typename T>
struct SchafferN4 {
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 4; }
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
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 1; }
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
  static constexpr size_t input_size() { return 4; }
  static constexpr size_t possible_minima() { return 1; }
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
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 1; }
  T operator()(const std::vector<T> &x) {
    return std::pow(x[0] + 2 * x[1] - 7, 2) + std::pow(2 * x[0] + x[1] - 5, 2);
  }
  const std::vector<T> minimum() const { return {1.0, 3.0}; }
};
template <typename T>
struct BukinN6 {
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 1; }
  T operator()(const std::vector<T> &x) {
    return 100 * std::sqrt(std::abs(x[1] - 0.01 * x[0] * x[0])) +
           0.01 * std::abs(x[0] + 10);
  }
  const std::vector<T> minimum() const { return {-10.0, 1.0}; }
};
template <typename T>
struct Matyas {
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 1; }
  T operator()(const std::vector<T> &x) {
    return 0.26 * (x[0] * x[0] + x[1] * x[1]) - 0.48 * x[0] * x[1];
  }
  const std::vector<T> minimum() const { return {0.0, 0.0}; }
};
template <typename T>
struct LeviN13 {
  static constexpr size_t input_size() { return 2; }
  static constexpr size_t possible_minima() { return 1; }
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
namespace nlsolver::testing {
template <typename T>
using xorshift = nlsolver::rng::xorshift<T>;
template <typename T>
using xoshiro = nlsolver::rng::xoshiro<T>;
template <typename T>
using recurrent = nlsolver::rng::recurrent<T>;

template <typename C>
using bfgs = nlsolver::BFGS<C, double>;
template <typename C>
using cgd = nlsolver::ConjugatedGradientDescent<C, double>;
template <typename C>
using de_xorshift_random = nlsolver::DE<C, xorshift<double>, double>;
template <typename C>
using de_xoshiro_random = nlsolver::DE<C, xoshiro<double>, double>;
template <typename C>
using de_recurrent_random = nlsolver::DE<C, recurrent<double>, double>;
template <typename C>
using de_mt_random =
    nlsolver::DE<C, mt, double, nlsolver::RecombinationStrategy::random>;
template <typename C>
using de_mt_best =
    nlsolver::DE<C, mt, double, nlsolver::RecombinationStrategy::best>;
template <typename C>
using nm = nlsolver::NelderMead<C, double>;
template <typename C>
using pso_xorshift_v =
    nlsolver::PSO<C, xorshift<double>, double, nlsolver::PSOType::Vanilla>;
template <typename C>
using pso_xoshiro_v =
    nlsolver::PSO<C, xoshiro<double>, double, nlsolver::PSOType::Vanilla>;
template <typename C>
using pso_rec_v =
    nlsolver::PSO<C, recurrent<double>, double, nlsolver::PSOType::Vanilla>;
template <typename C>
using pso_mt_v = nlsolver::PSO<C, mt, double, nlsolver::PSOType::Vanilla>;
template <typename C>
using pso_xorshift_a =
    nlsolver::PSO<C, xorshift<double>, double, nlsolver::PSOType::Accelerated>;
template <typename C>
using pso_xoshiro_a =
    nlsolver::PSO<C, xoshiro<double>, double, nlsolver::PSOType::Accelerated>;
template <typename C>
using pso_rec_a =
    nlsolver::PSO<C, recurrent<double>, double, nlsolver::PSOType::Accelerated>;
template <typename C>
using pso_mt_a = nlsolver::PSO<C, mt, double, nlsolver::PSOType::Accelerated>;
template <typename C>
using nm_pso_xorshift = nlsolver::NelderMeadPSO<C, xorshift<double>, double>;
template <typename C>
using nm_pso_xoshiro = nlsolver::NelderMeadPSO<C, xoshiro<double>, double>;
template <typename C>
using nm_pso_rec = nlsolver::NelderMeadPSO<C, recurrent<double>, double>;
template <typename C>
using nm_pso_mt = nlsolver::NelderMeadPSO<C, mt, double>;
template <typename C>
using gd_ls = nlsolver::GradientDescent<C, double,
                                        nlsolver::GradientStepType::Linesearch>;
template <typename C>
using gd_an =
    nlsolver::GradientDescent<C, double, nlsolver::GradientStepType::Anneal>;
template <typename C>
using gd_pg =
    nlsolver::GradientDescent<C, double, nlsolver::GradientStepType::PAGE>;

template <typename T>
void run_solver(T &solver, std::vector<double> &init) {
  solver.minimize(init);
}

template <typename T, typename U>
bool run_solver_on_problem(T &&solver, U &problem, const double tolerance,
                           const std::string &method_name,
                           const std::string &problem_name) {
  std::vector<double> x(problem.input_size(), -.5);
  run_solver(solver, x);
  std::vector<double> expected_minimum = problem.minimum();
  bool success = true;
  for (size_t i = 0; i < expected_minimum.size(); i++) {
    if (std::abs(x[i] - expected_minimum[i]) > tolerance) {
      success = false;
      break;
    }
  }
  // Inline ANSI color codes
  const std::string green = "\033[32m";
  const std::string red = "\033[31m";
  const std::string reset = "\033[0m";

  // Print results with solver and problem names
  if (success) {
    std::cout << green << "Solver " << method_name << " on Problem "
              << problem_name << " passed." << reset << "\n";
  } else {
    std::cout << red << "Solver " << method_name << " on Problem "
              << problem_name << " failed." << reset << "\n"
              << "Result: ";
    for (auto val : x) {
      std::cout << val << " ";
    }
    std::cout << ". Expected: ";
    for (auto val : problem.minimum()) {
      std::cout << val << " ";
    }
    std::cout << "\n" << reset;
  }
  return success;
}

template <typename T>
void invoke_solvers_on_problem(const std::string &problem_name, T &&problem,
                               const double tolerance = 0.05) {
  T prob{};
  xorshift<double> xors;
  xoshiro<double> xosh;
  recurrent<double> recc;
  mt mt_;

  auto run_solver_spec = [&](auto &&name, auto &&solver) {
    run_solver_on_problem(solver, prob, tolerance, name, problem_name);
  };
  run_solver_spec("Nelder-Mead", nm<T>(prob));
  run_solver_spec("BFGS", bfgs<T>(prob));
  run_solver_spec("Conjugate Gradient Descent", cgd<T>(prob));
  run_solver_spec("Differential evolution (random) with xorshift",
                  de_xorshift_random<T>(prob, xors));
  run_solver_spec("Differential evolution (random) with xoshiro",
                  de_xoshiro_random<T>(prob, xosh));
  run_solver_spec("Differential evolution (random) with recurrent",
                  de_recurrent_random<T>(prob, recc));
  run_solver_spec("Differential evolution (random) with mersene twister",
                  de_mt_random<T>(prob, mt_));
  run_solver_spec("Differential evolution (best) with mersene twister",
                  de_mt_best<T>(prob, mt_));
  run_solver_spec("Particle Swarm Optimization (Vanilla) xorshift",
                  pso_xorshift_v<T>(prob, xors));
  run_solver_spec("Particle Swarm Optimization (Vanilla) xoshiro",
                  pso_xoshiro_v<T>(prob, xosh));
  run_solver_spec("Particle Swarm Optimization (Vanilla) recurrent",
                  pso_rec_v<T>(prob, recc));
  run_solver_spec("Particle Swarm Optimization (Vanilla) mersene twister",
                  pso_mt_v<T>(prob, mt_));
  run_solver_spec("Particle Swarm Optimization (Accelerated) xorshift",
                  pso_xorshift_a<T>(prob, xors));
  run_solver_spec("Particle Swarm Optimization (Accelerated) xoshiro",
                  pso_xoshiro_a<T>(prob, xosh));
  run_solver_spec("Particle Swarm Optimization (Accelerated) recurrent",
                  pso_rec_a<T>(prob, recc));
  run_solver_spec("Particle Swarm Optimization (Accelerated) mersene twister",
                  pso_mt_a<T>(prob, mt_));
  run_solver_spec("Nelder-Mead Particle Swarm Optimization xorshift",
                  nm_pso_xorshift<T>(prob, xors));
  run_solver_spec("Nelder-Mead Particle Swarm Optimization xoshiro",
                  nm_pso_xoshiro<T>(prob, xosh));
  run_solver_spec("Nelder-Mead Particle Swarm Optimization recurrent",
                  nm_pso_rec<T>(prob, recc));
  run_solver_spec("Nelder-Mead Particle Swarm Optimization mersene twister",
                  nm_pso_mt<T>(prob, mt_));
  run_solver_spec("Gradient descent (line search)", gd_ls<T>(prob));
  run_solver_spec("Gradient descent (annealing)", gd_an<T>(prob));
  // run_solver_spec("Gradient descent (PAGE)", gd_pg<T>(prob));
}

// Test runner function with colored output and inline color codes
template <typename T>
void test_solvers(T tolerance) {
  invoke_solvers_on_problem("Sphere", nlsolver::test_functions::Sphere<T>());
  invoke_solvers_on_problem("Rosenbrock",
                            nlsolver::test_functions::Rosenbrock<T>());
  invoke_solvers_on_problem("Rastrigin",
                            nlsolver::test_functions::Rastrigin<T>());
  invoke_solvers_on_problem("Ackley", nlsolver::test_functions::Ackley<T>());
  invoke_solvers_on_problem("Beale", nlsolver::test_functions::Beale<T>());
  invoke_solvers_on_problem("Goldstein_Price",
                            nlsolver::test_functions::Goldstein_Price<T>());
  /*invoke_solvers_on_problem(
      "Himmelblau", nlsolver::test_functions::Himmelblau<T>());
  */
  invoke_solvers_on_problem("ThreeHumpCamel",
                            nlsolver::test_functions::ThreeHumpCamel<T>());
  /*invoke_solvers_on_problem(
      "CrossInTray", nlsolver::test_functions::CrossInTray<T>());
  invoke_solvers_on_problem(
      "Eggholder", nlsolver::test_functions::Eggholder<T>());
  */
  /*invoke_solvers_on_problem(
   * "HolderTable", nlsolver::test_functions::HolderTable<T>());
   */
  invoke_solvers_on_problem("McCormick",
                            nlsolver::test_functions::McCormick<T>());
  invoke_solvers_on_problem("SchafferN2",
                            nlsolver::test_functions::SchafferN2<T>());
  /*
  invoke_solvers_on_problem(
      "SchafferN4", nlsolver::test_functions::SchafferN4<T>());
  */
  invoke_solvers_on_problem("StyblinskiTang",
                            nlsolver::test_functions::StyblinskiTang<T>());
  invoke_solvers_on_problem("Shekel", nlsolver::test_functions::Shekel<T>());
  invoke_solvers_on_problem("Booth", nlsolver::test_functions::Booth<T>());
  invoke_solvers_on_problem("BukinN6", nlsolver::test_functions::BukinN6<T>());
  invoke_solvers_on_problem("Matyas", nlsolver::test_functions::Matyas<T>());
  invoke_solvers_on_problem("LeviN13", nlsolver::test_functions::LeviN13<T>());
}
};  // namespace nlsolver::testing

#endif  // TEST_FUNCTIONS_H_
