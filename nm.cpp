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

#include <math.h>

#include <iostream>
#include <tuple>
#include <vector>

#include "./utils.h"

template <typename T>
void print_vector(std::vector<T> &x) {
  for (auto &val : x) {
    std::cout << val << ",";
  }
  std::cout << "\n";
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
  const scalar_t alpha, beta, gamma, eps;
  // std::vector<scalar_t> point_values;
  const size_t max_iter;

 public:
  // constructor
  explicit NelderMead<Callable, scalar_t>(Callable &f, const scalar_t alpha = 1,
                                          const scalar_t beta = 0.5,
                                          const scalar_t gamma = 2,
                                          const size_t max_iter = 500,
                                          const scalar_t eps = 1.490116e-08)
      : f(f),
        alpha(alpha),
        beta(beta),
        gamma(gamma),
        eps(eps),
        max_iter(max_iter) {}
  // minimize interface
  [[maybe_unused]] solver_status<scalar_t> minimize(std::vector<scalar_t> &x) {
    return this->solve<true, false>(x);
  }
  // maximize interface
  [[maybe_unused]] solver_status<scalar_t> maximize(std::vector<scalar_t> &x) {
    return this->solve<false, false>(x);
  }

 private:
  template <const bool minimize = true, const bool bound = false>
  solver_status<scalar_t> solve(std::vector<scalar_t> &x) {
    constexpr scalar_t f_multiplier = minimize ? 1.0 : -1.0;
    // n * n+1 matrix
    const auto n = x.size() + 1;
    auto P =
        std::vector<std::vector<scalar_t>>(n + 1, std::vector<scalar_t>(n + 1));
    // auto simplex = std::vector<scalar_t>((n+1)*n, 0.0);
    auto f_vals = std::vector<scalar_t>(n + 1, 0.0);
    size_t function_calls_used = 0;
    // set up lambda
    auto f_lam = [&](auto x) {
      function_calls_used++;
      return f_multiplier * f(x);
    };
    scalar_t f_val = f_lam(x), size = 0.0, step = 0.0;
    size_t H, j, L = 1;
    scalar_t temp, try_step, VH, VL, VR;
    const scalar_t conv_tol = eps * (std::abs(f_val) + eps);
    // function values hidden in last index of simplex
    f_vals[0] = f_val;
    // first simplex value initialized from x
    for (size_t i = 0; i < n; i++) {
      P[i][0] = x[i];
      if (0.1 * std::abs(x[i]) > step) {
        step = 0.1 * std::abs(x[i]);
      }
    }
    if (step == 0.0) step = 0.1;
    // the rest of the simplex initialization
    for (j = 1; j < n + 1; j++) {
      for (size_t i = 0; i < n; i++) {
        P[i][j] = x[i];
      }
      try_step = step;
      while (P[j - 1][j] == x[j - 1]) {
        P[j - 1][j] = x[j - 1] + try_step;
        try_step *= 10;
      }
      size += try_step;
    }
    scalar_t old_size = size;
    size_t iter = 0;
    // compute simplex values
    for (j = 0; j < n + 1; j++) {
      if (j + 1 != L) {
        for (size_t i = 0; i < n; i++) {
          x[i] = P[i][j];
        }
        f_vals[j] = f_lam(x);
      }
    }
    while (true) {
      VL = f_vals[L - 1];
      VH = VL;
      H = L;
      // finds best
      for (j = 1; j <= n + 1; j++) {
        if (j != L) {
          f_val = f_vals[j - 1];
          if (f_val < VL) {
            L = j;
            VL = f_val;
          }
          if (f_val > VH) {
            H = j;
            VH = f_val;
          }
        }
      }
      if ((iter > max_iter) || ((VH <= VL + conv_tol)) && iter > 50) {
        for (size_t i = 0; i < n - 1; i++) x[i] = P[i][L - 1];
        return solver_status(f_vals[L - 1], iter, function_calls_used);
      }
      old_size = size;
      // average of all but worst
      for (size_t i = 0; i < n; i++) {
        temp = -P[i][H - 1];
        for (j = 0; j < n + 1; j++) {
          temp += P[i][j];
        }
        P[i][n + 1] = temp / n;
      }
      for (size_t i = 0; i < n; i++) {
        x[i] = (1.0 + alpha) * P[i][n + 1] - alpha * P[i][H - 1];
      }
      f_val = f_lam(x);
      VR = f_val;
      if (VR < VL) {
        f_vals[n + 1] = f_val;
        for (size_t i = 0; i < n; i++) {
          f_val = gamma * x[i] + (1 - gamma) * P[i][n + 1];
          P[i][n + 1] = x[i];
          x[i] = f_val;
        }
        f_val = f_lam(x);
        if (f_val < VR) {
          for (size_t i = 0; i < n; i++) P[i][H - 1] = x[i];
          f_vals[H - 1] = f_val;
        } else {
          for (size_t i = 0; i < n; i++) P[i][H - 1] = P[i][n + 1];
          f_vals[H - 1] = VR;
        }
      } else {
        if (VR < VH) {
          for (size_t i = 0; i < n; i++) P[i][H - 1] = x[i];
          f_vals[H - 1] = VR;
        }
        for (size_t i = 0; i < n; i++)
          x[i] = (1 - beta) * P[i][H - 1] + beta * P[i][n + 1];
        f_val = f_lam(x);
        if (f_val < f_vals[H - 1]) {
          for (size_t i = 0; i < n; i++) P[i][H - 1] = x[i];
          f_vals[H - 1] = f_val;
        } else if (VR >= VH) {
          size = 0.0;
          for (j = 0; j < n + 1; j++) {
            if (j + 1 != L) {
              for (size_t i = 0; i < n; i++) {
                P[i][j] = beta * (P[i][j] - P[i][L - 1]) + P[i][L - 1];
                size += abs(P[i][j] - P[i][L - 1]);
              }
            }
          }
          for (j = 0; j < n + 1; j++) {
            if (j + 1 != L) {
              for (size_t i = 0; i < n; i++) x[i] = P[i][j];
              f_vals[j] = f_lam(x);
            }
          }
        }
      }
      iter++;
    }
  }
};
// heavily based on R's nmmin
template <typename Callable, typename scalar_t = double>
solver_status<scalar_t> neldermead(std::vector<scalar_t> &x, Callable &f,
                                   const scalar_t intol = 1.490116e-08,
                                   const scalar_t alpha = 1,
                                   const scalar_t beta = 0.5,
                                   const scalar_t gamma = 2,
                                   const size_t max_iter = 500) {
  scalar_t temp, trystep, VH, VL, VR;
  // n * n+1 matrix
  const auto n = x.size() + 1;
  auto P =
      std::vector<std::vector<scalar_t>>(n + 1, std::vector<scalar_t>(n + 1));
  // auto simplex = std::vector<scalar_t>((n+1)*n, 0.0);
  size_t H, j, L = 1, function_calls_used = 0;
  auto f_lam = [&](auto x) {
    function_calls_used++;
    return f(x);
  };
  scalar_t f_val = f_lam(x), size = 0.0, step = 0.0;
  const scalar_t conv_tol = intol * (std::abs(f_val) + intol);
  P[n][0] = f_val;
  for (size_t i = 0; i < n; i++) {
    P[i][0] = x[i];
    if (0.1 * std::abs(x[i]) > step) {
      step = 0.1 * std::abs(x[i]);
    }
  }
  if (step == 0.0) step = 0.1;
  for (j = 1; j < n + 1; j++) {
    for (size_t i = 0; i < n; i++) {
      P[i][j] = x[i];
    }
    trystep = step;
    while (P[j - 1][j] == x[j - 1]) {
      P[j - 1][j] = x[j - 1] + trystep;
      trystep *= 10;
    }
    size += trystep;
  }
  scalar_t old_size = size;
  bool calc_vert = true;
  size_t iter = 0;
  while (true) {
    if (iter > max_iter) {
      for (size_t i = 0; i < n - 1; i++) x[i] = P[i][L - 1];
      return solver_status(P[n][L - 1], iter, function_calls_used);
    }
    if (calc_vert) {
      for (j = 0; j < n + 1; j++) {
        if (j + 1 != L) {
          for (size_t i = 0; i < n; i++) {
            x[i] = P[i][j];
          }
          f_val = f_lam(x);
          P[n][j] = f_val;
        }
      }
      calc_vert = false;
    }
    VL = P[n][L - 1];
    VH = VL;
    H = L;
    // finds best
    for (j = 1; j <= n + 1; j++) {
      if (j != L) {
        f_val = P[n][j - 1];
        if (f_val < VL) {
          L = j;
          VL = f_val;
        }
        if (f_val > VH) {
          H = j;
          VH = f_val;
        }
      }
    }
    // checks convergence
    if (VH <= VL + conv_tol) {
      for (size_t i = 0; i < n - 1; i++) x[i] = P[i][L - 1];
      return solver_status(P[n][L - 1], iter, function_calls_used);
    }
    // average of all but worst(?)
    for (size_t i = 0; i < n; i++) {
      temp = -P[i][H - 1];
      for (j = 0; j < n + 1; j++) {
        temp += P[i][j];
      }
      P[i][n + 1] = temp / n;
    }
    for (size_t i = 0; i < n; i++) {
      x[i] = (1.0 + alpha) * P[i][n + 1] - alpha * P[i][H - 1];
    }
    f_val = f_lam(x);
    VR = f_val;
    if (VR < VL) {
      P[n][n + 1] = f_val;
      for (size_t i = 0; i < n; i++) {
        f_val = gamma * x[i] + (1 - gamma) * P[i][n + 1];
        P[i][n + 1] = x[i];
        x[i] = f_val;
      }
      f_val = f_lam(x);
      if (f_val < VR) {
        for (size_t i = 0; i < n; i++) {
          P[i][H - 1] = x[i];
        }
        P[n][H - 1] = f_val;
      } else {
        for (size_t i = 0; i < n; i++) {
          P[i][H - 1] = P[i][n + 1];
        }
        P[n][H - 1] = VR;
      }
    } else {
      if (VR < VH) {
        for (size_t i = 0; i < n; i++) {
          P[i][H - 1] = x[i];
        }
        P[n][H - 1] = VR;
      }
      for (size_t i = 0; i < n; i++) {
        x[i] = (1 - beta) * P[i][H - 1] + beta * P[i][n + 1];
      }
      f_val = f_lam(x);
      if (f_val < P[n][H - 1]) {
        for (size_t i = 0; i < n; i++) {
          P[i][H - 1] = x[i];
        }
        P[n][H - 1] = f_val;
      } else if (VR >= VH) {
        calc_vert = true;
        size = 0.0;
        for (j = 0; j < n + 1; j++) {
          if (j + 1 != L) {
            for (size_t i = 0; i < n; i++) {
              P[i][j] = beta * (P[i][j] - P[i][L - 1]) + P[i][L - 1];
              size += abs(P[i][j] - P[i][L - 1]);
            }
          }
        }
        if (size < old_size) {
          old_size = size;
        } else {
          for (size_t i = 0; i < n - 1; i++) x[i] = P[i][L - 1];
          return solver_status(P[n][L - 1], iter, function_calls_used);
        }
      }
    }
    iter++;
  }
}

int main() {
  using scalar_t = float;
  class Rosenbrock {
   public:
    double operator()(std::vector<scalar_t> &x) {
      const scalar_t t1 = 1 - x[0];
      const scalar_t t2 = (x[1] - x[0] * x[0]);
      return t1 * t1 + 100 * t2 * t2;
    }
  };
  Rosenbrock prob;
  std::vector<scalar_t> x = {2, 7};
  auto f_lam = [&]() {
    std::vector<scalar_t> x = {2, 7};
    auto res = neldermead(x, prob);
    return;
  };
  auto res = neldermead(x, prob);
  res.print();
  print_vector(x);
  benchmark<scalar_t>(f_lam, 1000);
  auto nm_solver = NelderMead<Rosenbrock, scalar_t>(prob);
  std::vector<scalar_t> x2 = {2, 7};
  auto res2 = nm_solver.minimize(x2);
  res2.print();
  print_vector(x2);
  auto f_lam2 = [&]() {
    std::vector<scalar_t> x2 = {2, 7};
    auto res2 = nm_solver.minimize(x2);
    return;
  };
  benchmark<scalar_t>(f_lam2, 1000);
}
