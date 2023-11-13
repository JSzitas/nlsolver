// Copyright (c) 2023- Juraj Szitas
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "nlsolver.h"  // NOLINT

using nlsolver::BFGS;

void print_vector(std::vector<double> &x) {
  for (auto &val : x) {
    std::cout << val << ",";
  }
  std::cout << "\n";
}

int main() {
  std::cout << "Running ARMA(8,3) without intercept in R as: \n"
            << " 'arima(lynx[1:100], order =c(8,0,3),include.mean = F)' \n"
            << " yields coefficients: 1.4956, -1.1573, 0.583, -0.2557, 0.0968, "
               "0.0556, -0.2102, 0.3656, \n"
            << "-0.4517, 0.1225, 0.0714 \n"
            << " we should hopefully reproduce those(or at least similar.)\n";
  // define problem functor - nlsolver actually supports lambdas
  const std::vector<double> y = {
      269,  321,  585,  871,  1475, 2821, 3928, 5943, 4950, 2577, 523,  98,
      184,  279,  409,  2285, 2685, 3409, 1824, 409,  151,  45,   68,   213,
      546,  1033, 2129, 2536, 957,  361,  377,  225,  360,  731,  1638, 2725,
      2871, 2119, 684,  299,  236,  245,  552,  1623, 3311, 6721, 4254, 687,
      255,  473,  358,  784,  1594, 1676, 2251, 1426, 756,  299,  201,  229,
      469,  736,  2042, 2811, 4431, 2511, 389,  73,   39,   49,   59,   188,
      377,  1292, 4031, 3495, 587,  105,  153,  387,  758,  1307, 3465, 6991,
      6313, 3794, 1836, 345,  382,  808,  1388, 2713, 3800, 3091, 2985, 3790,
      674,  81,   80,   108};
  size_t p = 8, q = 3, n = 100;
  // std::vector<double> resid(n, 0.0);
  auto arma_lam = [&](std::vector<double> &x) {
    std::array<double, 100> resid;  // NOLINT
    resid[0] = 0;
    resid[1] = 0;
    // simple ARMA(2,2) model; we fix the p and q here
    // just to simplify, even though this would not
    // be done in the real world
    int ma_offset;
    double ssq = 0.0, tmp;
    for (size_t l = p; l < n; l++) {
      ma_offset =
          std::min<size_t>(l - p, q);  // NOLINT // for testing this is fine
      tmp = y[l];
      for (size_t j = 0; j < p; j++) {
        tmp -= x[j] * y[l - j - 1];
      }
      // to offset that this is all in one vector, we need to
      // start at p and go to p + q
      for (size_t j = 0; j < ma_offset; j++) {
        tmp -= x[p + j] * resid[l - j - 1];
      }
      resid[l] = tmp;
      ssq += tmp * tmp;
    }
    return 0.5 * std::log(ssq / static_cast<double>(n));
  };

  std::vector<double> bfgs_init(p + q, 0.0);
  std::cout << "nlsolver BFGS" << std::endl;
  [&]() {
    auto bfgs_solver = BFGS<decltype(arma_lam), double>(arma_lam);
    auto bfgs_res = bfgs_solver.minimize(bfgs_init);
    print_vector(bfgs_init);
    bfgs_res.print();
  }();
  return 0;
}
