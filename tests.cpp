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

#include "./nlsolver.h"
#include "./test_functions.h"

template <typename T>
bool test_solver(T& solver, std::vector<double> init = {2., 5.},
                 std::vector<double> minimum = {0., 0.},
                 std::vector<double> lower = {}, std::vector<double> upper = {},
                 const double tol = 1e-4) {
  if (upper.size() == lower.size() && upper.size() > 0) {
    solver.minimize(init, upper, lower);
    // res.print();
  } else {
    solver.minimize(init);
    // res.print();
  }
  for (size_t i = 0; i < init.size(); i++) {
    if (std::abs(init[i] - minimum[i]) > tol) return false;
  }
  return true;
}

int main() {
  using nlsolver::NelderMead;
  using nlsolver::test_functions::Rosenbrock;
  // using namespace nlsolver::rng;
  //  run optimizer on all of these functions and verify that the result is as
  //  expected
  /*
  Sphere, Rosenbrock, Rastrigin, Ackley, Beale, Goldstein_Price, Himmelblau,
  ThreeHumpCamel, CrossInTray, Eggholder, HolderTable, McCormick,
  SchafferN2, SchafferN4, StyblinskiTang, Booth, BukinN6, Matyas, LeviN13
  */
  // std::cout << "Making f" << std::endl;
  // Rosenbrock<double> f;
  // std::cout << "Getting minimum" << std::endl;
  // auto min = f.minimum();
  // std::cout << "Running f:" << f({-3.5, 2.5}) << std::endl;

  // using solver = NelderMead<Rosenbrock<double>>;
  // std::cout << "Making solver" << std::endl;
  // auto solver = NelderMead<decltype(f)>(f);

  Rosenbrock<double> f;
  auto min = f.minimum();
  auto solver = NelderMead<decltype(f)>(f);
  test_solver(solver, {2., 5.}, min);
  // std::cout << test_solver(solver, {-7,-7.}, min) << std::endl;

  // test_solver<NelderMead<Rosenbrock<double>, double>, Rosenbrock<double>>();
  // using DEStrat = nlsolver::RecombinationStrategy;
  // test_solver<DE<Rosenbrock<double>, xorshift<double>, double,
  // DEStrat::best>, Rosenbrock<double>>();
  /*
    using DEStrat = nlsolver::RecombinationStrategy;
    auto de_solver =
        DE<Rosenbrock, xorshift<double>, double, DEStrat::best>(prob, gen);
    auto pso_solver = PSO<Rosenbrock, xoshiro<double>, double>(prob, xos_gen);
    auto apso_solver =
        PSO<Rosenbrock, xoshiro<double>, double, PSOType::Accelerated>(prob,
                                                                       xos_gen);
    auto sann_solver = SANN<Rosenbrock, xoshiro<double>, double>(prob, xos_gen);
    auto nm_pso_solver =
        NelderMeadPSO<Rosenbrock, xoshiro<double>, double>(prob, xos_gen);
    auto gd_solver_fixed =
        GradientDescent<Rosenbrock, double, GDType::Fixed>(prob, 0.0005);
    auto gd_solver_linesearch =
        GradientDescent<Rosenbrock, double, GDType::Linesearch>(prob);
    auto gd_solver_bigstep =
        GradientDescent<Rosenbrock, double, GDType::Bigstep, 5, true>(prob);
    auto cgd_solver = ConjugatedGradientDescent<Rosenbrock, double>(prob);
    auto bfgs_solver = BFGS<Rosenbrock, double>(prob);
    LevenbergMarquardt<Rosenbrock, double>(prob);
  */

  /*

    "sum"_test = [] {
      expect(sum(0) == 0_i);
      expect(sum(1, 2) == 3_i);
      expect(sum(1, 2) > 0_i and 41_i == sum(40, 2));
    };
    */
}
