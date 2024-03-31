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

// baseline 'tried and tested' solvers
using nlsolver::ConjugatedGradientDescent;
using nlsolver::DE;
using nlsolver::GradientDescent;
using nlsolver::NelderMead;
using nlsolver::PSO;
using nlsolver::SANN;
// helper definition for GDType
using GDType = nlsolver::GradientStepType;
using nlsolver::BFGS;

// experimental solvers
using nlsolver::experimental::LevenbergMarquardt;
using nlsolver::experimental::NelderMeadPSO;

// RNG
using nlsolver::rng::xorshift;
using nlsolver::rng::xoshiro;

// experimental solvers

class Rosenbrock {
 public:
  double operator()(std::vector<double> &x) {
    const double t1 = 1 - x[0];
    const double t2 = (x[1] - x[0] * x[0]);
    return t1 * t1 + 100 * t2 * t2;
  }
};

void print_vector(std::vector<double> &x) {
  for (auto &val : x) {
    std::cout << val << ",";
  }
  std::cout << "\n";
}

// to use any C++ standard random number generator just pass in a generator
// functor e.g. using Mersene Twister
/*
#include <cstdint>  // NOLINT
#include <random>
struct std_MT {
 private:
  std::mt19937 generator;
  std::uniform_real_distribution<double> distribution;

 public:
  std_MT() {                           // NOLINT
    this->generator = std::mt19937();  // NOLINT
    this->distribution = std::uniform_real_distribution<double>(0.0, 1.0);
  }
  double operator()() { return this->distribution(this->generator); }
};
 */

template <typename T>
void run_solver(T &solver, std::vector<double> init = {2, 5}) {
  auto de_res = solver.minimize(init);
  de_res.print();
  print_vector(init);
}
template <typename T>
void run_solver(T &solver, std::vector<double> lower, std::vector<double> upper,
                std::vector<double> init = {2, 5}) {
  auto de_res = solver.minimize(init, lower, upper);
  de_res.print();
  print_vector(init);
}

int main() {
  // define problem functor - in our case a variant of the rosenbrock function
  Rosenbrock prob;
  std::cout << "Nelder-Mead: " << std::endl;
  // initialize solver - passing the functor
  auto nm_solver = NelderMead<Rosenbrock>(prob);
  // initialize function arguments
  std::vector<double> nm_init = {2, 7};
  auto nm_res = nm_solver.minimize(nm_init);
  // check solver status
  nm_res.print();
  // and estimated function parameters
  print_vector(nm_init);

  std::cout << "Nelder-Mead using a lambda function: " << std::endl;
  // try the same with a lambda function
  auto RosenbrockLambda = [](std::vector<double> &x) {
    const double t1 = 1 - x[0];
    const double t2 = (x[1] - x[0] * x[0]);
    return t1 * t1 + 100 * t2 * t2;
  };
  auto nm_solver_lambda =
      NelderMead<decltype(RosenbrockLambda), double>(RosenbrockLambda);
  // repeat with lambda function, using a simple wrapper defined above:
  run_solver(nm_solver_lambda, {2, 7});

  // use recombination strategy
  using DEStrat = nlsolver::RecombinationStrategy;
  std::cout << "Differential evolution with xorshift: " << std::endl;
  // differential evolution requires a random number generator - we
  // include some, including a xorshift RNG, and a xoshiro RNG
  xorshift<double> gen;
  // again initialize solver, this time also with the RNG
  auto de_solver =
      DE<Rosenbrock, xorshift<double>, double, DEStrat::best>(prob, gen);
  run_solver(de_solver, {2, 7});

  std::cout << "Particle Swarm Optimization with xoshiro: " << std::endl;
  // we also have a xoshiro generator
  xoshiro<double> xos_gen;

  // add PSO Solver Type - defaults to random
  using nlsolver::PSOType;
  // again initialize solver, this time also with the RNG
  auto pso_solver = PSO<Rosenbrock, xoshiro<double>, double>(prob, xos_gen);
  // set initial state - if no bounds are given, default initial parameters are
  // taken roughly as the scale of the parameter space
  run_solver(pso_solver, {3, 3});
  std::cout << "Particle Swarm Optimization with xoshiro (and bounds): "
            << std::endl;
  // this tends to be much worse than not specifying bounds for PSO - so
  // we heavily recommend those:
  run_solver(pso_solver, {-1, -1}, {1, 1}, {0, 0});
  std::cout << "Accelerated Particle Swarm Optimization with xoshiro: "
            << std::endl;
  // we also have an accelerated version - we reset the RNG as well.
  xos_gen.reset();
  auto apso_solver =
      PSO<Rosenbrock, xoshiro<double>, double, PSOType::Accelerated>(prob,
                                                                     xos_gen);
  // set initial state - if no bounds are given, default initial parameters are
  // taken roughly as the scale of the parameter space
  run_solver(apso_solver, {3, 3});

  std::cout << "Simulated Annealing with xoshiro: " << std::endl;
  // we also have an accelerated version - we reset the RNG as well.
  xos_gen.reset();
  auto sann_solver = SANN<Rosenbrock, xoshiro<double>, double>(prob, xos_gen);
  // set initial state - if no bounds are given, default initial parameters are
  // taken roughly as the scale of the parameter space
  run_solver(sann_solver, {5, 5});

  std::cout << "NelderMead-PSO hybrid with xoshiro: " << std::endl;
  // we also have an accelerated version - we reset the RNG as well.
  xos_gen.reset();
  auto nm_pso_solver =
      NelderMeadPSO<Rosenbrock, xoshiro<double>, double>(prob, xos_gen);
  // set initial state - if no bounds are given, default initial parameters are
  // taken roughly as the scale of the parameter space
  run_solver(nm_pso_solver, {3, 3});

  std::cout << "Gradient Descent without line-search using fixed step size: "
            << std::endl;
  auto gd_solver_fixed =
      GradientDescent<Rosenbrock, double, GDType::Fixed>(prob, 0.0005);
  run_solver(gd_solver_fixed, {2, 2});

  std::cout << "Gradient Descent with line-search: " << std::endl;
  auto gd_solver_linesearch =
      GradientDescent<Rosenbrock, double, GDType::Linesearch>(prob);
  run_solver(gd_solver_linesearch, {2, 2});

  std::cout
      << "Gradient Descent without line-search using big steps,"
      << " cycling through step-sizes (and lipschitz constant eyeballing): "
      << std::endl;
  auto gd_solver_bigstep =
      GradientDescent<Rosenbrock, double, GDType::Bigstep, 5, true>(prob);
  run_solver(gd_solver_bigstep, {2, 2});

  std::cout << "Conjugated Gradient Descent (always requires linesearch)"
            << std::endl;
  auto cgd_solver = ConjugatedGradientDescent<Rosenbrock, double>(prob);
  run_solver(cgd_solver, {2, 2});

  std::cout << "BFGS (always requires linesearch)" << std::endl;
  auto bfgs_solver = BFGS<Rosenbrock, double>(prob);
  run_solver(bfgs_solver, {2, 2});

  std::cout << "LevenbergMarquardt (always requires hessian)" << std::endl;
  auto lm_solver = LevenbergMarquardt<Rosenbrock, double>(prob);
  run_solver(lm_solver, {2, 2});

  return 0;
}
