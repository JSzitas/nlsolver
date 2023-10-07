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
using nlsolver::DE;
using nlsolver::GradientDescent;
using nlsolver::NelderMead;
using nlsolver::NelderMeadPSO;
using nlsolver::PSO;
using nlsolver::SANN;
// helper definition for GDType
using GDType = nlsolver::GradientStepType;

using nlsolver::ConjugatedGradientDescent;

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
  auto nm_solver_lambda = NelderMead(RosenbrockLambda);
  // initialize function arguments
  nm_init = {2, 7};
  // nm_init[0] = 2;
  // nm_init[1] = 7;
  auto nm_res_lambda = nm_solver_lambda.minimize(nm_init);
  // check solver status
  nm_res_lambda.print();
  // and estimated function parameters
  print_vector(nm_init);

  // use recombination strategy
  using DEStrat = nlsolver::RecombinationStrategy;
  std::cout << "Differential evolution with xorshift: " << std::endl;
  // differential evolution requires a random number generator - we
  // include some, including a xorshift RNG, and a xoshiro RNG
  xorshift<double> gen;
  // again initialize solver, this time also with the RNG
  auto de_solver =
      DE<Rosenbrock, xorshift<double>, double, DEStrat::best>(prob, gen);

  std::vector<double> de_init = {2, 7};

  auto de_res = de_solver.minimize(de_init);
  de_res.print();
  print_vector(de_init);

  std::cout << "Differential evolution with std::mt19937: " << std::endl;
  // using standard library random number generators
  std_MT std_gen;
  // again initialize solver, this time also with the RNG
  // if strategy is not specifieed, defaults to random
  auto de_solver_MT = DE<Rosenbrock, std_MT, double>(prob, std_gen);
  // reset initial state
  de_init[0] = 2;
  de_init[1] = 7;
  de_res = de_solver_MT.minimize(de_init);
  de_res.print();
  print_vector(de_init);

  std::cout << "Particle Swarm Optimization with xoshiro: " << std::endl;
  // we also have a xoshiro generator
  xoshiro<double> xos_gen;

  // add PSO Solver Type - defaults to random
  using nlsolver::PSOType;
  // again initialize solver, this time also with the RNG
  auto pso_solver = PSO<Rosenbrock, xoshiro<double>, double>(prob, xos_gen);
  // set initial state - if no bounds are given, default initial parameters are
  // taken roughly as the scale of the parameter space
  std::vector<double> pso_init = {3, 3};
  auto pso_res = pso_solver.minimize(pso_init);
  pso_res.print();
  print_vector(pso_init);
  std::cout << "Particle Swarm Optimization with xoshiro (and bounds): "
            << std::endl;
  // this tends to be much worse than not specifying bounds for PSO - so
  // we heavily recommend those:
  pso_init[0] = 0;
  pso_init[1] = 0;
  std::vector<double> pso_lower = {-1, -1};
  std::vector<double> pso_upper = {1, 1};

  pso_res = pso_solver.minimize(pso_init, pso_lower, pso_upper);
  pso_res.print();
  print_vector(pso_init);
  std::cout << "Accelerated Particle Swarm Optimization with xoshiro: "
            << std::endl;
  // we also have an accelerated version - we reset the RNG as well.
  xos_gen.reset();
  auto apso_solver =
      PSO<Rosenbrock, xoshiro<double>, double, PSOType::Accelerated>(prob,
                                                                     xos_gen);
  // set initial state - if no bounds are given, default initial parameters are
  // taken roughly as the scale of the parameter space
  std::vector<double> apso_init = {3, 3};
  auto apso_res = apso_solver.minimize(apso_init);
  apso_res.print();
  print_vector(apso_init);

  std::cout << "Simulated Annealing with xoshiro: " << std::endl;
  // we also have an accelerated version - we reset the RNG as well.
  xos_gen.reset();
  auto sann_solver = SANN<Rosenbrock, xoshiro<double>, double>(prob, xos_gen);
  // set initial state - if no bounds are given, default initial parameters are
  // taken roughly as the scale of the parameter space
  std::vector<double> sann_init = {3, 3};
  auto sann_res = sann_solver.minimize(sann_init);
  sann_res.print();
  print_vector(sann_init);

  std::cout << "NelderMead-PSO hybrid with xoshiro: " << std::endl;
  // we also have an accelerated version - we reset the RNG as well.
  xos_gen.reset();
  auto nm_pso_solver =
      NelderMeadPSO<Rosenbrock, xoshiro<double>, double>(prob, xos_gen);
  // set initial state - if no bounds are given, default initial parameters are
  // taken roughly as the scale of the parameter space
  std::vector<double> nm_pso_init = {3, 3};
  auto nm_pso_res = nm_pso_solver.minimize(nm_pso_init);
  nm_pso_res.print();
  print_vector(nm_pso_init);

  std::cout << "Gradient Descent without line-search using fixed steps: "
            << std::endl;
  auto gd_solver_fixed =
      GradientDescent<Rosenbrock, double, GDType::Bigstep>(prob);
  std::vector<double> gd_init_fixed = {2, 2};
  auto gd_res_fixed = gd_solver_fixed.minimize(gd_init_fixed);
  gd_res_fixed.print();
  print_vector(gd_init_fixed);

  std::cout << "Gradient Descent with line-search: " << std::endl;
  auto gd_solver_linesearch =
      GradientDescent<Rosenbrock, double, GDType::Linesearch>(prob);
  std::vector<double> gd_init_linesearch = {2, 2};
  auto gd_res_linesearch = gd_solver_linesearch.minimize(gd_init_linesearch);
  gd_res_linesearch.print();
  print_vector(gd_init_linesearch);

  std::cout
      << "Gradient Descent without line-search using big steps,"
      << " cycling through step-sizes (and lipschitz constant eyeballing): "
      << std::endl;
  auto gd_solver_bigstep =
      GradientDescent<Rosenbrock, double, GDType::Bigstep, 5, true>(prob);
  std::vector<double> gd_init_bigstep = {2, 2};
  auto gd_res_bigstep = gd_solver_bigstep.minimize(gd_init_bigstep);
  gd_res_bigstep.print();
  print_vector(gd_init_bigstep);

  std::cout << "Conjugated Gradient Descent (always requires linesearch)"
            << std::endl;
  auto cgd_solver = ConjugatedGradientDescent<Rosenbrock, double>(prob);
  std::vector<double> cgd_init = {2, 2};
  auto cgd_res = cgd_solver.minimize(cgd_init);
  cgd_res.print();
  print_vector(cgd_init);

  return 0;
}
