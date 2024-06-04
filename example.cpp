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
using nlsolver::NelderMeadPSO;

// also Levenberg - Marquardt for non-linear least squares
using nlsolver::LevenbergMarquardt;

// experimental solvers
// uni-variate optimizer
using nlsolver::experimental::Brent;

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
/*
void print_vector(std::vector<double> &x) {
  for (auto &val : x) {
    std::cout << val << ",";
  }
  std::cout << "\n";
}
 */

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
template <typename scalar_t, const size_t size = 114>
struct [[maybe_unused]] Guerrero {
  // this Functor's operator () effectively enables finding the Box Cox lambda
  // parameter using Guerrero's method. See below for full usage with optimizer
  // Brent
  std::array<scalar_t, size> data = {
      269,  321,  585,  871,  1475, 2821, 3928, 5943, 4950, 2577, 523,  98,
      184,  279,  409,  2285, 2685, 3409, 1824, 409,  151,  45,   68,   213,
      546,  1033, 2129, 2536, 957,  361,  377,  225,  360,  731,  1638, 2725,
      2871, 2119, 684,  299,  236,  245,  552,  1623, 3311, 6721, 4254, 687,
      255,  473,  358,  784,  1594, 1676, 2251, 1426, 756,  299,  201,  229,
      469,  736,  2042, 2811, 4431, 2511, 389,  73,   39,   49,   59,   188,
      377,  1292, 4031, 3495, 587,  105,  153,  387,  758,  1307, 3465, 6991,
      6313, 3794, 1836, 345,  382,  808,  1388, 2713, 3800, 3091, 2985, 3790,
      674,  81,   80,   108,  229,  399,  1132, 2432, 3574, 2935, 1537, 529,
      485,  662,  1000, 1590, 2657, 3396};
  std::array<scalar_t, size> mu_ests, sigma_ests;
  const size_t period = 12;
  const size_t num_buckets = ceil((scalar_t)data.size() / period);
  scalar_t operator()(const scalar_t x) {  // x == lambda
    // split data into buckets defined by period and compute averages
    for (size_t i = 0; i < num_buckets; i++) {
      for (size_t j = 0; j < period; j++) {
        // i indexes the seasonal period - the bucket
        // j index step within seasonal period
        // sum within the estimate
        mu_ests[i] = mu_ests[i] + data[(i * period) + j];
      }
      // now divide this by the number of seasonal periods to get the mean for
      // mus
      mu_ests[i] = mu_ests[i] / period;
    }
    // now compute the standard errors
    for (size_t i = 0; i < num_buckets; i++) {
      for (size_t j = 0; j < period; j++) {
        // i indexes the seasonal period - the bucket
        // j index step within seasonal period
        // sum the squares within the estimate
        sigma_ests[i] =
            sigma_ests[i] + std::pow(data[(i * period) + j] - mu_ests[i], 2);
      }
      // now divide this by the number of seasonal periods - 1 ( the N-1 formula
      // for variance ) (optionally 1 in case we only have a period of 1)
      sigma_ests[i] =
          sigma_ests[i] / std::max(period - 1, static_cast<size_t>(1));
      // finally we need to take the square root of this
      sigma_ests[i] = sqrt(sigma_ests[i]);
    }
    // now compute the ratios
    for (size_t i = 0; i < num_buckets; i++) {
      // we can very happily reuse the mu_ests without having to allocate more
      // memory
      mu_ests[i] = sigma_ests[i] / std::pow(mu_ests[i], 1.0 - x);
    }
    // compute standard deviation divided by the mean
    scalar_t final_mu = 0.0, final_sigma = 0.0;
    for (size_t i = 0; i < num_buckets; i++) {
      final_mu = final_mu + mu_ests[i];
      final_sigma = final_sigma + std::pow(mu_ests[i], 2);
    }
    final_mu = final_mu / static_cast<scalar_t>(num_buckets);
    final_sigma = final_sigma / static_cast<scalar_t>(num_buckets);
    // subtract mean
    final_sigma = final_sigma - std::pow(final_mu, 2);
    final_sigma = std::sqrt(final_sigma);
    return final_sigma / final_mu;
  }
};

int main() {
  // define problem functor - in our case a variant of the rosenbrock function
  Rosenbrock prob;
  std::cout << "Nelder-Mead: " << std::endl;
  // initialize solver - passing the functor
  auto nm_solver = NelderMead<Rosenbrock>(prob);
  run_solver(nm_solver, {2, 7});

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

  // uni-variate optimization example
  std::cout << "Brent for finding the BoxCox transform lambda parameter"
            << std::endl;
  Guerrero<double> box_cox_lambda;
  Brent<Guerrero<double>, double> brent_solver(box_cox_lambda);
  double x = 1.2;
  auto result = brent_solver.minimize(x, -5, 5);
  result.print();
  std::cout << "x: " << x << std::endl;

  std::cout << "Levenberg-Marquardt (always requires hessian)" << std::endl;
  auto lm_solver = LevenbergMarquardt<Rosenbrock, double>(prob);
  run_solver(lm_solver, {2, 2});

  // perhaps a bit more appropriate example
  // this forms a 'mountain' where we are trying to get into the nearest
  // 'valley'
  struct Mountain {
    double operator()(std::vector<double> &x) {
      return std::pow(std::pow(x[0], 2) + x[1] - 25, 2) +
             std::pow(x[0] + std::pow(x[1], 2) - 25, 2);
    }
    double operator()(std::vector<double> &&x) {
      return std::pow(std::pow(x[0], 2) + x[1] - 25, 2) +
             std::pow(x[0] + std::pow(x[1], 2) - 25, 2);
    }
  };
  Mountain prob2;
  std::cout << "Levenberg-Marquardt (better example)" << std::endl;
  auto lm_solver2 = LevenbergMarquardt<Mountain, double>(prob2);
  run_solver(lm_solver2, {0, 0});
  return 0;
}
