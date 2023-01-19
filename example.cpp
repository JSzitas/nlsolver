#include "nlsolver.h"

using nlsolver::NelderMeadSolver;
using nlsolver::DESolver;
using nlsolver::PSOSolver;
using nlsolver::rng::xorshift;
using nlsolver::rng::xoshiro;

class Rosenbrock {
public:
  double operator()(std::vector<double> &x) {
    const double t1 = x[0];
    const double t2 = (x[1] - x[0] * x[0]);
    return t1 * t1 + 100 * t2 * t2;
  }
};

void print_vector(std::vector<double> &x) {
  if( x.size() < 1 ) {
    return;
  }
  size_t i = 0;
  for(; i < x.size() - 1; i++ ) {
    std::cout << x[i] << ",";
  }
  i++;
  std::cout << x[i] << std::endl;
}

// to use any C++ standard random number generator just pass in a generator 
// functor e.g. using Mersene Twister 
#include <random>
struct std_MT {
private:
  std::mt19937 generator;
  std::uniform_real_distribution<double> distribution;
public:
  std_MT() {
    this->generator = std::mt19937();
    this->distribution = std::uniform_real_distribution<double>(0.0,1.0);
  }
  double operator()() {
    return this->distribution(this->generator);
  }
};

int main() {
  // define problem functor - in our case a variant of the rosenbrock function
  Rosenbrock prob;
  std::cout << "Nelder-Mead: " << std::endl;
  // initialize solver - passing the functor
  auto nm_solver = NelderMeadSolver<Rosenbrock>(prob);
  // initialize function arguments
  std::vector<double> nm_init = {5,7};
  auto nm_res = nm_solver.minimize(nm_init);
  // check solver status 
  nm_res.print();
  // and estimated function parameters
  print_vector(nm_init);
  
  std::cout << "Differential evolution with xorshift" << std::endl;
  // differential evolution requires a random number generator - we
  // include some, including a xorshift RNG, and a xoshiro RNG
  xorshift<double> gen;
  // again initialize solver, this time also with the RNG
  auto de_solver = DESolver<Rosenbrock, xorshift<double>, double> (prob, gen);

  std::vector<double> de_init = {5,7};;
  auto de_res = de_solver.minimize(de_init);
  de_res.print();
  print_vector(de_init);

  std::cout << "Differential evolution with std::mt19937" << std::endl;
  // using standard library random number generators
  std_MT std_gen;
  // again initialize solver, this time also with the RNG
  auto de_solver_MT = DESolver<Rosenbrock, std_MT, double> (prob, std_gen);
  // reset initial state
  de_init[0] = 5; de_init[1] = 7;
  de_res = de_solver_MT.minimize(de_init);
  de_res.print();
  print_vector(de_init);

  std::cout << "Particle Swarm Optimization with xoroshift" << std::endl;
  // using standard library random number generators
  xoshiro<double> xos_gen;
  std::vector<double> lower = {-5, -5};
  std::vector<double> upper = {5, 5};

  // again initialize solver, this time also with the RNG
  auto pso_solver = PSOSolver<Rosenbrock,
                              xoshiro<double>,
                              double> (prob, xos_gen);
  // reset initial state
  std::vector<double> pso_init = {0,0};
  auto pso_res = pso_solver.minimize(pso_init, lower, upper);
  pso_res.print();
  print_vector(pso_init);
  
  return 0;
}

