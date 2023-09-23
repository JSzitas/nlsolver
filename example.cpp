#include "nlsolver.h"

// baseline 'tried and tested' solvers
using nlsolver::NelderMeadSolver;
using nlsolver::DESolver;
using nlsolver::PSOSolver;
using nlsolver::SANNSolver;
using nlsolver::rng::xorshift;
using nlsolver::rng::xoshiro;
using nlsolver::NelderMeadPSO;

// experimental solvers 
using nlsolver::experimental::GradientDescent;

class Rosenbrock {
public:
  double operator()(std::vector<double> &x) {
    const double t1 = 1 - x[0];
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
#include <cstdint>
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
  std::vector<double> nm_init = {2,7};
  auto nm_res = nm_solver.minimize(nm_init);
  // check solver status 
  nm_res.print();
  // and estimated function parameters
  print_vector(nm_init);
  
  std::cout << "Nelder-Mead using a lambda function: " << std::endl;
  // try the same with a lambda function 
  auto RosenbrockLambda = [](std::vector<double> &x){
    const double t1 = 1 - x[0];
    const double t2 = (x[1] - x[0] * x[0]);
    return t1 * t1 + 100 * t2 * t2;
  };
  auto nm_solver_lambda = NelderMeadSolver(RosenbrockLambda);
  // initialize function arguments
  nm_init = {2,7};
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
  auto de_solver = DESolver<Rosenbrock, xorshift<double>, double, DEStrat::best> (prob, gen);
  
  std::vector<double> de_init = {2,7};;
  auto de_res = de_solver.minimize(de_init);
  de_res.print();
  print_vector(de_init);
  
  std::cout << "Differential evolution with std::mt19937: " << std::endl;
  // using standard library random number generators
  std_MT std_gen;
  // again initialize solver, this time also with the RNG
  // if strategy is not specifieed, defaults to random 
  auto de_solver_MT = DESolver<Rosenbrock, std_MT, double> (prob, std_gen);
  // reset initial state
  de_init[0] = 2; de_init[1] = 7;
  de_res = de_solver_MT.minimize(de_init);
  de_res.print();
  print_vector(de_init);
  
  std::cout << "Particle Swarm Optimization with xoshiro: " << std::endl;
  // we also have a xoshiro generator
  xoshiro<double> xos_gen;
  
  // add PSO Solver Type - defaults to random 
  using nlsolver::PSOType;
  // again initialize solver, this time also with the RNG
  auto pso_solver = PSOSolver<Rosenbrock,
                              xoshiro<double>,
                              double> (prob, xos_gen);
  // set initial state - if no bounds are given, default initial parameters are 
  // taken roughly as the scale of the parameter space
  std::vector<double> pso_init = {3,3};
  auto pso_res = pso_solver.minimize(pso_init);
  pso_res.print();
  print_vector(pso_init);
  std::cout << "Particle Swarm Optimization with xoshiro (and bounds): " << std::endl;
  // this tends to be much worse than not specifying bounds for PSO - so 
  // we heavily recommend those: 
  pso_init[0] = 0;
  pso_init[1] = 0;
  std::vector<double> pso_lower = {-1,-1};
  std::vector<double> pso_upper = {1,1};
  
  pso_res = pso_solver.minimize(pso_init, pso_lower, pso_upper);
  pso_res.print();
  print_vector(pso_init);
  std::cout << "Accelerated Particle Swarm Optimization with xoshiro: " << std::endl;
  // we also have an accelerated version - we reset the RNG as well. 
  xos_gen.reset();
  auto apso_solver = PSOSolver<Rosenbrock,
                               xoshiro<double>,
                               double,
                               PSOType::Accelerated> (prob, xos_gen);
  // set initial state - if no bounds are given, default initial parameters are 
  // taken roughly as the scale of the parameter space
  std::vector<double> apso_init = {3,3};
  auto apso_res = apso_solver.minimize(apso_init);
  apso_res.print();
  print_vector(apso_init);
  
  std::cout << "Simulated Annealing with xoshiro: " << std::endl;
  // we also have an accelerated version - we reset the RNG as well. 
  xos_gen.reset();
  auto sann_solver = SANNSolver<Rosenbrock,
                                xoshiro<double>,
                                double> (prob, xos_gen);
  // set initial state - if no bounds are given, default initial parameters are 
  // taken roughly as the scale of the parameter space
  std::vector<double> sann_init = {3,3};
  auto sann_res = sann_solver.minimize(sann_init);
  sann_res.print();
  print_vector(sann_init);
  
  std::cout << "NelderMead-PSO hybrid with xoshiro: " << std::endl;
  // we also have an accelerated version - we reset the RNG as well. 
  xos_gen.reset();
  auto nm_pso_solver = NelderMeadPSO<Rosenbrock,
                                     xoshiro<double>,
                                     double> (prob, xos_gen);
  // set initial state - if no bounds are given, default initial parameters are 
  // taken roughly as the scale of the parameter space
  std::vector<double> nm_pso_init = {3,3};
  auto nm_pso_res = nm_pso_solver.minimize(nm_pso_init);
  nm_pso_res.print();
  print_vector(nm_pso_init);
  
  std::cout << "Gradient Descent without linesearch: " << std::endl;
  auto gd_solver = GradientDescent<Rosenbrock, double, false> (prob, 0.0001, 100);
  std::vector<double> gd_init = {2,2};
  auto gd_res = gd_solver.minimize(gd_init);
  gd_res.print();
  print_vector(gd_init);
  
  return 0;
}

