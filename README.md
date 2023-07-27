# A simple, (single) header only C++17 library of Nonlinear Numerical Optimizers

The main goal of this library is to be as easy to use as possible, while providing 
good performance, and enough flexibility to be useful. For this reason, 
the library has no third party dependencies, requires no complex includes, 
no additional magic. Just a C++17 compliant compiler and a problem to solve :)

Just copy the header into your project, include and use:

```cpp
#include "nlsolver.h"
```

# Solvers: 

* Nelder-Mead 
* Differential Evolution (see https://ieeexplore.ieee.org/document/5601760 for explanation of different strategies)
  + Random Recombination
  + Best Recombination 
* Particle Swarm Optimization 
  + Vanilla
  + Accelerated
* Simulated Annealing 
  + Currently without option for custom sample generators, only using the Markov Gaussian Kernel 
* Nelder-Mead PSO hybrid
  + Still slightly experimental - should work without much issue, but might under-perform other 
    solvers, particularly on simpler problems where the function to optimize over is quite flat
  
## Experimental 
  
## Work in progress
* CVA-ES 
* BFGS
  + probably based on existing work in e.g. https://github.com/PatWie/CppNumericalSolvers, but still 
    dependency free

# Roadmap 

The goal going forward would be to implement two kinds of constructs for constrained 
optimization (even though the algorithms implemented here might not be premium solvers 
for such cases): 

* Fixed constraints (upper and lower bounds on acceptable parameter values)
* Inequality constraints (e.g. $\phi_1 + \phi_2 < 3$) - this will almost surely be implemented
as a functor
* Even more solvers
* Performance benchmarks

# Example usage: 

Solving the Rosenbrock function 

```cpp
#include "nlsolver.h"

using nlsolver::DESolver;
// random number generators
using nlsolver::rng::xorshift;

class Rosenbrock {
public:
  double operator()(std::vector<double> &x) {
    const double t1 = x[0];
    const double t2 = (x[1] - x[0] * x[0]);
    return t1 * t1 + 100 * t2 * t2;
  }
};

// Different Evolution (DE) solver example
int main() {
  Rosenbrock rosenbrock_prob;
  // DE requires a random number generator - we include some, including a xorshift RNG:
  xorshift<double> gen;
  // initialize solver, supplying objective function and random number generator
  auto de_solver = DESolver<Rosenbrock, xorshift<double>, double> (prob, gen);
  
  // prepare initial values - for DE these work to center the generated agents
  std::vector<double> de_init = {5,7};;
  auto de_res = de_solver.minimize(de_init);
  // print solver status
  de_res.print();
  // best parameters get written back to init
  std::cout << de_init[0] << "," << std::cout << de_init[1] << std::endl;

  return 0;
}
```

Or run all examples from the command line with **make**:
```{bash}
make example
```

And run the examples from the command line
```{bash}
./example
```

# Design notes

There are some design decisions in this library which warrant discussion: 

* the objective functions to minimize/ maximize are passed as objects, prefering functors, 
requiring an overloaded **public** **()** operator which takes a **std::vector<T>**, e.g. 
```cpp
struct example {
  double operator()( std::vector<double> &x) {
    return (x[0] + x[1])/(1-x[2]);
  }
};
```
note that this will also work with lambda functions, and a struct/class is not strictly necessary.[^lambda_note] (See example usage.)

* there are no virtual calls in this library - thus incurring no performance penalty
* each function exposes both a minimization and a maximization interface, and maximization is 
  always implemented as 'negative minimization', i.e. by multiplying the objective function by -1
* all optimizers come with default arguments that try to be sane and user friendly - but expert 
  users are highly encouraged to supply their own values
* currently no multi-threading is supported - this is by design as functors are *potentially*
  stateful objective functions and multi-threading would require ensuring no data races happen
  
Additionally, this library also includes a set of (pseudo)-random and (quasi)-random number generators
that also aim to get out of the way as much as possible, all of which are implemented as functors. 
The library thus assumes that functors are used for random number generation - there is an example on 
how to use standard library random number generators if one chooses to do so. 
  
# Contributing

Feel free to open an issue or create a PR if you feel that an important piece of functionality is missing!
Just keep civil, and stay within the spirit of the library (no dependencies, no virtual calls, must support 
impure objective functions). 

# Notes

[^lambda_note]: This flexibility is included for cases where you want to implicitly bundle mutable data within 
the struct, and do not want to have to pass the data (e.g. through a pointer) to your objective function. 
This makes the overall design cleaner - if your objective function needs data, mainstains state, or 
does anything else on evaluation, you can keep the entirety of that within the struct (and even extract it 
after the solver finishes). If you do not need the functionality and you simply want to optimize some ad-hoc function, using 
a lambda is probably much simpler and cleaner. 

