# A simple, (single) header only C++17 library of Nonlinear Numerical Optimizers

The main goal of this library is to be as easy to use as possible. For this reason, 
the library has no third party dependencies, requires no complex includes, 
no additional magic. Just copy the header into your project, include and happily use:

```cpp
#include "nlsolver.h"
```

# Solvers: 

* Nelder-Mead 
* Differential Evolution 
* Particle Swarm Optimization (vanilla)
* Particle Swarm Optimization (Accelerated) *[planned]*
* CVA-ES *[Work in progress]*

# Roadmap 

The goal going forward would be to implement two kinds of constructs for constrained 
optimization (even though the algorithms implemented here might not be premium solvers 
for such cases): 

* Fixed constraints (upper and lower bounds on acceptable parameter values)
* Inequality constraints (e.g. $\phi_1 + \phi_2 < 3$) - this will almost surely be implemented
as a functor

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

Or run all examples from the commmand line with **make**:
```{bash}
make example
```


# Design notes

There are some design decisions in this library which warrant discussion: 

* the objective functions to minimize/ maximize are always passed as functors, 
requiring an overloaded **public** **()** operator which takes a **std::vector<T>**, e.g. 
```cpp
struct example {
  double operator()( std::vector<double> &x) {
    return (x[0] + x[1])/(1-x[2]);
  }
};
```

* there are no virtual calls in this library - thus incurring no performance penalty
* each function exposes both a minimization and a maximization interface, and maximization is 
  always implemented as 'negative minimization', i.e. by multiply the objective function by -1
* all optimizers come with default arguments that try to be sane and user friendly - but expert 
  users are highly encouraged to supply their own values
* currently no multi-threading is supported - this is by design as functors are *potentially*
  stateful objective functions and multi-threading would require ensuring no data races happen 
  due to this stateful nature. 
  
Additionally, this library also includes a set of (pseudo)-random and (quasi)-random number generators
that also aim to get out of the way as much as possible, all of which are also implemented as functors. 
The library thus assumes that functors are used for random number generation - we demonstrate how to use 
standard library random number generators if one chooses to do so. 
  
# Contributing

Feel free to open an issue or create a PR if you feel that an important piece of functionality is missing!
Just keep civil, and try to keep within the spirit of the library (no dependencies, no virtual calls, all minimizers 
take functors as objective functions, and objective functions are impure). 
