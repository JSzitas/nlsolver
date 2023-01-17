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
* CVA-ES *[Work in progress]*
* Particle Swarm Optimization *[planned]*

# Example usage: 

Solving the Rosenbrock function 

```cpp
#include "nlsolver.h"

using nlsolver::NelderMeadSolver;
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

int main() {
  Rosenbrock rosenbrock_prob;
  // Nelder Mead solver
  auto problem = NelderMeadSolver<Rosenbrock>(rosenbrock_prob);

  std::vector<double> init = {0,0};
  // minimize interface - return solver status
  auto res = problem.minimize(init);
  // print solver status
  res.print();
  // print values of result 
  for( auto &val:init) {
    std::cout << val << "," << std::endl;
  }

  return 0;
}
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
  
Additionally, this library also includes a set of (pseudo)-random and (quasi)-random number generators
that also aim to get out of the way as much as possible, all of which are also immplemented as functors.   
  
# Contributing

Feel free to open an issue or create a PR if you feel that an important piece of functionality is missing!
Just keep civil, and try to keep within the spirit of the library (no dependencies, no virtual calls, )
