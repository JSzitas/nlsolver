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

// experimental solvers
// uni-variate optimizer
using nlsolver::experimental::rootfinder::bisection;
using nlsolver::experimental::rootfinder::brent;
using nlsolver::experimental::rootfinder::false_position;
using nlsolver::experimental::rootfinder::itp;
using nlsolver::experimental::rootfinder::ridders;
using nlsolver::experimental::rootfinder::tiruneh;

int main() {
  // uni-variate optimization example
  std::cout << "Brent" << std::endl;
  // Guerrero<double> box_cox_lambda;
  struct Problem {
    std::array<double, 2> bounds =
        //{1, 10};
        //{-2, 1};
        {-4, 8};  // 4/3};//4/3};
    double operator()(const double x) {
      // return std::pow(x,2)/12 + x -4;
      // return (x+3)*std::pow(x-1,2);
      return std::pow(x, 3) - x - 2.;
    }
    double upper() { return bounds[1]; }
    double lower() { return bounds[0]; }
  };

  Problem p;
  double x = p.lower();
  brent(p, x, p.lower(), p.upper()).print();
  std::cout << "x: " << x << std::endl;

  std::cout << "Bisection" << std::endl;
  x = p.lower();
  bisection(p, x, p.lower(), p.upper()).print();
  std::cout << "x: " << x << std::endl;

  std::cout << "False position" << std::endl;
  x = p.lower();
  false_position(p, x, p.lower(), p.upper()).print();
  std::cout << "x: " << x << std::endl;

  std::cout << "Ridders" << std::endl;
  x = p.lower();
  ridders(p, x, p.lower(), p.upper()).print();
  std::cout << "x: " << x << std::endl;

  std::cout << "Tiruneh" << std::endl;
  x = (p.upper() + p.lower()) / 2;
  tiruneh(p, x, {x - 0.1, x, x + 0.1}).print();
  std::cout << "x: " << x << std::endl;

  std::cout << "ITP" << std::endl;
  x = p.lower();
  itp(p, x, p.lower(), p.upper()).print();
  std::cout << "x: " << x << std::endl;
}
