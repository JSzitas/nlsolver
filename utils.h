// Tiny QR solver, header only library
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (C) 2023- Juraj Szitas
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

#ifndef UTILS_H_
#define UTILS_H_

#include <chrono>  // NOLINT [build/c++11]
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <queue>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

/* graciously taken from: https://stackoverflow.com/a/61881422
 * this is quite convenient, because to time a block of code you simply call
 * the constructor, and when the block finishes it will be automatically
 * cleaned up (and that will give you the timing).
 */
template <const bool loud = false,
          typename Resolution = std::chrono::duration<double, std::micro>>
class Stopwatch {
  typedef std::chrono::high_resolution_clock Clock;

 private:
  std::chrono::time_point<Clock> last;

 public:
  void reset() noexcept { last = Clock::now(); }
  Stopwatch() noexcept { reset(); }
  auto operator()() const noexcept {  // returns time in Resolution
    return Resolution(Clock::now() - last).count();
  }
  ~Stopwatch() {
    if constexpr (loud) {
      std::cout << "This code took: " << (*this)() * 1e-6 << " seconds.\n";
    }
  }
};

template <typename scalar_t>
std::vector<scalar_t> read_vec(std::string file) {
  std::ifstream input_stream(file);
  std::istream_iterator<scalar_t> start(input_stream), end;
  return std::vector<scalar_t>(start, end);
}
template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

// streaming median requires two medians
template <typename T>
struct StreamingMedian {
  std::priority_queue<T, std::vector<T>, std::greater<T>> right;
  std::priority_queue<T, std::vector<T>, std::less<T>> left;
  const size_t reorder_freq;

 public:
  StreamingMedian<T>(const size_t reorder_frequency = 15)
      : reorder_freq(reorder_frequency) {
    this->right = std::priority_queue<T, std::vector<T>, std::greater<T>>();
    this->left = std::priority_queue<T, std::vector<T>, std::less<T>>();
  }
  void push_back(const T x) {
    // push onto left heap
    this->left.push(x);
    // periodically call reorder
    if (this->left.size() > (this->right.size() + reorder_freq)) {
      reorder();
    }
  }
  const T value() {
    reorder();
    if (left.size() == right.size())
      return (this->left.top() + this->right.top()) / 2;
    // otherwise I know the right heap holds the median
    return this->right.top();
  }

 private:
  // reorder elements between heaps
  void reorder() {
    // this moves elements from the left heap to the right heap
    // if left + right is even, we will take an average of two tops,
    // so we need to do the correct number of reorderings
    // since we only ever push to left heap, this is really simple
    while (this->left.size() > this->right.size()) {
      this->right.push(this->left.top());
      this->left.pop();
    }
  }
};

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnusedParameter"
template <typename scalar_t, typename F>
void benchmark(F& fun, const size_t max_iterations = 1000) {
  const auto& fun_ = [&]() {
    Stopwatch sw;
    fun();
    return sw();
  };
  scalar_t timing = 0.0;
  StreamingMedian<scalar_t> median;
  for (size_t i = 0; i < max_iterations; i++) {
    const scalar_t run = fun_();
    timing += run;
    median.push_back(run);
  }
  std::cout << "Average time: " << timing / max_iterations
            << " \u03BCs | Median: " << median.value()
            << " \u03BCs | Total time: " << timing << " \u03BCs" << std::endl;
}

template <typename scalar_t>
struct Benchmarker {
  const size_t max_iter;
  StreamingMedian<scalar_t> median;
  scalar_t mean = 0.0, total = 0.0, min_ = 1.0, max_ = 1.0, mean1 = 0.0,
           mean2 = 0.0;
  Benchmarker<scalar_t>(const size_t max_iterations = 1000)
      : max_iter(max_iterations) {}
  template <typename F, typename F_>
  void operator()(F& f, F_& f2) {
    const auto& fun_ = [&]() {
      Stopwatch sw;
      f();
      return sw();
    };
    const auto& fun_2 = [&]() {
      Stopwatch sw;
      f2();
      return sw();
    };
    scalar_t timing = 0.0;
    scalar_t mean_ = 0.0;
    for (size_t i = 0; i < max_iter; i++) {
      const scalar_t f_res_1 = fun_();
      const scalar_t f_res_2 = fun_2();
      mean1 += f_res_1;
      mean2 += f_res_2;
      const scalar_t run = f_res_1 / f_res_2;
      median.push_back(run);
      mean_ += run;
      total += run;
      min_ = run < min_ ? run : min_;
      max_ = run > max_ ? run : max_;
    }
    mean_ /= max_iter;
    mean1 /= max_iter;
    mean2 /= max_iter;
    mean = (mean + mean_) / 2;
  }
  void report() {
    std::cout << "Average speedup: " << mean
              << " | Median speedup: " << median.value()
              << " | Total time: " << total << " | worst speedup: " << min_
              << " | best speedup: " << max_ << "\nAverage speed v1: " << mean1
              << " | Average speed v2: " << mean2 << std::endl;
  }
};
#pragma clang diagnostic pop

template <const size_t max_iterations = 1000, typename... Ts,
          typename scalar_t = double, const bool check_identity = true>
void benchmark_versions(Ts&&... versions) {
  const auto funs = {versions...};
  // if we should try to check the identity of outputs, first figure out if we
  // have non-void returns for all functions
  // bool check = check_identity;
  if constexpr (check_identity) {
    for (const auto& fun : funs) {
      if (std::is_same<typename decltype(std::function{fun})::result_type,
                       void>::value) {
        break;
      }
    }
  }
  size_t version = 1;
  // finally run benchmarks
  for (const auto& fun : funs) {
    std::cout << "Version: " << version++ << " | ";
    benchmark<scalar_t>(fun, max_iterations);
  }
}

template <typename scalar_t>
std::vector<scalar_t> make_random_matrix(const size_t n, const size_t p,
                                         const scalar_t mean = 0.0,
                                         const scalar_t std_dev = 1.0) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution d{mean, std_dev};

  std::vector<scalar_t> result(n * p);
  for (auto& val : result) val = d(gen);
  return result;
}

#endif  // UTILS_H_
