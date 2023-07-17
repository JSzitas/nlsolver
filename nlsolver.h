// Nonlinear optimization in C++
// https://github.com/JSzitas/nlsolver
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2023- Juraj Szitas
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef NLSOLVER
#define NLSOLVER

#include <vector>
#include <array>
#include <tuple>
#include <unordered_set>
#include <optional>

#include <iostream>
#include <cmath>

namespace nlsolver{

template <typename scalar_t = double> static inline scalar_t max_abs_vec(const std::vector<scalar_t>& x) {
  auto result = std::abs(x[0]);
  scalar_t temp = 0;
  for(size_t i =1; i < x.size(); i++) {
    temp = std::abs(x[i]);
    if(result < temp) {
      result = temp;
    }
  }
  return result;
}

template <typename scalar_t = double> struct simplex {
  simplex<scalar_t>( const size_t i = 0) {
    this->vals = std::vector<std::vector<scalar_t>>(i+1);
  }
  simplex<scalar_t>( const std::vector<scalar_t> &x, const scalar_t step = -1) {
    std::vector<std::vector<scalar_t>> init_simplex(x.size() + 1);
    init_simplex[0] = x;
    // this follows Gao and Han, see:
    // 'Proper initialization is crucial for the Nelder–Mead simplex search.' (2019),
    // Wessing, S.  Optimization Letters 13, p. 847–856 
    // (also at https://link.springer.com/article/10.1007/s11590-018-1284-4)
    // default initialization
    if(step < 0) {
      // get infinity norm of initial vector
      scalar_t x_inf_norm = max_abs_vec(x);
      // if smaller than 1, set to 1
      scalar_t a = x_inf_norm < 1.0 ? 1.0 : x_inf_norm;
      // if larger than 10, set to 10
      scalar_t scale = a < 10 ? a : 10;
      for( size_t i = 1; i < init_simplex.size(); i++ ) {
        init_simplex[i] = x;
        init_simplex[i][i] = x[i] + scale;
      }
      // update first simplex point
      scalar_t n = (scalar_t)x.size();
      for( size_t i = 0; i < x.size(); i++ ) {
        init_simplex[0][i] = x[i] + ((1.0 - sqrt(n + 1.0 ))/n * scale);
      }
    }
    // otherwise, first element of simplex has unchanged starting values 
    else {
      for( size_t i = 1; i < init_simplex.size(); i++ ) {
        init_simplex[i] = x;
        init_simplex[i][i] = x[i] + step;
      }
    }
    this->vals = init_simplex;
  }
  void restart( const std::vector<scalar_t> &x, const scalar_t step = 1) {
    std::vector<std::vector<scalar_t>> init_simplex(x.size() + 1);
    init_simplex[0] = x;
    // this follows Gao and Han, see:
    // 'Proper initialization is crucial for the Nelder–Mead simplex search.' (2019),
    // Wessing, S.  Optimization Letters 13, p. 847–856 
    // (also at https://link.springer.com/article/10.1007/s11590-018-1284-4)
    // default initialization
    if(step < 0) {
      // get infinity norm of initial vector
      scalar_t x_inf_norm = max_abs_vec(x);
      // if smaller than 1, set to 1
      scalar_t a = x_inf_norm < 1.0 ? 1.0 : x_inf_norm;
      // if larger than 10, set to 10
      scalar_t scale = a < 10 ? a : 10;
      for( size_t i = 1; i < init_simplex.size(); i++ ) {
        init_simplex[i] = x;
        init_simplex[i][i] = x[i] + scale;
      }
      // update first simplex point
      scalar_t n = (scalar_t)x.size();
      for( size_t i = 0; i < x.size(); i++ ) {
        init_simplex[0][i] = x[i] + ((1.0 - sqrt(n + 1.0 ))/n * scale);
      }
    }
    // otherwise, first element of simplex has unchanged starting values 
    else {
      for( size_t i = 1; i < init_simplex.size(); i++ ) {
        init_simplex[i] = x;
        init_simplex[i][i] = x[i] + step;
      }
    }
    this->vals = init_simplex;
  }
  simplex<scalar_t>( std::vector<std::vector<scalar_t>> &vals ) : vals(std::move(vals)) {}
  const size_t size() const { return this->vals.size(); }
  void replace( std::vector<scalar_t> & new_val, const size_t at ) {
    this->vals[at] = new_val;
  }
  void replace( std::vector<scalar_t> & new_val, 
                const size_t at,
                const std::vector<scalar_t> & upper,
                const std::vector<scalar_t> & lower,
                const scalar_t inversion_eps = 0.00001) {
    for( size_t i = 0; i < new_val.size(); i++ ) {
      this->vals[at][i] = new_val[i] < lower[i] ?
                            lower[i] + inversion_eps : 
                              new_val[i] > upper[i] ?
                                upper[i] - inversion_eps :
                                  new_val[i];
    }
  }
  std::vector<std::vector<scalar_t>> vals;
};

template <typename scalar_t = double> static inline void update_centroid(
  std::vector<scalar_t> &centroid,
  const simplex<scalar_t> &x,
  const size_t except ) {
  // reset centroid - fill with 0
  std::fill(centroid.begin(), centroid.end(), 0.0);
  size_t i = 0;
  for(; i < except; i++ ) {
    for(size_t j = 0; j < centroid.size(); j++) {
      centroid[j] += x.vals[i][j];
    }
  }
  i = except+1;
  for(; i < x.size(); i++ ) {
    for(size_t j = 0; j < centroid.size(); j++) {
      centroid[j] += x.vals[i][j];
    }
  }
  for(auto &val:centroid) val /= (scalar_t)i;
  return;
}

template <typename scalar_t = double> static inline void reflect(
  const std::vector<scalar_t> &point,
  const std::vector<scalar_t> &centroid,
  std::vector<scalar_t> &result,
  const scalar_t alpha) {
  for( size_t i= 0; i < point.size(); i++) {
    result[i] = centroid[i] + alpha * (centroid[i] - point[i]);
  }
}
// note that expand and contract are the same action - thus we only define one
// function - transform
template <typename scalar_t = double> static inline void transform( 
  const std::vector<scalar_t> &point,
  const std::vector<scalar_t> &centroid,
  std::vector<scalar_t> &result,
  const scalar_t coef) {
  for( size_t i= 0; i < point.size(); i++) {
    result[i] = centroid[i] + coef * (point[i] - centroid[i]);
  }
}

template <typename scalar_t = double> static inline void shrink(
  simplex<scalar_t> &current_simplex,
  const size_t best, 
  const scalar_t sigma) {
  // take a reference to the best vector
  std::vector<scalar_t> & best_val = current_simplex.vals[best];
  for(size_t i = 0; i < best; i++) {
    // update all items in current vector using the best vector - 
    // hopefully the continguous data here can help a bit with cache 
    // locality
    for( size_t j = 0; j < best; j++) {
      current_simplex.vals[i][j] = best_val[j] + 
        sigma * (current_simplex.vals[i][j] - best_val[j]);
    }
  }
  // skip the best point - this uses separate loops so we do not have to do 
  // extra work (e.g. check i == best) which could lead to a branch misprediction
  for( size_t i = best + 1; i < current_simplex.size(); i++) {
    for( size_t j = 0; j < best; j++) {
      current_simplex.vals[i][j] = best_val[j] + 
        sigma * (current_simplex.vals[i][j] - best_val[j]);
    }
  }
}

template <typename scalar_t = double> static inline scalar_t std_err(
  const std::vector<scalar_t> & x) {
  size_t i = 0;
  scalar_t mean_val = 0, result = 0;
  for(; i < x.size(); i++) {
    mean_val += x[i];
  }
  mean_val /= (scalar_t)i;
  i = 0;
  for(; i < x.size(); i++) {
    result += pow( x[i] - mean_val, 2);
  }
  result /= (scalar_t)(i-1);
  return sqrt(result);
}

template <typename scalar_t = double> struct solver_status{
  solver_status<scalar_t>( const scalar_t f_val, 
                           const size_t iter_used,
                           const size_t f_calls_used) : function_value(f_val),
                           iteration(iter_used), function_calls_used(f_calls_used) {}
  void print() const {
    std::cout << "Function calls used: " << this->function_calls_used << std::endl;
    std::cout << "Algorithm iterations used: " << this->iteration << std::endl;
    std::cout << "With final function value of " << this->function_value << std::endl;
  }
  std::tuple<size_t, size_t, scalar_t> get_summary() const {
    return std::make_tuple(this->function_calls_used, this->iteration, this->function_value);
  }
  void add( const solver_status<scalar_t> &additional_runs ) {
    auto other = additional_runs.get_summary();
    this->function_calls_used += std::get<0>(other);
    this->iteration += std::get<1>(other);
    this->function_value = std::get<2>(other);
  }
private:
  scalar_t function_value;
  size_t iteration, function_calls_used;
};

template <typename Callable, typename scalar_t = double> class NelderMeadSolver {
private:
  Callable &f;
  const scalar_t step, alpha, gamma, rho, sigma;
  scalar_t eps;
  std::vector<scalar_t> point_values;
  size_t best_point, worst_point;
  const size_t max_iter, no_change_best_tol, restarts;
  std::optional<std::vector<scalar_t>> lower, upper;
public:
  // constructor
  NelderMeadSolver<Callable, scalar_t>( Callable &f,
                                         const scalar_t step = -1,
                                         const scalar_t alpha = 1,
                                         const scalar_t gamma = 2,
                                         const scalar_t rho = 0.5,
                                         const scalar_t sigma = 0.5,
                                         const scalar_t eps = 10e-4,
                                         const size_t max_iter = 500,
                                         const size_t no_change_best_tol = 100,
                                         const size_t restarts = 3) : f(f),
    step(step), alpha(alpha), gamma(gamma), rho(rho), sigma(sigma), eps(eps),
    max_iter(max_iter), no_change_best_tol(no_change_best_tol),
    restarts(restarts) {}
  // minimize interface
  solver_status<scalar_t> minimize( std::vector<scalar_t> &x) {
    auto res = this->solve<true, false>(x);
    for( size_t i = 0; i < this->restarts; i++ ) {
      res.add(this->solve<true, false>(x));
    }
    return res;
  }
  // minimize with known bounds interface
  solver_status<scalar_t> minimize( std::vector<scalar_t> &x,
                                    const std::vector<scalar_t> &upper,
                                    const std::vector<scalar_t> &lower) {
    this->upper = upper;
    this->lower = lower;
    auto res = this->solve<true, true>(x);
    for( size_t i = 0; i < this->restarts; i++ ) {
      res.add(this->solve<true, true>(x));
    }
    return res;
  }
  // maximize interface
  solver_status<scalar_t> maximize( std::vector<scalar_t> &x) {
    auto res = this->solve<false, false>(x);
    for( size_t i = 0; i < this->restarts; i++ ) {
      res.add(this->solve<false, false>(x));
    }
    return res;
  }
  // maximize with known bounds interface
  solver_status<scalar_t> maximize( std::vector<scalar_t> &x,
                                    const std::vector<scalar_t> &upper,
                                    const std::vector<scalar_t> &lower) {
    this->upper = upper;
    this->lower = lower;
    auto res = this->solve<false, true>(x);
    for( size_t i = 0; i < this->restarts; i++ ) {
      res.add(this->solve<false, true>(x));
    }
    return res;
  }
private:
  template <const bool minimize = true,
            const bool bound = false> solver_status<scalar_t> 
  solve( std::vector<scalar_t> & x) {
    // set up simplex
    simplex<scalar_t> current_simplex( x, this->step);
    std::vector<scalar_t> scores(current_simplex.size());
    /* this basically ensures that for minimization we are seeking
     * minimum of function **f**, and for maximization we are seeking minimum of 
     * **-f** - and the compiler should hopefully treat this fairly well  
     */
    constexpr scalar_t f_multiplier = minimize ? 1.0 : -1.0;
    // score simplex values
    for( size_t i = 0; i < current_simplex.size(); i++) {
      scores[i] = f_multiplier * f(current_simplex.vals[i]);
    }
    size_t function_calls_used = current_simplex.size();
    // set relative convergence tolerance using evaluation with initial
    // parameters
    this->eps = eps * (scores[0] * eps);
    // find best and worst score
    size_t best, worst, second_worst, last_best, no_change_iter = 0;
    
    size_t iter = 0;
    std::vector<scalar_t> centroid(x.size());
    
    std::vector<scalar_t> temp_reflect(x.size());
    std::vector<scalar_t> temp_expand(x.size());
    std::vector<scalar_t> temp_contract(x.size());
    
    scalar_t ref_score = 0, exp_score = 0, cont_score = 0, fun_std_err= 0;
    // simplex iteration 
    while(true) {
      // find best, worst and second worst scores
      best = 0;
      worst = 0;
      second_worst = 0; 
      fun_std_err = std_err(scores);
      
      for( size_t i = 1; i < scores.size(); i++ ) {
        // if function value is lower than the current smallest, 
        // this is the new best point
        if( scores[i] < scores[best]) {
          best = i;
        // otherwise if its worse than the current worst, we know it is the new 
        // worst point - and the old worst point becomes the new second worst
        } else if( scores[i] > scores[worst] ) {
          second_worst = worst;
          worst = i;
        }
      }
      // check if we changed the best value
      if( last_best == best ) {
        // if not, increment counter for last value change 
        no_change_iter++;
      } else {
        // otherwise reset counter and reassign last_change 
        no_change_iter = 0; 
        last_best = best;
      }
      // check whether we should stop - either by exceeding iterations or by 
      // reaching tolerance 
      if( iter >= this->max_iter ||
          fun_std_err < this->eps ||
          no_change_iter >= this->no_change_best_tol ) {
        x = current_simplex.vals[best];
        return solver_status<scalar_t>(scores[best], iter, function_calls_used); 
      }
      iter++;
      // compute centroid of all points except for the worst one 
      // centroid = get_centroid(current_simplex, worst);
      // update centroid of all points except for the worst one 
      update_centroid(centroid, current_simplex, worst);
      // reflect worst point 
      reflect(current_simplex.vals[worst], centroid, temp_reflect, this->alpha);
      // score reflected point
      ref_score = f_multiplier * f(temp_reflect);
      function_calls_used++;
      // if reflected point is better than second worst, but not better than the best 
      if( ref_score >= scores[best] && ref_score < scores[second_worst]) {
        if constexpr(bound) {
          current_simplex.replace(temp_reflect, worst, *this->upper, *this->lower);
        }
        if constexpr(!bound) {
          current_simplex.replace(temp_reflect, worst);
        }
        // otherwise if this is the best score so far, expand
      } else if( ref_score < scores[best] ) {
        transform(temp_reflect, centroid, temp_expand, this->gamma);
        // obtain score for expanded point 
        exp_score = f_multiplier * f(temp_expand);
        function_calls_used++;
        // if this is better than the expanded point score, replace the worst point 
        // with the expanded point, otherwise replace it with the reflected point 
        if constexpr(bound) {
          current_simplex.replace(
            exp_score < ref_score ? temp_expand : temp_reflect,
                        worst,
                        *this->upper,
                        *this->lower);
        }
        if constexpr(!bound) {
          current_simplex.replace(
            exp_score < ref_score ? temp_expand : temp_reflect,
                        worst);
        }
        
        scores[worst] = exp_score < ref_score ? exp_score : ref_score;
        // otherwise we have a point  worse than the 'second worst'
      } else {
        // contract outside
        transform( ref_score < scores[worst] ? temp_reflect : 
                    // or point is the worst point so far - contract inside
                    current_simplex.vals[worst],
                                        centroid, temp_contract, this->rho);
        cont_score = f_multiplier * f(temp_contract);
        function_calls_used++;
        // if this contraction is better than the reflected point or worst point, respectively
        if( cont_score < ( ref_score < scores[worst] ? ref_score : scores[worst]) ) {
          // replace worst point with contracted point 
          if constexpr(bound) {
            current_simplex.replace( temp_contract, worst, *this->upper, *this->lower);
          }
          if constexpr(!bound) {
            current_simplex.replace(temp_contract, worst);
          }
          scores[worst] = cont_score;
          // otherwise shrink 
        } else {
          // if we had not violated the bounds before shrinking, shrinking 
          // will not cause new violations - hence no bounds applied here
          shrink(current_simplex, best, this->sigma);
          // only in this case do we have to score again
          for( size_t i = 0; i < best; i++) {
            scores[i] = f_multiplier * f(current_simplex.vals[i]);
          }
          // we have not updated the best value - hence no need to 'rescore'
          for( size_t i = best + 1; i < current_simplex.size(); i++) {
            scores[i] = f_multiplier * f(current_simplex.vals[i]);
          }
          function_calls_used += current_simplex.size()-1;
        }
      }
    }
  }
};

template <typename RNG, typename scalar_t = double> static inline std::vector<scalar_t>
  generate_sequence(const std::vector<scalar_t> &offset,
                    RNG & generator) {
    const size_t samples = offset.size();
    std::vector<scalar_t> result(samples);
    for( size_t i = 0; i < samples;  i++ ) {
      // the -0.5 achieves centering around offset
      result[i] = (generator() - 0.5) * offset[i];
    }
    return result;
  }

template <typename RNG, typename scalar_t = double> static inline
  std::vector<std::vector<scalar_t>> init_agents( const std::vector<scalar_t> & init,
                                                  RNG &generator,
                                                  const size_t n_agents ) {
    std::vector<std::vector<scalar_t>> agents(n_agents);
    // first element of simplex is unchanged starting values 
    for( auto &agent:agents) {
      agent = generate_sequence(init, generator);
    }
    return agents;
  }
//static inline
template <typename RNG> size_t generate_index(const size_t max, RNG & generator) {
  // a slightly evil typecast
  return (size_t)(generator() * max);
}

template <typename RNG> static inline std::array<size_t,4> generate_indices( 
    const size_t fixed,
    const size_t max,
    RNG& generator ) {
  // prepase set for uniqueness checks
  std::unordered_set<size_t> used_set = {};
  // used_set.reserve(max);
  // fixed is the reference agent - hence should already be in the set
  used_set.insert(fixed);
  // set result array
  std::array<size_t,4> result;
  result[0] = fixed;
  size_t proposal;
  size_t samples = 1;
  while( true ) {    
    proposal = generate_index(max, generator);
    if( !used_set.count(proposal) ) {
      result[samples] = proposal;
      samples++;
      if(samples == 4) {
        return result;
      }
      used_set.insert(proposal);
    }
  }
}

template <typename RNG, typename scalar_t = double> static inline void propose_new_agent( 
  const std::array<size_t,4> &ids, 
  std::vector<scalar_t> & proposal,
  const std::vector<std::vector<scalar_t>> & agents,
  const scalar_t crossover_probability, 
  const scalar_t diff_weight, 
  RNG & generator) {
  // pick dimensionality to always change
  size_t dim = generate_index(proposal.size(), generator);
  for( size_t i = 0; i < proposal.size(); i++) {
    // check if we mutate 
    if(generator() < crossover_probability || i == dim) {
      proposal[i] = agents[ids[1]][i] + 
        diff_weight * (agents[ids[2]][i] - agents[ids[3]][i]);
    } else {
      // no replacement
      proposal[i] = agents[ids[0]][i];
    }
  }
}

enum RecombinationStrategy { best, random };

template <typename Callable, typename RNG,
          typename scalar_t = double,
          RecombinationStrategy RecombinationType=random> class DESolver {
private:
  Callable &f;
  RNG &generator;
  const scalar_t crossover_prob, differential_weight, eps; 
  const size_t pop_size, max_iter, best_value_no_change;
public:
  // constructor
  DESolver<Callable, RNG, scalar_t, RecombinationType>( 
      Callable &f,
      RNG &generator,
      const scalar_t crossover_prob = 0.9,
      const scalar_t differential_weight = 0.8,
      const scalar_t eps = 10e-4,
      const size_t pop_size = 100,
      const size_t max_iter = 10000,
      const size_t best_val_no_change = 50) : 
  f(f), generator(generator), crossover_prob(crossover_prob),
  differential_weight(differential_weight), eps(eps), pop_size(pop_size),
  max_iter(max_iter), best_value_no_change(best_val_no_change) {}
  // minimize interface
  solver_status<scalar_t> minimize( std::vector<scalar_t> &x) {
      return this->solve<true>(x);
  }
  // maximize interface
  solver_status<scalar_t> maximize( std::vector<scalar_t> &x) {
    return this->solve<false>(x);
  }
private:
  template <const bool minimize=true>
  solver_status<scalar_t> solve( std::vector<scalar_t> & x) {

    std::vector<std::vector<scalar_t>> agents = init_agents(x,
                                                            this->generator,
                                                            this->pop_size);
    std::array<size_t,4> new_indices = {0,0,0,0};
    constexpr scalar_t f_multiplier = minimize ? 1.0 : -1.0;
    std::vector<scalar_t> proposal_temp(x.size());
    
    std::vector<scalar_t> scores(agents.size());
    // evaluate all randomly generated agents 
    for( size_t i = 0; i < agents.size(); i++ ) {
      scores[i] =  f_multiplier * this->f(agents[i]);
    }
    size_t function_calls_used = agents.size();
    scalar_t score = 0;
    size_t iter = 0; 
    size_t best_id = 0, val_no_change = 0;
    bool not_updated = true;
    while(true) {
      not_updated = true;
      // track how good the solutions are
      for( size_t i = 0; i < scores.size(); i++) {
        if( scores[i] < scores[best_id] ) {
          best_id = i;
          not_updated = false;
        }
      }
      // if we did not update the best function value, increment counter 
      val_no_change = not_updated * (val_no_change+1);
      // if agents have stabilized, return
      if( iter >= this->max_iter || 
          val_no_change >= this->best_value_no_change || 
          std_err(scores) < this->eps ) {
        x = agents[best_id];
        return solver_status<scalar_t>(scores[best_id], iter, function_calls_used);
      }
      // main loop - this can in principle be parallelized
      for( size_t i = 0; i < agents.size(); i++) {
        // generate agent indices - either using the best or the current agent
        if constexpr( RecombinationType == random) {
          new_indices = generate_indices( i, agents.size(), this->generator);
        }
        if constexpr( RecombinationType == best) {
          new_indices = generate_indices( best_id, agents.size(), this->generator);
        }
        // create new mutate proposal
        propose_new_agent( new_indices, proposal_temp, agents, 
                           this->crossover_prob,
                           this->differential_weight, this->generator);
        // evaluate proposal
        score = f_multiplier * f(proposal_temp);
        function_calls_used++;
        // if score is better than previous score, update agent
        if(score < scores[i]) {
          for( size_t j = 0; j < proposal_temp.size(); j++ ) {
            agents[i][j] = proposal_temp[j];
          }
          scores[i] = score;
        }
      }
      // increment iteration counter
      iter++;
    }
  }
};

template <typename scalar_t, typename RNG >static inline scalar_t rnorm(
    RNG & generator) {
  // this is not a particularly good generator, but it is 'good enough' for 
  // our purposes. 
  constexpr scalar_t pi_ = 3.141593; 
  return sqrt(-2*log(generator()))*cos(2*pi_*generator());
}

enum PSOType{
  Vanilla,
  Accelerated
};

template <typename Callable, typename RNG,
          typename scalar_t=double,
          PSOType Type=Vanilla> class PSOSolver {
private:
  // user supplied
  RNG &generator;
  Callable &f;
  scalar_t init_inertia, inertia, cognitive_coef, social_coef;
  std::vector<scalar_t> lower, upper;
  // static, derived from above 
  size_t n_dim;
  // internally created
  std::vector<std::vector<scalar_t>> particle_positions;
  std::vector<std::vector<scalar_t>> particle_velocities;
  std::vector<std::vector<scalar_t>> particle_best_positions;
  std::vector<scalar_t> particle_best_values;
  std::vector<scalar_t> swarm_best_position;
  scalar_t swarm_best_value;
  // book-keeping
  size_t f_evals, val_no_change;
  // static limits 
  const size_t n_particles, max_iter, best_val_no_change;
  const scalar_t eps;
public:
  PSOSolver<Callable, RNG, scalar_t, Type>( Callable &f,
                                      RNG & generator,
                                      scalar_t inertia = 0.8,
                                      scalar_t cognitive_coef = 1.8,
                                      scalar_t social_coef = 1.8,
                                      const size_t n_particles = 10,
                                      const size_t max_iter = 5000,
                                      const size_t best_val_no_change = 50,
                                      const scalar_t eps = 10e-4) :
  generator(generator), f(f), inertia(inertia), cognitive_coef(cognitive_coef),
  social_coef(social_coef), n_particles(n_particles), max_iter(max_iter), 
  best_val_no_change(best_val_no_change), eps(eps) {
    this->particle_positions = std::vector<std::vector<scalar_t>>(this->n_particles);
    if constexpr(Type == Vanilla) {
      this->particle_velocities = std::vector<std::vector<scalar_t>>(this->n_particles);
    }  
    if constexpr(Type == Accelerated) {
      // keep track of original inertia 
      this->init_inertia = inertia;
    }
    this->particle_best_positions = std::vector<std::vector<scalar_t>>(this->n_particles);
  }
  // minimize interface
  solver_status<scalar_t> minimize( std::vector<scalar_t> &x) {
    std::vector<scalar_t> lower(x.size());
    std::vector<scalar_t> upper(x.size());
    scalar_t temp = 0;
    for( size_t i = 0; i < x.size(); i++) {
      temp = std::abs(x[i]);
      lower[i] = -temp;
      upper[i] = temp;
    }
    this->init_solver_state(lower, upper);
    return this->solve<true, false>(x);
  }
  // maximize helper 
  solver_status<scalar_t> maximize( std::vector<scalar_t> &x) {
    std::vector<scalar_t> lower(x.size());
    std::vector<scalar_t> upper(x.size());
    scalar_t temp = 0;
    for( size_t i = 0; i < x.size(); i++) {
      temp = std::abs(x[i]);
      lower[i] = -temp;
      upper[i] = temp;
    }
    this->init_solver_state(lower, upper);
    return this->solve<false, false>(x);
  }
  // minimize interface
  solver_status<scalar_t> minimize( std::vector<scalar_t> &x, 
                                    const std::vector<scalar_t> &lower,
                                    const std::vector<scalar_t> &upper) {
    this->init_solver_state(lower, upper);
    return this->solve<true, true>(x);
  }
  // maximize helper 
  solver_status<scalar_t> maximize( std::vector<scalar_t> &x,
                                    const std::vector<scalar_t> &lower,
                                    const std::vector<scalar_t> &upper) {
    this->init_solver_state(lower, upper);
    return this->solve<false, true>(x);
  }
private:
  template <const bool minimize = true, const bool constrained = true>
  solver_status<scalar_t> solve( std::vector<scalar_t> & x) {
    size_t iter = 0;
    this->update_best_positions<minimize>();
    while(true) {
      // if particles have stabilized (no improvement in objective iteration or 
      // no heterogeneity of particles) or we are over the limit, return
      if( iter >= this->max_iter ||
          val_no_change >= best_val_no_change ||
          std_err(this->particle_best_values) < this->eps ) {
        x = this->swarm_best_position;
        // best scores, iteration number and function calls used total
        return solver_status<scalar_t>(this->swarm_best_value, iter, this->f_evals);
      }
      if constexpr(Type == Vanilla) {
        // Vanilla velocity update
        this->update_velocities();
      } 
      if constexpr(Type == Accelerated) {
        // update inertia - we might want to create a nicer way to do this 
        // updating schedule... maybe a functor for it too? 
        this->inertia = pow(this->init_inertia, iter);
      }
      this->update_positions();
      if constexpr(constrained) {
        this->threshold_positions();
      }
      this->update_best_positions<minimize>();
      // increment iteration counter
      iter++;
    }
  }
  // for repeated initializations we will init solver with new bounds 
  void init_solver_state( const std::vector<scalar_t> &lower,
                          const std::vector<scalar_t> &upper) {
    this->n_dim = lower.size();
    this->upper = upper;
    this->lower= lower;
    
    this->swarm_best_value = 100000.0;
    this->f_evals = 0;
    this-> val_no_change = 0;
    // create particles 
    for( size_t i = 0; i < this->n_particles; i++ ) {
      this->particle_positions[i] = std::vector<scalar_t>(this->n_dim);
      if constexpr(Type == Vanilla) {
        this->particle_velocities[i] = std::vector<scalar_t>(this->n_dim);
      }  
      this->particle_best_positions[i] = std::vector<scalar_t>(this->n_dim);
    }
    scalar_t temp = 0;
    for( size_t i = 0; i < n_particles; i++ ) {
      for( size_t j = 0; j < this->n_dim; j++ ) {
        // update velocities and positions
        temp = std::abs(upper[j] - lower[j]);
        this->particle_positions[i][j] = lower[j] + ( (upper[j] - lower[j]) * generator());
        if constexpr(Type == Vanilla) {
          this->particle_velocities[i][j] = -temp + (generator() * temp);
        }  
        // update particle best positions 
        this->particle_best_positions[i][j] = this->particle_positions[i][j];
      }
    }
    this->particle_best_values = std::vector<scalar_t>(this->n_particles, 10000);
  }
  void update_velocities() {
    scalar_t r_p = 0, r_g = 0;
    for( size_t i=0; i < this->n_particles; i++ ) {
      for( size_t j = 0; j < this->n_dim; j++ ) {
        // generate random movements 
        r_p = generator();
        r_g = generator();
        // update current velocity for current particle - inertia update
        this->particle_velocities[i][j] = (this->inertia * this->particle_velocities[i][j]) +
          // cognitive update (moving more if futher away from 'best' position of particle)
          this->cognitive_coef * r_p * (particle_best_positions[i][j] - particle_positions[i][j]) + 
          // social update (moving more if further away from 'best' position of swarm)
          this->social_coef * r_g * (this->swarm_best_position[i] - particle_positions[i][j]);
      }
    }
  }
  void threshold_velocities() {
    for( size_t i=0; i < this->n_particles; i++ ) {
      for( size_t j = 0; j < this->n_dim; j++ ) {
        // threshold velocities between lower and upper
        this->particle_velocities[i][j] = this->particle_velocities[i][j] <
          this->lower[j] ? this->lower[j] : this->particle_velocities[i][j];
        this->particle_velocities[i][j] = this->particle_velocities[i][j] >
          this->upper[j] ? this->upper[j] : this->particle_velocities[i][j];
      }
    }
  }
  void update_positions() {
    if constexpr(Type == Vanilla) {
      for( size_t i= 0; i < this->n_particles; i++ ) {
        for( size_t j = 0; j < this->n_dim; j++ ) {
          // update positions using current velocity
          this->particle_positions[i][j] += this->particle_velocities[i][j];
        }
      }
    }
    if constexpr(Type == Accelerated) {
      for( size_t i= 0; i < this->n_particles; i++ ) {
        for( size_t j = 0; j < this->n_dim; j++ ) {
          // no need to use velocity - all can be inlined here
          this->particle_positions[i][j] = 
            this->inertia * rnorm<scalar_t>(this->generator) + 
            // discount position 
            (1 - this->cognitive_coef) * this->particle_positions[i][j] +
            // add best position 
            this->social_coef * swarm_best_position[j];
        }
      }
    }
  }
  void threshold_positions() {
    for( size_t i=0; i < this->n_particles; i++ ) {
      for( size_t j = 0; j < this->n_dim; j++ ) {
        // threshold velocities between lower and upper
        this->particle_positions[i][j] = this->particle_positions[i][j] <
          this->lower[j] ? this->lower[j] : this->particle_positions[i][j];
        this->particle_positions[i][j] = this->particle_positions[i][j] >
          this->upper[j] ? this->upper[j] : this->particle_positions[i][j];
      }
    }
  }
  template <const bool minimize = true> void update_best_positions() {
    scalar_t temp = 0;
    size_t best_index = 0;
    constexpr scalar_t f_multiplier = minimize ? 1.0 : -1.0;
    bool update_happened = false;
    for( size_t i = 0; i < this->n_particles; i++) {
      temp = f_multiplier * f(particle_positions[i]);
      this->f_evals++;
      if( temp < this->swarm_best_value ) {
        this->swarm_best_value = temp;
        // save update of swarm best position for after the loop so we do not 
        // by chance do many many copies here
        best_index = i;
        update_happened = true;
      }
      if( temp < this->particle_best_values[i] ) {
        this->particle_best_values[i] = temp;
      }
    }
    if(update_happened) {
      this->swarm_best_position = this->particle_positions[best_index];
    }
    // either increment to indicate no change in best objective value, 
    // or reset to 0
    this->val_no_change = (best_index == 0) * (this->val_no_change+1);
  }
};

template <typename Callable, typename RNG,
          typename scalar_t=double> class SANNSolver {
private:
  // user supplied
  RNG &generator;
  Callable &f;
  // book-keeping
  size_t f_evals;
  // static limits 
  const size_t max_iter, temperature_iter;
  const scalar_t temperature_max;
  public:
    SANNSolver<Callable, RNG, scalar_t>( Callable &f,
                                         RNG & generator,
                                         const size_t max_iter = 5000,
                                         const size_t temperature_iter = 10,
                                         const scalar_t temperature_max = 10.0) :
            generator(generator), f(f), f_evals(0), max_iter(max_iter),
            temperature_iter(temperature_iter), 
            temperature_max(temperature_max) {}
    // minimize interface
    solver_status<scalar_t> minimize( std::vector<scalar_t> &x) {
      return this->solve<true>(x);
    }
    // maximize helper 
    solver_status<scalar_t> maximize( std::vector<scalar_t> &x) {
      return this->solve<false>(x);
    }
private:
  template <const bool minimize = true>
  solver_status<scalar_t> solve( std::vector<scalar_t> & x) {
    constexpr scalar_t f_multiplier = minimize ? 1.0 : -1.0,
      e_minus_1 = 1.7182818; 
    scalar_t best_val = f_multiplier * this->f(x), scale = 1.0/temperature_max;
    this->f_evals++;
    const size_t n_dim = x.size();
    std::vector<scalar_t> p = x, ptry = x;
    size_t iter = 0; 
    while(true) {  
      if( iter >= this->max_iter ) {
        // best scores, iteration number and function calls used total
        return solver_status<scalar_t>(best_val, iter, this->f_evals);
      }
      // temperature annealing schedule - cooling
      const scalar_t t = temperature_max/std::log(static_cast<scalar_t>(iter) + e_minus_1);
      for(size_t j = 1; j < this->temperature_iter; j++) {
        const scalar_t current_scale = t * scale;
        // use random normal variates - this should allow user specified values 
        for (size_t i = 0; i < n_dim; i++) {
          // generate new candidate function values 
          ptry[i] = p[i] + current_scale * rnorm<scalar_t>(this->generator);
        }
        const scalar_t current_val = f_multiplier * f(ptry); 
        this->f_evals ++;
        const scalar_t difference = current_val - best_val;
        if ((difference <= 0.0) || (this->generator() < exp(-difference/t))) {
          for (size_t k = 0; k < n_dim; k++) p[k] = ptry[k];
          if (current_val <= best_val)
          {
            for (size_t k = 0; k < n_dim; k++) x[k] = p[k];
            best_val = current_val;
          }
        }
      }
      iter++;
    }
  }
};
};

namespace nlsolver::experimental {
// TODO: WIP
template <typename Callable, typename RNG, typename scalar_t = double> class CMAESSolver {
private:
  Callable &f;
  RNG &generator;
  const size_t pop_size;
  const scalar_t crossover_prob, differential_weight; 
public:
  // constructor
  CMAESSolver<Callable, RNG, scalar_t>( Callable &f, RNG & generator){}
  // minimize interface
  void minimize( std::vector<scalar_t> &x) {
    this->solve<true>(x);
  }
  // maximize helper 
  void maximize( std::vector<scalar_t> &x) {
    this->solve<false>(x);
  }
private:
  template <const bool minimize = true> void solve( std::vector<scalar_t> & x) {
    
  }
};
// TODO: this is currently a WIP based on https://github.com/PatWie/CppNumericalSolvers 
// the goal here is to simplify the original interface, and get rid of the Eigen dependency
template <typename Callable, typename scalar_t = double> class BFGS {
private:
  Callable &f;
  std::vector<scalar_t> inverse_hessian, search_direction, gradient;
public:
  // constructor
  BFGS<Callable, scalar_t>( Callable &f){}
  // minimize interface
  void minimize( std::vector<scalar_t> &x) {
    this->solve<true>(x);
  }
  // maximize helper 
  void maximize( std::vector<scalar_t> &x) {
    this->solve<false>(x);
  }
private:
  template <const bool minimize = true> void solve( std::vector<scalar_t> &x) {
    const size_t dim = x.size();
    this->inverse_hessian = std::vector<scalar_t>(dim * dim);
    this->search_direction = std::vector<scalar_t>(dim, 0.0);
    this->gradient = std::vector<scalar_t>(dim, 0.0);
    // initialize to identity matrix
    for(size_t i = 0; i < dim; i++) this->inverse_hessian[i + (i*dim)] = 1.0; 
    while(true) {
      this->step(x);
      this->update();
    }
  }
  // this should really be called "check" or happen within the loop above 
  // template <class vector_t, class hessian_t>
  // void Update(const function::State<scalar_t, vector_t, hessian_t> previous_function_state,
  //             const function::State<scalar_t, vector_t, hessian_t> current_function_state,
  //             const State &stop_state) {
  //   if( std::isnan(previous_function_state.value)) {
  //     status = Status::NaNLoss;
  //     return;
  //   }
  //   num_iterations++;
  //   f_delta =
  //     fabs(current_function_state.value - previous_function_state.value);
  //   x_delta = (current_function_state.x - previous_function_state.x)
  //                                    .template lpNorm<Eigen::Infinity>();
  //   gradient_norm =
  //   current_function_state.gradient.template lpNorm<Eigen::Infinity>();
  //   if ((stop_state.num_iterations > 0) &&
  //       (num_iterations > stop_state.num_iterations)) {
  //     status = Status::IterationLimit;
  //     return;
  //   }
  //   if ((stop_state.x_delta > 0) && (x_delta < stop_state.x_delta)) {
  //     x_delta_violations++;
  //     if (x_delta_violations >= stop_state.x_delta_violations) {
  //       status = Status::XDeltaViolation;
  //       return;
  //     }
  //   } else {
  //     x_delta_violations = 0;
  //   }
  //   if ((stop_state.f_delta > 0) && (f_delta < stop_state.f_delta)) {
  //     f_delta_violations++;
  //     if (f_delta_violations >= stop_state.f_delta_violations) {
  //       status = Status::FDeltaViolation;
  //       return;
  //     }
  //   } else {
  //     f_delta_violations = 0;
  //   }
  //   if ((stop_state.gradient_norm > 0) &&
  //       (gradient_norm < stop_state.gradient_norm)) {
  //     status = Status::GradientNormViolation;
  //     return;
  //   }
  //   if (previous_function_state.order == 2) {
  //     if ((stop_state.condition_hessian > 0) &&
  //         (condition_hessian > stop_state.condition_hessian)) {
  //       status = Status::HessianConditionViolation;
  //       return;
  //     }
  //   }
  //   status = Status::Continue;
  // }
  
  scalar_t step(std::vector<scalar_t> &x) {
    // update search direction vector using -inverse_hessian * gradient
    for(size_t j = 0; j < this->dim; j++) {
      scalar_t temp = 0.0;
      for(size_t i = 0; i < this->dim; i++) {
        temp -= this->inverse_hessian[i + (j*this->dim)] * this->gradient[i];
      }
      this->search_direction[j] = temp; 
    }
    // 
    scalar_t phi = 0.0;
    for(size_t i = 0; i < this->dim; i++) {
      phi += this->gradient[i] * this->search_direction[i];
    }
    if ((phi > 0) || std::isnan(phi)) {
      std::fill(this->inverse_hessian.begin(), this->inverse_hessian.end(), 0.0);
      // reset hessian approximation and search_direction 
      for(size_t i = 0; i < this->dim; i++) {
        this->inverse_hessian[i + (i*this->dim)] = 1.0; 
        this->search_direction[i] = -this->gradient[i];
      }
    }
    constexpr scalar_t rate = 0.0;
    // const scalar_t rate = linesearch::MoreThuente<function_t, 1>::Search(
    //   current, search_direction, function);
    // update parameters
    for(size_t i = 0; i < this->dim; i++) {
      const scalar_t temp = rate * this->search_direction[i];
      this->s[i] = temp;
      x[i] += temp;
    }
    scalar_t val = f(x);
    
    // this also contains the updated gradient - so we should update the gradient 
    // and record gradient step
    // function_state_t next = f(current.x + rate * search_direction, 1);
    
    // Update inverse Hessian estimate.
    // const vector_t s = rate * search_direction;
    // const vector_t y = next.gradient - current.gradient;
    const std::vector<scalar_t> y(x.size(), 0.0);    
    const std::vector<scalar_t> rho(x.size(), 1.0);
    // const scalar_t rho = 1.0 / y.dot(s);
    
    // update inverse hessian using some version of Sherman Morisson (though I am 
    // not sure exactly where its coming from)
    // inverse_hessian_ =
    //   inverse_hessian_ -
    //   rho * (s * (y.transpose() * inverse_hessian_) +
    //   (inverse_hessian_ * y) * s.transpose()) +
    //   rho * (rho * y.dot(inverse_hessian_ * y) + 1.0) * (s * s.transpose());
    
    return val;
  }
  void finite_gradient(std::vector<scalar_t> &x,
                       std::vector<scalar_t> &grad,
                       const int accuracy = 0) {
    // The 'accuracy' can be 0, 1, 2, 3.
    constexpr scalar_t eps = 2.2204e-6;
    static const std::array<std::vector<scalar_t>, 4> coeff = {
      {{1, -1},
      {1, -8, 8, -1},
      {-1, 9, -45, 45, -9, 1},
      {3, -32, 168, -672, 672, -168, 32, -3}}};
    static const std::array<std::vector<scalar_t>, 4> coeff2 = {
      {{1, -1},
      {-2, -1, 1, 2},
      {-3, -2, -1, 1, 2, 3},
      {-4, -3, -2, -1, 1, 2, 3, 4}}};
    static const std::array<scalar_t, 4> dd = {2, 12, 60, 840};
    
    const int innerSteps = 2 * (accuracy + 1);
    const scalar_t ddVal = dd[accuracy] * eps;
    
    for (size_t d = 0; d < x.size(); d++) {
      grad[d] = 0;
      for (int s = 0; s < innerSteps; ++s) {
        scalar_t tmp = x[d];
        x[d] += coeff2[accuracy][s] * eps;
        grad[d] += coeff[accuracy][s] * f(x);
        x[d] = tmp;
      }
      grad[d] /= ddVal;
    }
  }
  // A O(n^2) implementation of the Sherman Morisson update formula 
  template <const size_t max_size = 20> void
    sherman_morrison_update(
      std::vector<scalar_t> &A,
      const std::vector<scalar_t> &g) {
      const size_t p = g.size();
      std::array<scalar_t, max_size> temp_left, temp_right;
      scalar_t denominator = 1.0;
      // this is actually just a multiplication of 2 distinct vectors - Ag' and gA
      for(size_t j = 0; j < p; j++) {
        scalar_t tmp = 0.0, denom_tmp = 0.0;
        for(size_t i = 0; i < p; i++) {
          // dereference once and put on the stack - hopefully faster than 
          // using two dereferences via operator []
          scalar_t g_i = g[i];
          tmp += A[(i*p) + j] * g_i;
          denom_tmp += A[(j*p) + i] * g_i;
        }
        denominator += denom_tmp * g[j];
        // this is the first vector
        temp_left[j] = tmp;
        temp_right[j] = denom_tmp;
      }
      // this loop is only not superfluous since we do not know the denominator
      for(size_t j = 0; j < p; j++) {
        // likewise avoid extra dereferences via operator []
        const scalar_t tmp = temp_right[j];
        for(size_t i = 0; i < p; i++) {
          A[(p*j) + i] -= (temp_left[i] * tmp)/denominator;
        }
      }
    }
//   void armijo_search(const state_t &state,
//                      const vector_t &search_direction) {
//     constexpr scalar_t c = 0.2;
//     constexpr scalar_t rho = 0.9;
//     scalar_t alpha = 1.0;
//     scalar_t search_val = this->f(state.x + alpha * search_direction);
//     const scalar_t f_in = state.value;
//     const scalar_t cache = c * state.gradient.dot(search_direction);
//     
//     while (f > f_in + alpha * cache) {
//       alpha *= rho;
//       search_val = this->f(state.x + alpha * search_direction);
//     }
//     return alpha;
// };
};
}

namespace nlsolver::rng {

#include <stdint.h>
#define MAX_SIZE_64_BIT_UINT (18446744073709551615U)

uint64_t static bitwise_rotate(uint64_t x, int bits, int rotate_bits) {
  return (x << rotate_bits) | (x >> (bits - rotate_bits));
}

template <typename scalar_t = float> struct halton {
  halton<scalar_t>(scalar_t base = 2){
    b=base;
    y = 1;
    n=0;
    d=1;
    x = 1;
  };
  scalar_t yield(){
    x = d-n;
    if(x == 1){
      n = 1;
      d *= b;
    }
    else {
      y = d;
      while(x <= y) {
        y /= b;
        n = (b + 1) * y - x;
      }
    }
    return (scalar_t)(n/d);
  };
  scalar_t operator ()() {
    return this->yield();
  }
  void reset(){
    b=2;
    y = 1;
    n = 0;
    d = 1;
    x = 1;
  };
  std::vector<scalar_t> get_state() const {
    std::vector<scalar_t> result(5);
    result[0] = b;
    result[1] = y;
    result[2] = n;
    result[3] = d;
    result[4] = x;
    return result;
  }
  void set_state(scalar_t b, scalar_t y, scalar_t n, scalar_t d, scalar_t x) {
    this->b = b;
    this->y = y;
    this->n = n; 
    this->d = d;
    this->x = x;
  }
private:
  scalar_t b, y, n, d, x;
};


template <typename scalar_t = float> struct recurrent {
  recurrent<scalar_t>(){
    seed = 0.5;
    alpha = 0.618034;
    z = alpha + seed;
    z -= (scalar_t)(uint64_t)(z);
  };
  recurrent( scalar_t init_seed ) {
    seed = init_seed;
    alpha = 0.618034;
    z = alpha + seed;
    z -= (scalar_t)(uint64_t)(z);
  };
  scalar_t yield() {
    z = (z+alpha);
    // a slightly evil way to do z % 1 with floats
    z -= (scalar_t)(uint64_t)(z);
    return z;
  };
  scalar_t operator ()() {
    return this->yield();
  }
  void reset(){
    alpha = 0.618034;
    seed = 0.5;
    z = 0;
  };
  std::vector<scalar_t> get_state() const {
    std::vector<scalar_t> result(2);
    result[0] = alpha;
    result[1] = z;
    return result;
  }
  void set_state(scalar_t alpha = 0.618034, scalar_t z = 0) {
    this->alpha = alpha;
    this->z = z;
  }
private:
  scalar_t alpha = 0.618034, seed = 0.5, z = 0;
};

template <typename scalar_t = float> struct splitmix {
  splitmix<scalar_t>(){
    s = 12374563468;
  };
  scalar_t yield() {
    uint64_t result = (s += 0x9E3779B97f4A7C15);
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
    return (scalar_t)(result ^ (result >> 31))/(scalar_t)MAX_SIZE_64_BIT_UINT;
  };
  scalar_t operator ()() {
    return this->yield();
  }
  uint64_t yield_init() {
    uint64_t result = (s += 0x9E3779B97f4A7C15);
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
    return result ^ (result >> 31);
  };
  void set_state( uint64_t s ) {
    this->s = s;
  };
  std::vector<scalar_t> get_state() const {
    std::vector<scalar_t> result(1);
    result[0] = this->s;
    return result;
  }
private:
  uint64_t s;
};

template <typename scalar_t = float> struct xoshiro {
  xoshiro<scalar_t>(){
    splitmix<scalar_t> gn;
    s[0] = gn.yield_init();
    s[1] = s[0] >> 32;
    
    s[2] = gn.yield();
    s[3] = s[2] >> 32;
  };
  scalar_t yield(){
    uint64_t const result = s[0] + s[3];
    uint64_t const t = s[1] << 17;
    
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    
    s[2] ^= t;
    s[3] = bitwise_rotate(s[3], 64, 45);
    
    return (scalar_t)result/(scalar_t)MAX_SIZE_64_BIT_UINT;
  }
  scalar_t operator ()() {
    return this->yield();
  }
  void reset(){
    splitmix<scalar_t> gn;
    s[0] = gn.yield_init();
    s[1] = s[0] >> 32;
    
    s[2] = gn.yield();
    s[3] = s[2] >> 32;
  };
  void set_state( uint64_t x,
                  uint64_t y,
                  uint64_t z,
                  uint64_t t) {
    this->s[0] = x;
    this->s[1] = y;
    this->s[2] = z;
    this->s[3] = t;
  };
  std::vector<scalar_t> get_state() const {
    std::vector<scalar_t> result(4);
    for(size_t i = 0; i < 4; i++) {
      result[i] = this->s[i];
    }
    return result;
  }
private:
  uint64_t rol64(uint64_t x, int k)
  {
    return (x << k) | (x >> (64 - k));
  }
  uint64_t s[4];
};

template <typename scalar_t = float> struct xorshift {
  xorshift<scalar_t>() {
    splitmix<scalar_t> gn;
    x[0] = gn.yield_init();
    x[1] = x[0] >> 32;
  };
  scalar_t yield() {
    uint64_t t = x[0];
    uint64_t const s = x[1];
    x[0] = s;
    t ^= t << 23;		// a
    t ^= t >> 18;		// b -- Again, the shifts and the multipliers are tunable
    t ^= s ^ (s >> 5);	// c
    x[1] = t;
    return (scalar_t)(t + s)/(scalar_t)MAX_SIZE_64_BIT_UINT;
  };
  scalar_t operator ()() {
    return this->yield();
  }
  void reset(){
    splitmix<scalar_t> gn;
    x[0] = gn.yield_init();
    x[1] = x[0] >> 32;
  };
  void set_state( uint64_t y, uint64_t z ) {
    x[0] = y;
    x[1] = z;
  };
  std::vector<scalar_t> get_state() const {
    std::vector<scalar_t> result(2);
    for(size_t i = 0; i < 2; i++) {
      result[i] = this->x[i];
    }
    return result;
  }
private:
  uint64_t x[2];
};

template <typename scalar_t = float> struct lehmer {
  lehmer<scalar_t>() {
    splitmix<scalar_t> gn;
    x = gn.yield_init() << 64;
  };
  scalar_t yield() {
    this->x *= 0xda942042e4dd58b5;
    return this->x >> 64;
  };
  scalar_t operator ()() {
    return this->yield();
  }
  void reset(){
    splitmix<scalar_t> gn;
    x = gn.yield_init() << 64;
  };
  void set_state( uint64_t x ) {
    this->x = x;
  };
  std::vector<scalar_t> get_state() const {
    std::vector<scalar_t> result = {x};
    return result;
  }
private:
  uint64_t x;
};
}

#endif
