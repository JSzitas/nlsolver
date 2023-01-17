#ifndef NLSOLVER
#define NLSOLVER

#include <vector>
#include <array>
#include <tuple>
#include <unordered_set>

#include <iostream>
#include "cmath"

namespace nlsolver::rng {

#include <stdint.h>
#define MAX_SIZE_64_BIT_UINT (18446744073709551615U)

uint64_t static bitwise_rotate(uint64_t x, int bits, int rotate_bits) {
  return (x << rotate_bits) | (x >> (bits - rotate_bits));
}

template <typename scalar_t = float> struct halton {
  halton<scalar_t>(){
    b=2;
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
  void set( scalar_t base ) {
    b = base;
    y = 1;
    n = 0;
    d = 1;
    x = 1;
  };
  void reset(){
    b=2;
    y = 1;
    n = 0;
    d = 1;
    x = 1;
  };
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
  void set( uint64_t x ) {
    s = x;
  };
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
  void set( uint64_t x,
            uint64_t y,
            uint64_t z,
            uint64_t t) {
    s[0] = x;
    s[1] = y;
    s[2] = z;
    s[3] = t;
  };
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
  void set( uint64_t y, uint64_t z ) {
    x[0] = y;
    x[1] = z;
  };
private:
  uint64_t x[2];
};
}

namespace nlsolver{

template <typename scalar_t = double> static inline scalar_t max_abs_vec(const std::vector<scalar_t>& x) {
  auto result = abs(x[0]);
  for(size_t i =1; i < x.size(); i++) {
    if(result < abs(x[i])) {
      result = abs(x[i]);
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
  simplex<scalar_t>( std::vector<std::vector<scalar_t>> &vals ) : vals(std::move(vals)) {}
  const size_t size() const { return this->vals.size(); }
  void replace( std::vector<scalar_t> & new_val, const size_t at ) {
    this->vals[at] = new_val;
  }
  std::vector<std::vector<scalar_t>> vals;
};

template <typename scalar_t = double> static inline void operator +=(
  std::vector<scalar_t> & a,
  std::vector<scalar_t> & b) {
  for(size_t i = 0; i < a.size(); i++) {
    a[i] += b[i];
  } 
}

template <typename scalar_t = double> static inline void operator +=(
  std::vector<scalar_t> a,
  const std::vector<scalar_t> b) {
  for(size_t i = 0; i < a.size(); i++) {
    a[i] += b[i];
  } 
}

template <typename scalar_t = double> static inline void operator /=(
  std::vector<scalar_t> & a,
  scalar_t b) {
  for(size_t i = 0; i < a.size(); i++) {
    a[i] /= b;
  } 
}

template <typename scalar_t = double> static inline std::vector<scalar_t> get_centroid(
  const simplex<scalar_t> &x,
  const size_t except ) {
  
  std::vector<scalar_t> dim_means(x.size());
  size_t i = 0;
  for(; i < except; i++ ) {
    dim_means += x.vals[i];
  }
  i = except+1;
  for(; i < x.size(); i++ ) {
    dim_means += x.vals[i];
  }
  for( auto &val:dim_means ) {
    val /= (scalar_t)i;
  }
  return dim_means;
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
private:
  size_t iteration, function_calls_used;
  scalar_t function_value;
};

template <class Callable, typename scalar_t = double> class NelderMeadSolver {
private:
  Callable f;
  const scalar_t step, alpha, gamma, rho, sigma;
  scalar_t eps;
  std::vector<scalar_t> point_values;
  size_t best_point, worst_point;
  const size_t max_iter, no_change_best_tol;
public:
  // constructor
  NelderMeadSolver<Callable, scalar_t>( Callable &f,
                                         const scalar_t step = -1,
                                         const scalar_t alpha = 1,
                                         const scalar_t gamma = 2,
                                         const scalar_t rho = 0.5,
                                         const scalar_t sigma = 0.5,
                                         const size_t max_iter = 500,
                                         const scalar_t eps = 10e-4,
                                         const size_t no_change_best_tol = 100) : f(f), step(step),
                                         alpha(alpha), gamma(gamma), rho(rho),
                                         sigma(sigma), eps(eps),
                                         max_iter(max_iter), no_change_best_tol(no_change_best_tol) {}
  // minimize interface
  solver_status<scalar_t> minimize( std::vector<scalar_t> &x) {
    return this->solve<true>(x);
  }
  // maximize interface
  solver_status<scalar_t> maximize( std::vector<scalar_t> &x) {
    return this->solve<false>(x);
  }
private:
  template <const bool minimize = true> solver_status<scalar_t> 
  solve( std::vector<scalar_t> & x) {
    // set up simplex
    simplex<scalar_t> current_simplex( x, this->step);
    std::vector<scalar_t> scores(current_simplex.size());
    /* this basically ensures that for minimization we are seeking
     * minimum of function **f**, and for maximization we are seeking minimum of 
     * **-f** - and the compiler should hopefully treat this fairly well  
     */
    constexpr scalar_t f_multiplier = minimize ? 1 : -1;
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
    
    scalar_t ref_score = 0, exp_score = 0, cont_score = 0;
    // simplex iteration 
    while(true) {
      // find best, worst and second worst scores
      best = 0;
      worst = 0;
      second_worst = 0;
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
      if( iter >= this->max_iter || (std_err(scores) < this->eps) ||
          no_change_iter >= this->no_change_best_tol) {
        x = current_simplex.vals[best];
        return solver_status<scalar_t>(scores[best], iter, function_calls_used); 
      }
      iter++;
      // compute centroid of all points except for the worst one 
      centroid = get_centroid(current_simplex, worst);
      // reflect worst point 
      reflect(current_simplex.vals[worst], centroid, temp_reflect, this->alpha);
      // score reflected point
      ref_score = f_multiplier * f(temp_reflect);
      function_calls_used++;
      // if reflected point is better than second worst, but not better than the best 
      if( ref_score >= scores[best] && ref_score < scores[second_worst]) {
        current_simplex.replace(temp_reflect, worst);
        // otherwise if this is the best score so far, expand
      } else if( ref_score < scores[best] ) {
        transform(temp_reflect, centroid, temp_expand, this->gamma);
        // obtain score for expanded point 
        exp_score = f_multiplier * f(temp_expand);
        function_calls_used++;
        // if this is better than the expanded point score, replace the worst point 
        // with the expanded point, otherwise replace it with the reflected point 
        current_simplex.replace( exp_score < ref_score ? temp_expand : temp_reflect , worst );
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
          current_simplex.replace(temp_contract, worst);
          scores[worst] = cont_score;
          // otherwise shrink 
        } else {
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

template <class RNG, typename scalar_t = double> static inline std::vector<scalar_t>
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

template <class RNG, typename scalar_t = double> static inline
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
template <class RNG> size_t generate_index(const size_t max, RNG & generator) {
  // a slightly evil typecast
  return (size_t)(generator() * max);
}

template <class RNG> static inline std::array<size_t,4> generate_indices( 
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

template <class RNG, typename scalar_t = double> static inline void propose_new_agent( 
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

enum Recombination { best=1, random=2 };

template <class Callable, class RNG, typename scalar_t = double> class DESolver {
private:
  Callable f;
  RNG generator;
  const scalar_t crossover_prob, differential_weight, eps; 
  const size_t pop_size, max_iter;
  const Recombination recomb;
public:
  // constructor
  DESolver<Callable, RNG, scalar_t>( 
      Callable &f,
      RNG &generator,
      const scalar_t crossover_prob = 0.9,
      const scalar_t differential_weight = 0.8,
      const scalar_t eps = 10e-4,
      const size_t pop_size = 100,
      const size_t max_iter = 10000,
      Recombination recomb = random) : 
  f(f), generator(generator), crossover_prob(crossover_prob),
  differential_weight(differential_weight), eps(eps), pop_size(pop_size),
  max_iter(max_iter), recomb(recomb) {}
  // minimize interface
  solver_status<scalar_t> minimize( std::vector<scalar_t> &x) {
    if(this->recomb == random) {
      return this->solve<true, random>(x);
    }
    return this->solve<true, best>(x);
  }
  // maximize interface
  solver_status<scalar_t> maximize( std::vector<scalar_t> &x) {
    if(this->recomb == random) {
      return this->solve<false, random>(x);
    }
    return this->solve<false, best>(x);
  }
private:
  template <const bool minimize=true, Recombination recomb = random>
  solver_status<scalar_t> solve( std::vector<scalar_t> & x) {

    std::vector<std::vector<scalar_t>> agents = init_agents(x,
                                                            this->generator,
                                                            this->pop_size);
    std::array<size_t,4> new_indices = {0,0,0,0};
    constexpr scalar_t f_multiplier = minimize ? 1 : -1;
    std::vector<scalar_t> proposal_temp(x.size());
    
    std::vector<scalar_t> scores(agents.size());
    // evaluate all randomly generated agents 
    for( size_t i = 0; i < agents.size(); i++ ) {
      scores[i] =  f_multiplier * this->f(agents[i]);
    }
    size_t function_calls_used = agents.size();
    scalar_t score = 0;
    size_t iter = 0; 
    size_t best_id = 0;
    while(true) {
      // if agents have stabilized, return
      if( iter >= this->max_iter || std_err(scores) < this->eps ) {
        for( size_t i = 1; i < scores.size(); i++) {
          if( scores[i] < scores[best_id] ) {
            best_id = i;
          }
        }
        x = agents[best_id];
        return solver_status<scalar_t>(scores[best_id], iter, function_calls_used);
      }
      for( size_t i = 0; i < agents.size(); i++) {
        // generate agent indices - either using the best or the current agent
        if constexpr(recomb == random) {
          new_indices = generate_indices( i, agents.size(), this->generator);
        }
        if constexpr(recomb == best) {
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
        if(scores[i] > score) {
          for( size_t j = 0; j < proposal_temp.size(); j++ ) {
            agents[i][j] = proposal_temp[j];
          }
        }
      }
      // only for strategy best do we need to keep track of how good all solutions are 
      if constexpr(recomb == best) {
        for( size_t i = 0; i < scores.size(); i++) {
          if( scores[i] < scores[best_id] ) {
            best_id = i;
          }
        }
      }
      // increment iteration counter
      iter++;
    }
  }
};

template <class Callable, class RNG, typename scalar_t = double> class CMAESSolver {
private:
  Callable f;
  const size_t pop_size;
  const scalar_t crossover_prob, differential_weight; 
public:
  // constructor
  CMAESSolver<Callable, RNG, scalar_t>( Callable &f){}
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
};

#endif
