#pragma once

#include <proxsuite-nlp/linesearch-base.hpp>
#include <functional>

namespace aligator {
using proxsuite::nlp::Linesearch;
using proxsuite::nlp::LinesearchStrategy;

/// @brief Nonmonotone Linesearch algorithm. Modifies the Armijo condition with
/// a moving average of function values.
/// @details This is the algorithm from Zhang and Hager, SiOpt 2004.
template <typename T> struct NonmonotoneLinesearch : Linesearch<T> {
  using typename Linesearch<T>::FunctionSample;
  using typename Linesearch<T>::Options;
  using fun_t = std::function<T(T)>;

  T run(fun_t f, T phi0, T dphi0, T &a_opt);
  NonmonotoneLinesearch(const Options &options);

  void reset() {
    mov_avg = T(0.);
    avg_weight = T(0.);
  }

  /// Weight for moving average
  T avg_eta = 0.85;
  T beta_dec = 0.5;

private:
  T mov_avg;
  T avg_weight;
};

template <typename T>
NonmonotoneLinesearch<T>::NonmonotoneLinesearch(const Options &options)
    : Linesearch<T>(options), mov_avg(0.), avg_weight(0.) {}

template <typename T>
T NonmonotoneLinesearch<T>::run(fun_t f, T phi0, T dphi0, T &a_opt) {
  const Options &opts = this->options_;
  mov_avg = avg_eta * avg_weight * mov_avg + phi0;
  avg_weight = avg_eta * avg_weight + 1;
  mov_avg /= avg_weight;

  while (a_opt > opts.alpha_min) {
    try {
      const T phia = f(a_opt);
      bool suff_decrease = phia <= mov_avg + opts.armijo_c1 * a_opt * dphi0;
      if (suff_decrease)
        return phia;
    } catch (const std::runtime_error &) {
    }
    a_opt *= beta_dec;
  }

  // then, a_opt <= opts.alpha_min
  a_opt = std::max(a_opt, opts.alpha_min);
  return f(a_opt);
}

} // namespace aligator
