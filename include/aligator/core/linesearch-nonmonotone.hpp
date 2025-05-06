#pragma once

#include "linesearch-base.hpp"
#include <functional>

namespace aligator {

/// @brief Nonmonotone Linesearch algorithm. Modifies the Armijo condition with
/// a moving average of function values.
/// @details This is the algorithm from Zhang and Hager, SiOpt 2004.
template <typename Scalar> struct NonmonotoneLinesearch : Linesearch<Scalar> {
  using typename Linesearch<Scalar>::FunctionSample;
  using typename Linesearch<Scalar>::Options;
  using fun_t = std::function<Scalar(Scalar)>;

  Scalar run(fun_t f, Scalar phi0, Scalar dphi0, Scalar &a_opt);
  NonmonotoneLinesearch(const Options &options);

  void reset() {
    mov_avg = Scalar(0.);
    avg_weight = Scalar(0.);
  }

  /// Weight for moving average
  Scalar avg_eta = 0.85;
  Scalar beta_dec = 0.5;

private:
  Scalar mov_avg;
  Scalar avg_weight;
};

template <typename Scalar>
NonmonotoneLinesearch<Scalar>::NonmonotoneLinesearch(const Options &options)
    : Linesearch<Scalar>(options), mov_avg(0.), avg_weight(0.) {}

template <typename Scalar>
Scalar NonmonotoneLinesearch<Scalar>::run(fun_t f, Scalar phi0, Scalar dphi0,
                                          Scalar &a_opt) {
  const Options &opts = this->options_;
  mov_avg = avg_eta * avg_weight * mov_avg + phi0;
  avg_weight = avg_eta * avg_weight + 1;
  mov_avg /= avg_weight;

  while (a_opt > opts.alpha_min) {
    try {
      const Scalar phia = f(a_opt);
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
