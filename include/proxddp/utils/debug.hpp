#pragma once

#include <vector>

namespace proxddp {


template <typename Fn, typename T>
std::vector<T> plot_linesearch_function(Fn phi, T amax, std::size_t nsteps) {
  std::vector<T> alphas;
  const T da = amax / nsteps;
  for (std::size_t i = 0; i <= nsteps; i++) {
    alphas.push_back(phi(da * i));
  }
  return alphas;
}


} // namespace proxddp

