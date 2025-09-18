/// @file linesearch-base.hpp
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
/// @brief  Base structs for linesearch algorithms.
#pragma once

#include <fmt/format.h>
#include <ostream>

namespace aligator {
enum class LinesearchStrategy { ARMIJO, WOLFE };
enum class LSInterpolation { BISECTION, QUADRATIC, CUBIC };

template <typename T> struct LinesearchOptions {
  LinesearchOptions() noexcept
      : armijo_c1(1e-4)
      , wolfe_c2(0.9)
      , dphi_thresh(1e-13)
      , alpha_min(1e-6)
      , max_num_steps(20)
      , interp_type(LSInterpolation::CUBIC)
      , contraction_min(0.5)
      , contraction_max(0.8) {}
  T armijo_c1;
  T wolfe_c2;
  T dphi_thresh;
  T alpha_min;
  std::size_t max_num_steps;
  LSInterpolation interp_type;
  T contraction_min;
  T contraction_max;
  friend std::ostream &operator<<(std::ostream &oss,
                                  const LinesearchOptions &self) {
    oss << "{";
    oss << fmt::format("armijo_c1 = {:.3e}", self.armijo_c1);
    oss << ", "
        << fmt::format("contraction_min = {:.3e}", self.contraction_min);
    oss << ", "
        << fmt::format("contraction_max = {:.3e}", self.contraction_max);
    oss << "}";
    return oss;
  }
};

/// @brief Base linesearch class.
/// Design pattern inspired by Google Ceres-Solver.
template <typename T> class Linesearch {
public:
  using Options = LinesearchOptions<T>;
  explicit Linesearch(const Linesearch::Options &options) noexcept
      : options_(options) {}
  ~Linesearch() = default;

  struct FunctionSample {
    T alpha;
    T phi;
    T dphi;
    bool valid;
    FunctionSample()
        : alpha(0.)
        , phi(0.)
        , dphi(0.)
        , valid(false) {}
    FunctionSample(T a, T v)
        : alpha(a)
        , phi(v)
        , dphi(0.)
        , valid(true) {}
    FunctionSample(T a, T v, T g)
        : alpha(a)
        , phi(v)
        , dphi(g)
        , valid(true) {}
  };

  void setOptions(const Linesearch::Options &options) { options_ = options; }

  void reset() {}

  Linesearch::Options options_;
};
} // namespace aligator
