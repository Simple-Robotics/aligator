/// @file
/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/fwd.hpp"

#include <functional>

namespace aligator {

/**
 * @brief   Filter class.
 */
template <typename Scalar> struct FilterTpl {
public:
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  // List of all filter pairs
  std::vector<std::pair<Scalar, Scalar>> filter_pairs_;

  // Proximity parameter
  Scalar beta_;
  Scalar alpha_min_;
  std::size_t max_num_steps_;

  FilterTpl(const Scalar &beta, const Scalar &alpha_min, const std::size_t &max_num_steps) {
    beta_ = beta;
    alpha_min_ = alpha_min;
    max_num_steps_ = max_num_steps;
    filter_pairs_.clear();
  }

  virtual ~FilterTpl() = default;

  void resetFilter(const Scalar &beta, const Scalar &alpha_min, const std::size_t &max_num_steps) {
    beta_ = beta;
    alpha_min_ = alpha_min;
    max_num_steps_ = max_num_steps;
    filter_pairs_.clear();
  }

  Scalar run(std::function<std::pair<Scalar, Scalar>(Scalar)> phi,
             Scalar &alpha_try) {
    alpha_try = 1;
    std::pair<Scalar, Scalar> fpair;
    // Try full step, backtrack if failure
    while (true) {
      try {
        fpair = phi(alpha_try);
        break;
      } catch (const std::runtime_error &e) {
        alpha_try *= 0.5;
        if (alpha_try <= alpha_min_) {
          alpha_try = alpha_min_;
          break;
        }
      }
    }

    // Try to accept pair, backtrack if failure
    for (std::size_t i = 0; i < max_num_steps_; i++) {
      if (!accept_pair(fpair)) {
        alpha_try *= 0.5;
        if (alpha_try <= alpha_min_) {
          alpha_try = alpha_min_;
          fpair = phi(alpha_try);
          break;
        }
        fpair = phi(alpha_try);
      }
      else break;
    }
    
    // TODO: else, feasilibity restauration by minimizing h
    return fpair.first;
  }

  bool accept_pair(const std::pair<Scalar, Scalar> &fpair) {
    // Check if pair is acceptable to the filter
    for (auto el = filter_pairs_.begin(); el != filter_pairs_.end(); el++) {
      std::pair<Scalar, Scalar> element = *el;
      if (element.first - beta_ * element.second <= fpair.first and
          element.second - beta_ * element.second <= fpair.second) {
        return false;
      }
    }

    // If acceptable, remove all pairs dominated by it
    for (auto el = filter_pairs_.begin(); el != filter_pairs_.end();) {
      std::pair<Scalar, Scalar> element = *el;
      if (fpair.first <= element.first and fpair.second <= element.second) {
        el = filter_pairs_.erase(el);
      } else {
        el++;
      }
    }

    // Push new pair inside filter
    filter_pairs_.push_back(fpair);

    return true;
  }
};
} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./filter.txx"
#endif
