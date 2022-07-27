#pragma once

#include "proxddp/fwd.hpp"

namespace proxddp {
namespace helpers {

template <typename Scalar> struct base_callback {
  virtual void call(const SolverProxDDP<Scalar> *, const WorkspaceTpl<Scalar> &,
                    const ResultsTpl<Scalar> &) = 0;
  virtual ~base_callback() = default;
};

} // namespace helpers
} // namespace proxddp
