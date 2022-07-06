#pragma once

#include "proxddp/core/solver-workspace.hpp"
#include "proxddp/core/solver-results.hpp"

namespace proxddp {
namespace helpers {

template <typename Scalar> struct base_callback {
  virtual void call(const WorkspaceTpl<Scalar> &,
                    const ResultsTpl<Scalar> &) = 0;
  virtual ~base_callback() = default;
};

} // namespace helpers
} // namespace proxddp
