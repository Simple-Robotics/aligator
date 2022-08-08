#pragma once

#include "proxddp/fwd.hpp"

namespace proxddp {
namespace helpers {

template <typename Scalar> struct base_callback {
  using Workspace = WorkspaceBaseTpl<Scalar>;
  using Results = ResultsBaseTpl<Scalar>;
  virtual void call(const Workspace &, const Results &) = 0;
  virtual ~base_callback() = default;
};

} // namespace helpers
} // namespace proxddp
