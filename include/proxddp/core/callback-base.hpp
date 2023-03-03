#pragma once

#include "proxddp/fwd.hpp"
#include <boost/any.hpp>

namespace proxddp {
namespace helpers {

/// Base callback class.
template <typename Scalar> struct base_callback {
  using Workspace = WorkspaceBaseTpl<Scalar>;
  using Results = ResultsBaseTpl<Scalar>;
  virtual void call(const Workspace &, const Results &) = 0;
  /// Call this after linesearch.
  virtual void post_linesearch_call(boost::any) {}
  virtual ~base_callback() = default;
};

} // namespace helpers
} // namespace proxddp
