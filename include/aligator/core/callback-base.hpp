#pragma once

#include "aligator/fwd.hpp"
#include <boost/any.hpp>

namespace aligator {

/// Base callback class.
template <typename Scalar> struct CallbackBaseTpl {
  using Workspace = WorkspaceBaseTpl<Scalar>;
  using Results = ResultsBaseTpl<Scalar>;
  virtual void call(const Workspace &, const Results &) = 0;
  /// Call this after linesearch.
  virtual void post_linesearch_call(boost::any) {}
  virtual ~CallbackBaseTpl() = default;
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/callback-base.txx"
#endif
