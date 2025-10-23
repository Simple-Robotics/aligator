#pragma once

#include "aligator/fwd.hpp"

namespace aligator {

/// Base callback class.
template <typename Scalar> struct CallbackBaseTpl {
  using Workspace = WorkspaceBaseTpl<Scalar>;
  using Results = ResultsBaseTpl<Scalar>;
  virtual void call(const Workspace &, const Results &) = 0;
  virtual ~CallbackBaseTpl() = default;
};
} // namespace aligator
