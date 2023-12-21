/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/python/fwd.hpp"
#include "aligator/core/callback-base.hpp"

namespace aligator {
namespace python {

using context::CallbackBase;

struct CallbackWrapper : CallbackBase, bp::wrapper<CallbackBase> {
  CallbackWrapper() = default;
  void call(const WorkspaceBaseTpl<context::Scalar> &w,
            const ResultsBaseTpl<context::Scalar> &r) {
    ALIGATOR_PYTHON_OVERRIDE_PURE(void, "call", boost::cref(w), boost::cref(r));
  }
};
} // namespace python
} // namespace aligator
