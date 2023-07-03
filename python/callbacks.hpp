/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "proxddp/python/fwd.hpp"
#include "proxddp/core/callback-base.hpp"

namespace proxddp {
namespace python {

using CallbackBase = helpers::base_callback<context::Scalar>;

struct CallbackWrapper : CallbackBase, bp::wrapper<CallbackBase> {
  CallbackWrapper() = default;
  void call(const WorkspaceBaseTpl<context::Scalar> &w,
            const ResultsBaseTpl<context::Scalar> &r) {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "call", boost::cref(w), boost::cref(r));
  }
};
} // namespace python
} // namespace proxddp
