#pragma once

#include "proxddp/python/fwd.hpp"

#include "proxddp/compat/crocoddyl/problem-wrap.hpp"

namespace proxddp {
namespace python {

void exposeCrocoddylCompat() {
  using context::Scalar;
  namespace ns_croc = ::proxddp::compat::croc;
  bp::scope croc = get_namespace("croc");
  bp::def("convertCrocoddylProblem",
          &ns_croc::convertCrocoddylProblem<context::Scalar>,
          bp::args("croc_problem"),
          "Convert a Crocoddyl problem to a ProxDDP problem.");

  using CrocActionModel = crocoddyl::ActionModelAbstractTpl<Scalar>;
  bp::class_<ns_croc::CrocActionModelWrapperTpl<Scalar>,
             bp::bases<context::StageModel>>(
      "ActionModelWrapper", "Wrapper for Crocoddyl action models.",
      bp::init<boost::shared_ptr<CrocActionModel>>(bp::args("action_model")));
}

} // namespace python
} // namespace proxddp
