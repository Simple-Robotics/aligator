#include "proxddp/python/fwd.hpp"

#include "proxddp/core/stage-model.hpp"

namespace proxddp {
namespace python {

context::StageConstraint *
make_constraint_wrap(const shared_ptr<context::StageFunction> &f,
                     const shared_ptr<context::ConstraintSet> &c) {
  return new context::StageConstraint{f, c};
}

void exposeStage() {
  using context::ConstraintSet;
  using context::DynamicsModel;
  using context::Manifold;
  using context::Scalar;
  using context::StageModel;
  using StageData = StageDataTpl<Scalar>;

  using CostPtr = shared_ptr<context::CostBase>;
  using FunctionPtr = shared_ptr<context::StageFunction>;
  using ManifoldPtr = shared_ptr<Manifold>;
  using CstrSetPtr = shared_ptr<ConstraintSet>;

  bp::register_ptr_to_python<shared_ptr<StageModel>>();
  pinpy::StdVectorPythonVisitor<std::vector<StageModel>, true>::expose(
      "StdVec_StageModel");

  bp::class_<StageModel>(
      "StageModel",
      "A stage of the control problem. Holds costs, dynamics, and constraints.",
      bp::init<const ManifoldPtr &, const int, const ManifoldPtr &,
               const CostPtr &, const shared_ptr<DynamicsModel> &>(
          bp::args("self", "space1", "nu", "space2", "cost", "dyn_model")))
      .def(bp::init<const ManifoldPtr &, const int, const CostPtr &,
                    const shared_ptr<DynamicsModel> &>(
          bp::args("self", "space", "nu", "cost", "dyn_model")))
      .def("addConstraint",
           (void(StageModel::*)(const context::StageConstraint &)) &
               StageModel::addConstraint,
           bp::args("self", "constraint"),
           "Add an existing constraint to the stage.")
      .def("addConstraint",
           (void(StageModel::*)(const FunctionPtr &, const CstrSetPtr &)) &
               StageModel::addConstraint,
           bp::args("self", "func", "cstr_set"),
           "Constructs a new constraint (from the underlying function and set) "
           "and adds it to the stage.")
      .add_property("xspace",
                    bp::make_function(&StageModel::xspace,
                                      bp::return_internal_reference<>()),
                    "State space for the current state :math:`x_k`.")
      .add_property("xspace_next",
                    bp::make_function(&StageModel::xspace_next,
                                      bp::return_internal_reference<>()),
                    "State space corresponding to next state :math:`x_{k+1}`.")
      .add_property("uspace",
                    bp::make_function(&StageModel::uspace,
                                      bp::return_internal_reference<>()),
                    "Control space.")
      .add_property("cost",
                    bp::make_function(&StageModel::cost,
                                      bp::return_internal_reference<>()),
                    "Stage cost.")
      .add_property("dyn_model",
                    bp::make_function(&StageModel::dyn_model,
                                      bp::return_internal_reference<>()),
                    "Stage dynamics.")
      .def("evaluate", &StageModel::evaluate,
           bp::args("self", "x", "u", "y", "data"),
           "Evaluate the stage cost, dynamics, constraints.")
      .def("computeDerivatives", &StageModel::computeDerivatives,
           bp::args("self", "x", "u", "y", "data"),
           "Compute derivatives of the stage cost, dynamics, and constraints.")
      .add_property("ndx1", &StageModel::ndx1)
      .add_property("ndx2", &StageModel::ndx2)
      .add_property("nu", &StageModel::nu, "Control space dimension.")
      .add_property("num_primal", &StageModel::numPrimal,
                    "Number of primal variables.")
      .add_property("num_dual", &StageModel::numDual,
                    "Number of dual variables.")
      .def(CreateDataPythonVisitor<StageModel>())
      .def(ClonePythonVisitor<StageModel>())
      .def(bp::self_ns::str(bp::self));

  bp::register_ptr_to_python<shared_ptr<StageData>>();
  pinpy::StdVectorPythonVisitor<std::vector<shared_ptr<StageData>>,
                                true>::expose("StdVec_StageData");

  bp::class_<StageData>("StageData", "Data struct for StageModel objects.",
                        bp::init<const StageModel &>())
      .def_readonly("cost_data", &StageData::cost_data)
      .add_property("dyn_data", &StageData::dyn_data,
                    "Data struct for dynamics data.")
      .def_readwrite("constraint_data", &StageData::constraint_data)
      .def(ClonePythonVisitor<StageData>());

  bp::class_<context::StageConstraint>(
      "StageConstraint",
      "A stage-wise constraint, of the form :math:`c(x,u) \\leq 0 c(x,u)`.\n"
      ":param f: underlying function\n"
      ":param cs: constraint set",
      bp::no_init)
      .def("__init__",
           bp::make_constructor(make_constraint_wrap,
                                bp::default_call_policies(),
                                bp::args("func", "cstr_set")),
           "Contruct a StageConstraint from a StageFunction and a constraint "
           "set.")
      .def_readwrite("func", &context::StageConstraint::func_)
      .def_readwrite("cstr_set", &context::StageConstraint::set_);
}

} // namespace python
} // namespace proxddp
