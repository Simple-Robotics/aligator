/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/fwd.hpp"
#include "proxddp/python/costs.hpp"

#include "proxddp/modelling/quad-costs.hpp"
#include "proxddp/modelling/composite-costs.hpp"
#include "proxddp/modelling/sum-of-costs.hpp"

namespace proxddp {
namespace python {

void exposeCosts() {
  using context::MatrixXs;
  using context::Scalar;
  using context::StageFunction;
  using context::VectorXs;
  using CostData = CostDataAbstractTpl<Scalar>;

  bp::register_ptr_to_python<shared_ptr<context::CostBase>>();

  bp::class_<internal::PyCostFunction<>, boost::noncopyable>(
      "CostAbstract", "Base class for cost functions.",
      bp::init<const int, const int>(bp::args("self", "ndx", "nu")))
      .def("evaluate", bp::pure_virtual(&context::CostBase::evaluate),
           bp::args("self", "x", "u", "data"), "Evaluate the cost function.")
      .def("computeGradients", bp::pure_virtual(&context::CostBase::evaluate),
           bp::args("self", "x", "u", "data"),
           "Compute the cost function gradients.")
      .def("computeHessians",
           bp::pure_virtual(&context::CostBase::computeHessians),
           bp::args("self", "x", "u", "data"),
           "Compute the cost function hessians.")
      .add_property("ndx", &context::CostBase::ndx)
      .add_property("nu", &context::CostBase::nu)
      .def(CreateDataPythonVisitor<context::CostBase>());

  bp::class_<CostStackTpl<Scalar>, bp::bases<context::CostBase>>(
      "CostStack",
      bp::init<const int, const int,
               const std::vector<shared_ptr<context::CostBase>> &,
               const std::vector<Scalar> &>((
          bp::arg("self"), bp::arg("ndx"), bp::arg("nu"),
          bp::arg("components") = bp::list(), bp::arg("weights") = bp::list())))
      .def_readonly("components", &CostStackTpl<Scalar>::components_,
                    "Components of this cost stack.")
      .def_readonly("weights", &CostStackTpl<Scalar>::weights_,
                    "Weights of this cost stack.")
      .def("addCost", &CostStackTpl<Scalar>::addCost,
           (bp::arg("self"), bp::arg("cost"), bp::arg("weight") = 1.),
           "Add a cost to the stack of costs.")
      .def("size", &CostStackTpl<Scalar>::size,
           "Get the number of cost components.")
      .def(CopyableVisitor<CostStackTpl<Scalar>>())
      .def(CreateDataPythonVisitor<CostStackTpl<Scalar>>());

  bp::class_<SumCostDataTpl<Scalar>, bp::bases<context::CostData>>(
      "CostStackData", "Data struct for costs.", bp::no_init)
      .add_property(
          "sub_cost_data",
          bp::make_getter(&SumCostDataTpl<Scalar>::sub_cost_data,
                          bp::return_value_policy<bp::return_by_value>()));

  bp::class_<QuadraticCost<Scalar>, bp::bases<context::CostBase>>(
      "QuadraticCost", "Quadratic cost in both state and control.",
      bp::init<const MatrixXs &, const MatrixXs &, const VectorXs &,
               const VectorXs &>(
          bp::args("self", "w_x", "w_u", "interp_x", "interp_u")))
      .def(bp::init<const MatrixXs &, const MatrixXs &>(
          "Constructor with just weights.", bp::args("self", "w_x", "w_u")))
      .def(CopyableVisitor<QuadraticCost<Scalar>>());

  bp::class_<CostData, shared_ptr<CostData>>(
      "CostData", "Cost function data struct.",
      bp::init<const int, const int>(bp::args("self", "ndx", "nu")))
      .def_readwrite("value", &CostData::value_)
      .add_property(
          "Lx", bp::make_getter(&CostData::Lx_,
                                bp::return_value_policy<bp::return_by_value>()))
      .add_property(
          "Lu", bp::make_getter(&CostData::Lu_,
                                bp::return_value_policy<bp::return_by_value>()))
      .add_property("Lxx", bp::make_getter(
                               &CostData::Lxx_,
                               bp::return_value_policy<bp::return_by_value>()))
      .add_property("Lxu", bp::make_getter(
                               &CostData::Lxu_,
                               bp::return_value_policy<bp::return_by_value>()))
      .add_property("Lux", bp::make_getter(
                               &CostData::Lux_,
                               bp::return_value_policy<bp::return_by_value>()))
      .add_property("Luu", bp::make_getter(
                               &CostData::Luu_,
                               bp::return_value_policy<bp::return_by_value>()))
      .add_property("grad", bp::make_getter(
                                &CostData::grad_,
                                bp::return_value_policy<bp::return_by_value>()))
      .add_property(
          "hess",
          bp::make_getter(&CostData::hess_,
                          bp::return_value_policy<bp::return_by_value>()));

  pinpy::StdVectorPythonVisitor<std::vector<shared_ptr<context::CostBase>>,
                                true>::expose("StdVec_CostAbstract",
                                              "Vector of cost objects.");
  pinpy::StdVectorPythonVisitor<std::vector<shared_ptr<CostData>>,
                                true>::expose("StdVec_CostData",
                                              "Vector of CostData objects.");

  /* Composite costs */

  bp::class_<QuadraticResidualCostTpl<Scalar>, bp::bases<context::CostBase>>(
      "QuadraticResidualCost", "Weighted 2-norm of a given residual function.",
      bp::init<const shared_ptr<StageFunction> &, const context::MatrixXs &>(
          bp::args("self", "function", "weights")))
      .def_readwrite("residual", &QuadraticResidualCostTpl<Scalar>::residual_)
      .def_readwrite("weights", &QuadraticResidualCostTpl<Scalar>::weights_)
      .def(CopyableVisitor<QuadraticResidualCostTpl<Scalar>>())
      .def(CreateDataPythonVisitor<QuadraticResidualCostTpl<Scalar>>());

  {
    using CompositeData = CompositeCostDataTpl<Scalar>;
    bp::class_<CompositeData, bp::bases<CostData>>("CompositeCostData",
                                                   bp::init<int, int>())
        .def_readwrite("residual_data", &CompositeData::residual_data);
  }
}

} // namespace python
} // namespace proxddp
