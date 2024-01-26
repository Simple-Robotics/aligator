/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "aligator/python/costs.hpp"
#include "aligator/python/visitors.hpp"

#include "aligator/modelling/costs/quad-costs.hpp"
#include "aligator/modelling/costs/constant-cost.hpp"
#include "aligator/modelling/costs/sum-of-costs.hpp"

namespace aligator {
namespace python {
using context::ConstMatrixRef;
using context::ConstVectorRef;
using context::CostBase;
using context::CostData;
using context::Manifold;
using context::MatrixXs;
using context::Scalar;
using context::VectorXs;
using internal::PyCostFunction;
using QuadraticCost = QuadraticCostTpl<Scalar>;
using CostPtr = shared_ptr<CostBase>;

struct CostDataWrapper : CostData, bp::wrapper<CostData> {
  using CostData::CostData;
};

void exposeQuadCost() {

  bp::class_<ConstantCostTpl<Scalar>, bp::bases<CostBase>>(
      "ConstantCost", "A constant cost term.",
      bp::init<shared_ptr<Manifold>, int, Scalar>(
          bp::args("self", "space", "nu", "value")))
      .def_readwrite("value", &ConstantCostTpl<Scalar>::value_)
      .def(CopyableVisitor<ConstantCostTpl<Scalar>>());

  bp::class_<QuadraticCost, bp::bases<CostBase>>(
      "QuadraticCost",
      "Quadratic cost in both state and control - only for Euclidean spaces.",
      bp::no_init)
      .def(bp::init<ConstMatrixRef, ConstMatrixRef, ConstMatrixRef,
                    ConstVectorRef, ConstVectorRef>(
          bp::args("self", "w_x", "w_u", "w_cross", "interp_x", "interp_u")))
      .def(bp::init<ConstMatrixRef, ConstMatrixRef, ConstVectorRef,
                    ConstVectorRef>(
          bp::args("self", "w_x", "w_u", "interp_x", "interp_u")))
      .def(bp::init<ConstMatrixRef, ConstMatrixRef>(
          "Constructor with just weights (no cross-term).",
          bp::args("self", "w_x", "w_u")))
      .def(bp::init<ConstMatrixRef, ConstMatrixRef, ConstMatrixRef>(
          "Constructor with just weights (with cross-term).",
          bp::args("self", "w_x", "w_u", "w_cross")))
      .def_readwrite("w_x", &QuadraticCost::Wxx_, "Weights on the state.")
      .def_readwrite("w_u", &QuadraticCost::Wuu_, "Weights on the control.")
      .def_readwrite("interp_x", &QuadraticCost::interp_x)
      .def_readwrite("interp_u", &QuadraticCost::interp_u)
      .add_property("has_cross_term", &QuadraticCost::hasCrossTerm,
                    "Whether there is a cross term.")
      .add_property("weights_cross", &QuadraticCost::getCrossWeights,
                    &QuadraticCost::setCrossWeight, "Cross term weight.")
      .def(CopyableVisitor<QuadraticCostTpl<Scalar>>());

  bp::class_<QuadraticCost::Data, bp::bases<CostData>>(
      "QuadraticCostData", "Quadratic cost data.", bp::no_init);
}

/// Composite cost functions.
void exposeComposites();

void exposeCostStack() {
  using CostStack = CostStackTpl<Scalar>;
  using CostStackData = CostStackDataTpl<Scalar>;

  bp::class_<CostStack, bp::bases<CostBase>>(
      "CostStack", "A weighted sum of other cost functions.",
      bp::init<shared_ptr<Manifold>, int, const std::vector<CostPtr> &,
               const std::vector<Scalar> &>(("self"_a, "space", "nu",
                                             "components"_a = bp::list(),
                                             "weights"_a = bp::list())))
      .def_readwrite("components", &CostStack::components_,
                     "Components of this cost stack.")
      .def_readonly("weights", &CostStack::weights_,
                    "Weights of this cost stack.")
      .def("addCost", &CostStack::addCost, ("self"_a, "cost", "weight"_a = 1.),
           "Add a cost to the stack of costs.")
      .def("size", &CostStack::size, "Get the number of cost components.")
      .def(CopyableVisitor<CostStack>());

  bp::register_ptr_to_python<shared_ptr<CostStackData>>();
  bp::class_<CostStackData, bp::bases<CostData>>(
      "CostStackData", "Data struct for CostStack.", bp::no_init)
      .def_readonly("sub_cost_data", &CostStackData::sub_cost_data);
}

void exposeCostBase() {
  bp::register_ptr_to_python<CostPtr>();

  bp::class_<PyCostFunction<>, boost::noncopyable>(
      "CostAbstract", "Base class for cost functions.", bp::no_init)
      .def(bp::init<shared_ptr<Manifold>, const int>(
          bp::args("self", "space", "nu")))
      .def("evaluate", bp::pure_virtual(&CostBase::evaluate),
           bp::args("self", "x", "u", "data"), "Evaluate the cost function.")
      .def("computeGradients", bp::pure_virtual(&CostBase::computeGradients),
           bp::args("self", "x", "u", "data"),
           "Compute the cost function gradients.")
      .def("computeHessians", bp::pure_virtual(&CostBase::computeHessians),
           bp::args("self", "x", "u", "data"),
           "Compute the cost function hessians.")
      .def_readonly("space", &CostBase::space)
      .add_property("nx", &CostBase::nx)
      .add_property("ndx", &CostBase::ndx)
      .add_property("nu", &CostBase::nu)
      .def(CreateDataPythonVisitor<CostBase>());

  bp::register_ptr_to_python<shared_ptr<CostData>>();
  bp::class_<CostDataWrapper, boost::noncopyable>(
      "CostData", "Cost function data struct.", bp::no_init)
      .def(bp::init<const int, const int>(bp::args("self", "ndx", "nu")))
      .def(bp::init<const CostBase &>(bp::args("self", "cost")))
      .def_readwrite("value", &CostData::value_)
      .def_readwrite("grad", &CostData::grad_)
      .def_readwrite("hess", &CostData::hess_)
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
                               bp::return_value_policy<bp::return_by_value>()));

  StdVectorPythonVisitor<std::vector<CostPtr>, true>::expose(
      "StdVec_CostAbstract");
  StdVectorPythonVisitor<std::vector<shared_ptr<CostData>>, true>::expose(
      "StdVec_CostData");
}

void exposeCostOps();

void exposeCosts() {
  exposeCostBase();
  exposeCostStack();
  exposeQuadCost();
  exposeComposites();
  exposeCostOps();
}

} // namespace python
} // namespace aligator
