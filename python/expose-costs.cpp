/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/fwd.hpp"
#include "proxddp/python/costs.hpp"

#include "proxddp/modelling/quad-costs.hpp"
#include "proxddp/modelling/composite-costs.hpp"
#include "proxddp/modelling/sum-of-costs.hpp"


namespace proxddp
{
  namespace python
  {
    
    void exposeCosts()
    {
      using context::Scalar;
      using context::StageFunction;
      using CostData = CostDataAbstract<Scalar>;

      bp::register_ptr_to_python<shared_ptr<context::CostBase>>();

      bp::class_<internal::PyCostFunction<>>(
        "CostAbstract", "Base class for cost functions.",
        bp::init<const int, const int>(
          bp::args("self", "ndx", "nu")
        )
      )
        .def("evaluate", bp::pure_virtual(&context::CostBase::evaluate),
             bp::args("self", "x", "u", "data"),
             "Evaluate the cost function.")
        .def("computeGradients", bp::pure_virtual(&context::CostBase::evaluate),
             bp::args("self", "x", "u", "data"),
             "Compute the cost function gradients.")
        .def("computeHessians", bp::pure_virtual(&context::CostBase::computeHessians),
             bp::args("self", "x", "u", "data"),
             "Compute the cost function hessians.")
        .def(CreateDataPythonVisitor<context::CostBase>());

      // bp::class_<SumOfCosts<Scalar>, bp::bases<context::CostBase>>(
      //   "SumOfCosts",
      //   bp::init<const std::vector<shared_ptr<context::CostBase>>&,
      //            const std::vector<Scalar>&>(
      //              bp::args("self", "components", "weights")
      //            )
      // )
      //   .def("addCost", &SumOfCosts<Scalar>::addCost, "Add a cost to the stack of costs.")
      //   .def("size", &SumOfCosts<Scalar>::size, "Get the number of cost components.");

      bp::class_<QuadResidualCost<Scalar>, bp::bases<context::CostBase>>(
        "QuadResidualCost",
        "Weighted 2-norm of a given residual function.",
        bp::init<const shared_ptr<StageFunction>&,
                 const context::MatrixXs&>(
                   bp::args("self", "function", "weights")
                 )
      )
        .def_readwrite("residual", &QuadResidualCost<Scalar>::residual_)
        .def_readwrite("weights", &QuadResidualCost<Scalar>::weights_)
        .def(CreateDataPythonVisitor<QuadResidualCost<Scalar>>());

      bp::class_<CostData, shared_ptr<CostData>>(
        "CostData", "Cost function data struct.",
        bp::init<const int, const int>(
          bp::args("self", "ndx", "nu")
        )
      )
        .def_readwrite("value", &CostData::value_)
        .add_property("Lx",  bp::make_getter(&CostData::Lx_, bp::return_value_policy<bp::return_by_value>()) )
        .add_property("Lu",  bp::make_getter(&CostData::Lu_, bp::return_value_policy<bp::return_by_value>()) )
        .add_property("Lxx", bp::make_getter(&CostData::Lxx_,bp::return_value_policy<bp::return_by_value>()) )
        .add_property("Lxu", bp::make_getter(&CostData::Lxu_,bp::return_value_policy<bp::return_by_value>()) )
        .add_property("Lux", bp::make_getter(&CostData::Lux_,bp::return_value_policy<bp::return_by_value>()) )
        .add_property("Luu", bp::make_getter(&CostData::Luu_,bp::return_value_policy<bp::return_by_value>()) )
        .def_readwrite("_grad", &CostData::grad_)
        .def_readwrite("_hessian", &CostData::hess_)
        ;

      pinpy::StdVectorPythonVisitor<std::vector<shared_ptr<CostData>>, true>::expose("StdVec_CostData", "Vector of CostData.");
    }
    
  } // namespace python
} // namespace proxddp

