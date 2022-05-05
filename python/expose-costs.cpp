/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/fwd.hpp"
#include "proxddp/core/costs.hpp"


namespace proxddp
{
  namespace python
  {
    namespace internal
    {
      /// @brief Wrapper for the CostDataTpl class and its children.
      template<typename T = CostBaseTpl<context::Scalar>>
      struct PyCostFunction : T, bp::wrapper<T>
      {
        using Scalar = context::Scalar;
        using bp::wrapper<T>::get_override;
        using CostData = CostDataTpl<Scalar>;
        PROXNLP_DYNAMIC_TYPEDEFS(Scalar)

        /// forwarding constructor
        template<typename... Args>
        PyCostFunction(Args&&... args)
          : T(std::forward<Args>(args)...) {}

        virtual void evaluate(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const override
        {
          PROXDDP_PYTHON_OVERRIDE_PURE(void, "evaluate", x, u, data);
        }

        virtual void computeGradients(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const override
        {
          PROXDDP_PYTHON_OVERRIDE_PURE(void, "computeGradients", x, u, data);
        }

        virtual void computeHessians(const ConstVectorRef& x, const ConstVectorRef& u, CostData& data) const override
        {
          PROXDDP_PYTHON_OVERRIDE_PURE(void, "computeHessians", x, u, data);
        }

        virtual shared_ptr<CostData> createData() const override
        {
          PROXDDP_PYTHON_OVERRIDE(shared_ptr<CostData>, T, createData,);
        }

      };
    } // namespace internal
    
    void exposeCosts()
    {
      using context::Scalar;

      bp::register_ptr_to_python<shared_ptr<context::CostBase>>();

      bp::class_<internal::PyCostFunction<>>(
        "CostBase", "Base class for cost functions.",
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

      using CostData = CostDataTpl<Scalar>;
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

