#pragma once

namespace proxddp
{
  /// @brief  The Python bindings.
  namespace python
  {
  }
} // namespace proxddp

#include "proxddp/python/context.hpp"
#include "proxddp/python/macros.hpp"

#include <pinocchio/fwd.hpp>
#include <pinocchio/bindings/python/utils/std-vector.hpp>

#include <eigenpy/eigenpy.hpp>

namespace proxddp
{
  namespace python
  {
    namespace pinpy = pinocchio::python;
    namespace bp = boost::python;

    /// Expose ternary functions
    void exposeFunctions();
    void exposeCosts();
    void exposeStage();
    void exposeProblem();

    void exposeDynamics();
    void exposeIntegrators();
    void exposeSolvers();

    template <typename T>
    struct ClonePythonVisitor : bp::def_visitor<ClonePythonVisitor<T>>
    {
      template <typename PyT>
      void visit(PyT &obj) const
      {
        obj.def("clone", &T::clone, "Clone the object.");
      }
    };

    template <typename T>
    struct CreateDataPythonVisitor : bp::def_visitor<CreateDataPythonVisitor<T>>
    {
      template <typename Pyclass>
      void visit(Pyclass &obj) const
      {
        obj.def("createData", &T::createData, "Create a data object.");
      }
    };

  } // namespace python
} // namespace proxddp
