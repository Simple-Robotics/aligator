/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include <eigenpy/fwd.hpp>
#include <fmt/format.h>
#include "proxddp/python/utils/deprecation.hpp"

namespace proxddp {
namespace python {
namespace bp = boost::python;

template <typename T>
struct ClonePythonVisitor : bp::def_visitor<ClonePythonVisitor<T>> {
  template <typename PyT> void visit(PyT &obj) const {
    obj.def("clone", &T::clone, bp::args("self"), "Clone the object.");
  }
};

template <typename T>
struct CreateDataPythonVisitor : bp::def_visitor<CreateDataPythonVisitor<T>> {
  template <typename Pyclass> void visit(Pyclass &obj) const {
    obj.def("createData", &T::createData, bp::args("self"),
            "Create a data object.");
  }
};

/// Visitor for exposing a polymorphic factory for data classes that can be
/// overridden from Python.
/// @sa CreateDataPythonVisitor
/// @sa StageFunctionTpl
/// @tparam T The wrapped class
/// @tparam TWrapper The wrapper class
template <typename T, typename TWrapper>
struct CreateDataPolymorphicPythonVisitor
    : bp::def_visitor<CreateDataPolymorphicPythonVisitor<T, TWrapper>> {
  template <typename PyClass> void visit(PyClass &obj) const {
    obj.def("createData", &T::createData, &TWrapper::default_createData,
            bp::args("self"), "Create a data object.");
  }
};

template <typename T>
struct CopyableVisitor : bp::def_visitor<CopyableVisitor<T>> {
  template <typename PyClass> void visit(PyClass &obj) const {
    obj.def("copy", &copy, bp::arg("self"), "Returns a copy of this.");
  }

private:
  static T copy(const T &self) { return T(self); }
};

template <typename T>
struct PrintableVisitor : bp::def_visitor<PrintableVisitor<T>> {
  template <typename Pyclass> void visit(Pyclass &obj) const {
    obj.def(bp::self_ns::str(bp::self)).def(bp::self_ns::repr(bp::self));
  }
};

template <typename SolverType>
struct SolverVisitor : bp::def_visitor<SolverVisitor<SolverType>> {
  using CallbackPtr = typename SolverType::CallbackPtr;
  static auto getCallback(const SolverType &obj, const std::string &name)
      -> CallbackPtr {
    const auto &cbs = obj.getCallbacks();
    auto cb = cbs.find(name);
    if (cb == cbs.end()) {
      PyErr_SetString(PyExc_KeyError,
                      fmt::format("Key {} not found.", name).c_str());
      bp::throw_error_already_set();
    }
    return cb->second;
  }

  static void registerCallback_old(const SolverType &, const CallbackPtr &) {}

  template <typename PyClass> void visit(PyClass &obj) const {
    obj.def_readwrite("verbose", &SolverType::verbose_,
                      "Verbosity level of the solver.")
        .def_readwrite("max_iters", &SolverType::max_iters,
                       "Maximum number of iterations.")
        .def_readwrite("ls_params", &SolverType::ls_params,
                       "Linesearch parameters.")
        .def_readwrite("target_tol", &SolverType::target_tol_,
                       "Target tolerance.")
        .def_readwrite("xreg", &SolverType::xreg_,
                       "Newton regularization parameter.")
        .def_readwrite("ureg", &SolverType::ureg_,
                       "Newton regularization parameter.")
        .def_readwrite("reg_init", &SolverType::reg_init)
        .def_readwrite("force_initial_condition",
                       &SolverType::force_initial_condition_,
                       "Set x0 to be fixed to the initial condition.")
        .def("getResults", &SolverType::getResults, bp::args("self"),
             deprecated_member<bp::return_internal_reference<>>(
                 "This getter is deprecated. Access the results using "
                 "`solver.results` instead."),
             "Get the results instance.")
        .def("getWorkspace", &SolverType::getWorkspace, bp::args("self"),
             deprecated_member<bp::return_internal_reference<>>(
                 "This getter is deprecated. Access the workspace using "
                 "`solver.workspace` instead."),
             "Get the workspace instance.")
        .def_readonly("results", &SolverType::results_, "Solver results.")
        .def_readonly("workspace", &SolverType::workspace_, "Solver workspace.")
        .def("setup", &SolverType::setup, bp::args("self", "problem"),
             "Allocate solver workspace and results data for the problem.")
        .def("registerCallback", &SolverType::registerCallback,
             bp::args("self", "name", "cb"), "Add a callback to the solver.")
        .def("registerCallback", registerCallback_old,
             deprecated_member<bp::default_call_policies,
                               DeprecationTypes::DEPRECATION>(
                 "This signature for registerCallback() has been deprecated "
                 "and does nothing. It will be removed in the future."),
             bp::args("self", "cb"))
        .def("removeCallback", &SolverType::removeCallback,
             bp::args("self", "key"), "Remove a callback.")
        .def("getCallback", getCallback, bp::args("self", "key"))
        .def("clearCallbacks", &SolverType::clearCallbacks, bp::args("self"),
             "Clear callbacks.");
  }
};

} // namespace python
} // namespace proxddp
