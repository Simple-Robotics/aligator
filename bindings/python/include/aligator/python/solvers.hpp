/// @file
/// @copyright Copyright (C) 2024 LAAS-CNRS, 2024-2025 INRIA
#pragma once

#include <eigenpy/fwd.hpp>
#include <fmt/format.h>

namespace aligator::python {
namespace bp = boost::python;

// fwd-declaration
bp::arg operator""_a(const char *argname, std::size_t);

template <typename SolverType>
struct SolverVisitor : bp::def_visitor<SolverVisitor<SolverType>> {
  using CallbackPtr = typename SolverType::CallbackPtr;
  static auto getCallback(const SolverType &obj, std::string_view name)
      -> CallbackPtr {
    const CallbackPtr &cb = obj.getCallback(name);
    if (!cb) {
      PyErr_SetString(PyExc_KeyError,
                      fmt::format("Key {} not found.", name).c_str());
      bp::throw_error_already_set();
    }
    return cb;
  }

  template <typename... Args> void visit(bp::class_<Args...> &obj) const {
    obj.def_readwrite("verbose", &SolverType::verbose_,
                      "Verbosity level of the solver.")
        .def_readwrite("max_iters", &SolverType::max_iters,
                       "Maximum number of iterations.")
        .def_readwrite("ls_params", &SolverType::ls_params,
                       "Linesearch parameters.")
        .def_readwrite("target_tol", &SolverType::target_tol_,
                       "Target tolerance.")
        .def_readwrite("reg_init", &SolverType::reg_init)
        .def_readwrite("preg", &SolverType::preg_)
        .def_readwrite("force_initial_condition",
                       &SolverType::force_initial_condition_,
                       "Set x0 to be fixed to the initial condition.")
        .add_property("num_threads", &SolverType::getNumThreads)
        .def("setNumThreads", &SolverType::setNumThreads,
             ("self"_a, "num_threads"))
        .def_readonly("results", &SolverType::results_, "Solver results.")
        .def_readonly("workspace", &SolverType::workspace_, "Solver workspace.")
        .def("setup", &SolverType::setup, ("self"_a, "problem"),
             "Allocate solver workspace and results data for the problem.")
        .def("registerCallback", &SolverType::registerCallback,
             ("self"_a, "name", "cb"), "Add a callback to the solver.")
        .def("getCallbackNames", &SolverType::getCallbackNames, "self"_a,
             "Get names of registered callbacks.")
        .def("removeCallback", &SolverType::removeCallback, ("self"_a, "key"),
             "Remove a callback.")
        .def("getCallback", getCallback, ("self"_a, "key"))
        .def("clearCallbacks", &SolverType::clearCallbacks, ("self"_a),
             "Clear callbacks.");
  }
};

} // namespace aligator::python
