/// @file
/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#pragma once

#include <eigenpy/fwd.hpp>
#include <proxsuite-nlp/python/deprecation-policy.hpp>
#include <fmt/format.h>

namespace aligator::python {
namespace bp = boost::python;

// fwd-declaration
bp::arg operator""_a(const char *argname, std::size_t);

template <typename SolverType>
struct SolverVisitor : bp::def_visitor<SolverVisitor<SolverType>> {
  using CallbackPtr = typename SolverType::CallbackPtr;
  static auto getCallback(const SolverType &obj,
                          const std::string &name) -> CallbackPtr {
    const auto &cbs = obj.getCallbacks();
    auto cb = cbs.find(name);
    if (cb == cbs.end()) {
      PyErr_SetString(PyExc_KeyError,
                      fmt::format("Key {} not found.", name).c_str());
      bp::throw_error_already_set();
    }
    return cb->second;
  }

  template <typename PyClass> void visit(PyClass &obj) const {
    using proxsuite::nlp::deprecation_warning_policy;
    using proxsuite::nlp::DeprecationType;
    obj.def_readwrite("verbose", &SolverType::verbose_,
                      "Verbosity level of the solver.")
        .def_readwrite("max_iters", &SolverType::max_iters,
                       "Maximum number of iterations.")
        .def_readwrite("ls_params", &SolverType::ls_params,
                       "Linesearch parameters.")
        .def_readwrite("target_tol", &SolverType::target_tol_,
                       "Target tolerance.")
        .def_readwrite("reg_init", &SolverType::reg_init)
        .def_readwrite("force_initial_condition",
                       &SolverType::force_initial_condition_,
                       "Set x0 to be fixed to the initial condition.")
        .add_property("num_threads", &SolverType::getNumThreads)
        .def("setNumThreads", &SolverType::setNumThreads,
             ("self"_a, "num_threads"))
        .def("getResults", &SolverType::getResults, ("self"_a),
             deprecation_warning_policy<DeprecationType::DEPRECATION,
                                        bp::return_internal_reference<>>(
                 "This getter is deprecated. Access the results using "
                 "`solver.results` instead."),
             "Get the results instance.")
        .def("getWorkspace", &SolverType::getWorkspace, ("self"_a),
             deprecation_warning_policy<DeprecationType::DEPRECATION,
                                        bp::return_internal_reference<>>(
                 "This getter is deprecated. Access the workspace using "
                 "`solver.workspace` instead."),
             "Get the workspace instance.")
        .def_readonly("results", &SolverType::results_, "Solver results.")
        .def_readonly("workspace", &SolverType::workspace_, "Solver workspace.")
        .def("setup", &SolverType::setup, ("self"_a, "problem"),
             "Allocate solver workspace and results data for the problem.")
        .def("registerCallback", &SolverType::registerCallback,
             ("self"_a, "name", "cb"), "Add a callback to the solver.")
        .def("removeCallback", &SolverType::removeCallback, ("self"_a, "key"),
             "Remove a callback.")
        .def("getCallback", getCallback, ("self"_a, "key"))
        .def("clearCallbacks", &SolverType::clearCallbacks, ("self"_a),
             "Clear callbacks.");
  }
};

} // namespace aligator::python
