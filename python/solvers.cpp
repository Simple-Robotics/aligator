#include "proxddp/python/fwd.hpp"

#include "proxddp/core/solver-proxddp.hpp"

namespace proxddp {
namespace python {

void exposeSolvers() {
  using context::Scalar;
  using context::TrajOptProblem;
  using Workspace = WorkspaceTpl<Scalar>;
  using Results = ResultsTpl<Scalar>;

  {
    using QParams = proxddp::internal::q_function_storage<Scalar>;
    using VParams = proxddp::internal::value_storage<Scalar>;
    bp::class_<QParams>("QParams", "Q-function parameters.", bp::no_init)
        .def_readonly("storage", &QParams::storage)
        .add_property(
            "grad_",
            bp::make_getter(&QParams::grad_,
                            bp::return_value_policy<bp::return_by_value>()))
        .add_property(
            "hess_",
            bp::make_getter(&QParams::hess_,
                            bp::return_value_policy<bp::return_by_value>()));
    pinpy::StdVectorPythonVisitor<std::vector<QParams>, true>::expose(
        "StdVec_QParams");

    bp::class_<VParams>("VParams", "Value function parameters.", bp::no_init)
        .def_readonly("storage", &VParams::storage);
    pinpy::StdVectorPythonVisitor<std::vector<VParams>, true>::expose(
        "StdVec_VParams");
  }

  bp::class_<Workspace>(
      "Workspace", "Workspace for ProxDDP.",
      bp::init<const TrajOptProblem &>(bp::args("self", "problem")))
      .def_readonly("value_params", &Workspace::value_params)
      .def_readonly("q_params", &Workspace::q_params)
      .def_readonly("value_params", &Workspace::value_params)
      .def_readonly("kkt_matrix_buffer_", &Workspace::kktMatrixFull_)
      .def_readonly("inner_crit", &Workspace::inner_criterion)
      .def_readonly("prim_infeas_by_stage", &Workspace::primal_infeas_by_stage)
      .def_readonly("dual_infeas_by_stage", &Workspace::dual_infeas_by_stage)
      .def_readonly("inner_criterion_by_stage",
                    &Workspace::inner_criterion_by_stage)
      .def_readonly("problem_data", &Workspace::problem_data)
      .def(bp::self_ns::str(bp::self));

  bp::class_<Results>("Results", "Results struct for proxDDP.",
                      bp::init<const TrajOptProblem &>())
      .def_readonly("gains", &Results::gains_)
      .def_readonly("xs", &Results::xs_)
      .def_readonly("us", &Results::us_)
      .def_readonly("lams", &Results::lams_)
      .def_readonly("co_state", &Results::co_state_)
      .def_readonly("primal_infeas", &Results::primal_infeasibility)
      .def_readonly("dual_infeas", &Results::dual_infeasibility)
      .def_readonly("traj_cost", &Results::traj_cost_, "Trajectory cost.")
      .def_readonly("merit_value", &Results::merit_value_,
                    "Merit function value.")
      .def(bp::self_ns::str(bp::self));

  using SolverType = SolverProxDDP<Scalar>;

  bp::enum_<MultiplierUpdateMode>(
      "MultiplierUpdateMode", "Enum for the kind of multiplier update to use.")
      .value("NEWTON", MultiplierUpdateMode::NEWTON)
      .value("PRIMAL", MultiplierUpdateMode::PRIMAL)
      .value("PRIMAL_DUAL", MultiplierUpdateMode::PRIMAL_DUAL);

  bp::enum_<LinesearchMode>("LinesearchMode", "Linesearch mode.")
      .value("PRIMAL", LinesearchMode::PRIMAL)
      .value("PRIMAL_DUAL", LinesearchMode::PRIMAL_DUAL);

  bp::class_<LinesearchParams<Scalar>>(
      "LinesearchParams", "Parameters for the linesearch.", bp::init<>())
      .def_readwrite("alpha_min", &LinesearchParams<Scalar>::alpha_min)
      .def_readwrite("armijo_c1", &LinesearchParams<Scalar>::armijo_c1)
      .def_readwrite("ls_beta", &LinesearchParams<Scalar>::ls_beta,
                     "Decrease factor; only applies to Armijo linesearch.")
      .def_readwrite("mode", &LinesearchParams<Scalar>::mode,
                     "Linesearch mode.")
      .def_readwrite("strategy", &LinesearchParams<Scalar>::strategy,
                     "Linesearch strategy.");

  bp::class_<SolverType, boost::noncopyable>(
      "ProxDDP",
      "A primal-dual augmented Lagrangian solver, based on DDP to compute "
      "search directions."
      " The solver instance initializes both a Workspace and Results which can "
      "be retrieved"
      " through the `getWorkspace` and `getResults` methods, respectively.",
      bp::init<Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar,
               std::size_t, VerboseLevel>(
          (bp::arg("self"), bp::arg("tol"), bp::arg("mu_init") = 1e-2,
           bp::arg("rho_init") = 0., bp::arg("prim_alpha") = 0.1,
           bp::arg("prim_beta") = 0.9, bp::arg("dual_alpha") = 1.0,
           bp::arg("dual_beta") = 1.0, bp::arg("max_iters") = 1000,
           bp::arg("verbose") = VerboseLevel::QUIET)))
      .def_readonly("mu_init", &SolverType::mu_init,
                    "Initial dual penalty parameter.")
      .def_readonly("rho_init", &SolverType::rho_init,
                    "Initial (primal) proximal parameter.")
      .def_readonly("prim_alpha", &SolverType::prim_alpha,
                    "Primal tolerance log-factor (when steps are accepted).")
      .def_readonly("dual_alpha", &SolverType::dual_alpha,
                    "Dual tolerance log-factor (when steps are accepted).")
      .def_readonly("prim_beta", &SolverType::prim_beta,
                    "Primal tolerance log-factor (when steps are rejected).")
      .def_readonly("dual_beta", &SolverType::dual_beta,
                    "Dual tolerance log-factor (when steps are rejected).")
      .def_readwrite("target_tol", &SolverType::target_tolerance,
                     "Desired tolerance.")
      .def_readwrite("mu_factor", &SolverType::mu_update_factor_)
      .def_readwrite("rho_factor", &SolverType::rho_update_factor_)
      .def_readwrite("multiplier_update_mode", &SolverType::mul_update_mode)
      .def_readwrite("max_iters", &SolverType::MAX_ITERS,
                     "Maximum number of iterations.")
      .def_readwrite("verbose", &SolverType::verbose_,
                     "Verbosity level of the solver.")
      .def("getResults", &SolverType::getResults, bp::args("self"),
           bp::return_internal_reference<>(), "Get the results instance.")
      .def("getWorkspace", &SolverType::getWorkspace, bp::args("self"),
           bp::return_internal_reference<>(), "Get the workspace instance.")
      .def("setup", &SolverType::setup, bp::args("self", "problem"),
           "Allocate workspace and results memory for the problem.")
      .def(
          "run", &SolverType::run,
          bp::args("self", "problem", "xs_init", "us_init"),
          "Run the algorithm. This requires providing initial guesses for both "
          "trajectory and control.");
}

} // namespace python
} // namespace proxddp
