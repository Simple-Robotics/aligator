#include "proxddp/python/fwd.hpp"

#include "proxddp/core/solver-proxddp.hpp"
#include "proxddp/fddp/solver-fddp.hpp"

namespace proxddp {
namespace python {

template <typename SolverType>
struct SolverVisitor : bp::def_visitor<SolverVisitor<SolverType>> {
  template <typename PyClass> void visit(PyClass &obj) const {
    obj.def_readwrite("verbose", &SolverType::verbose_,
                      "Verbosity level of the solver.")
        .def_readwrite("max_iters", &SolverType::MAX_ITERS,
                       "Maximum number of iterations.")
        .def_readwrite("ls_params", &SolverType::ls_params,
                       "Linesearch parameters.")
        .def_readwrite("target_tol", &SolverType::target_tol_,
                       "Target tolerance.")
        .def_readwrite("xreg", &SolverType::xreg_,
                       "Newton regularization parameter.")
        .def_readwrite("ureg", &SolverType::ureg_,
                       "Newton regularization parameter.")
        .def("getResults", &SolverType::getResults, bp::args("self"),
             bp::return_internal_reference<>(), "Get the results instance.")
        .def("getWorkspace", &SolverType::getWorkspace, bp::args("self"),
             bp::return_internal_reference<>(), "Get the workspace instance.")
        .def("setup", &SolverType::setup, bp::args("self", "problem"),
             "Allocate solver workspace and results data for the problem.")
        .def("registerCallback", &SolverType::registerCallback,
             bp::args("self", "cb"), "Add a callback to the solver.")
        .def("clearCallbacks", &SolverType::clearCallbacks, bp::args("self"),
             "Clear callbacks.");
  }
};

void exposeBase() {
  using context::Scalar;

  using QParams = proxddp::internal::q_storage<Scalar>;
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

  bp::class_<VParams>("VParams", "Value function parameters.", bp::no_init)
      .def_readonly("storage", &VParams::storage);

  pinpy::StdVectorPythonVisitor<std::vector<QParams>, true>::expose(
      "StdVec_QParams");
  pinpy::StdVectorPythonVisitor<std::vector<VParams>, true>::expose(
      "StdVec_VParams");

  using WorkspaceBase = WorkspaceBaseTpl<Scalar>;
  bp::class_<WorkspaceBase>("WorkspaceBase", bp::no_init)
      .def_readonly("nsteps", &WorkspaceBase::nsteps)
      .def_readonly("problem_data", &WorkspaceBase::problem_data)
      .def_readonly("trial_prob_data", &WorkspaceBase::trial_prob_data)
      .def_readonly("trial_xs", &WorkspaceBase::trial_xs_)
      .def_readonly("trial_us", &WorkspaceBase::trial_us_)
      .def_readonly("value_params", &WorkspaceBase::value_params)
      .def_readonly("q_params", &WorkspaceBase::q_params);

  using ResultsBase = ResultsBaseTpl<Scalar>;
  bp::class_<ResultsBase>("ResultsBase", "Base results struct.", bp::no_init)
      .def_readonly("num_iters", &ResultsBase::num_iters,
                    "Number of solver iterations.")
      .def_readonly("conv", &ResultsBase::conv)
      .def_readonly("gains", &ResultsBase::gains_)
      .def_readonly("xs", &ResultsBase::xs_)
      .def_readonly("us", &ResultsBase::us_)
      .def_readonly("lams", &ResultsBase::lams_)
      .def_readonly("co_state", &ResultsBase::co_state_)
      .def_readonly("primal_infeas", &ResultsBase::primal_infeasibility)
      .def_readonly("dual_infeas", &ResultsBase::dual_infeasibility)
      .def_readonly("traj_cost", &ResultsBase::traj_cost_, "Trajectory cost.")
      .def_readonly("merit_value", &ResultsBase::merit_value_,
                    "Merit function value.")
      .def(PrintableVisitor<ResultsBase>());
}

void exposeFDDP() {
  using context::Manifold;
  using context::Scalar;
  using SolverType = SolverFDDP<Scalar>;
  using Workspace = WorkspaceFDDPTpl<Scalar>;
  using Results = ResultsFDDPTpl<Scalar>;

  bp::class_<Workspace, bp::bases<WorkspaceBaseTpl<Scalar>>>("WorkspaceFDDP",
                                                             bp::no_init)
      .def_readonly("dxs", &Workspace::dxs_)
      .def_readonly("dus", &Workspace::dus_);

  bp::class_<Results, bp::bases<ResultsBaseTpl<Scalar>>>(
      "ResultsFDDP",
      bp::init<const context::TrajOptProblem &>(bp::args("self", "problem")))
      .def("getFeedforward",
           bp::make_function(
               +[](const Results &m, std::size_t i) {
                 return m.getFeedforward(i);
               },
               bp::return_value_policy<bp::return_by_value>(),
               bp::args("self", "i")),
           "Get the feedforward gain at time index :math:`i`.")
      .def(
          "getFeedback",
          bp::make_function(
              +[](const Results &m, std::size_t i) { return m.getFeedback(i); },
              bp::return_value_policy<bp::return_by_value>(),
              bp::args("self", "i")),
          "Get the feedback gain at time index :math:`i`.");

  bp::class_<SolverType, boost::noncopyable>(
      "SolverFDDP", "An implementation of the FDDP solver from Crocoddyl.",
      bp::init<Scalar, bp::optional<VerboseLevel, Scalar>>(
          bp::args("self", "tol", "verbose", "reg_init")))
      .def_readwrite("reg_min", &SolverType::reg_min_)
      .def_readwrite("reg_max", &SolverType::reg_max_)
      .def(SolverVisitor<SolverType>())
      .def("run", &SolverType::run,
           (bp::arg("self"), bp::arg("problem"), bp::arg("xs_init"),
            bp::arg("us_init")));
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(run_overloads, run, 1, 4)

void exposeProxDDP() {
  using context::Scalar;
  using context::TrajOptProblem;
  using Workspace = WorkspaceTpl<Scalar>;
  using Results = ResultsTpl<Scalar>;

  bp::class_<Workspace, bp::bases<WorkspaceBaseTpl<Scalar>>>(
      "Workspace", "Workspace for ProxDDP.",
      bp::init<const TrajOptProblem &>(bp::args("self", "problem")))
      .def_readonly("kkt_matrix_", &Workspace::kkt_matrix_buf_)
      .def_readonly("kkt_rhs_", &Workspace::kkt_rhs_buf_)
      .def_readonly("inner_crit", &Workspace::inner_criterion)
      .def_readonly("prim_infeas_by_stage", &Workspace::primal_infeas_by_stage)
      .def_readonly("dual_infeas_by_stage", &Workspace::dual_infeas_by_stage)
      .def_readonly("inner_criterion_by_stage",
                    &Workspace::inner_criterion_by_stage)
      .def_readonly("prox_datas", &Workspace::prox_datas)
      .def(PrintableVisitor<Workspace>());

  bp::class_<Results, bp::bases<ResultsBaseTpl<Scalar>>>(
      "Results", "Results struct for proxDDP.",
      bp::init<const TrajOptProblem &>())
      .def_readonly("al_iter", &Results::al_iter);

  using SolverType = SolverProxDDP<Scalar>;

  bp::enum_<MultiplierUpdateMode>(
      "MultiplierUpdateMode", "Enum for the kind of multiplier update to use.")
      .value("NEWTON", MultiplierUpdateMode::NEWTON)
      .value("PRIMAL", MultiplierUpdateMode::PRIMAL)
      .value("PRIMAL_DUAL", MultiplierUpdateMode::PRIMAL_DUAL);

  bp::enum_<LinesearchMode>("LinesearchMode", "Linesearch mode.")
      .value("PRIMAL", LinesearchMode::PRIMAL)
      .value("PRIMAL_DUAL", LinesearchMode::PRIMAL_DUAL);

  {
    using BCLType = BCLParams<Scalar>;
    bp::class_<BCLType>("BCLParams",
                        "Parameters for the bound-constrained Lagrangian (BCL) "
                        "penalty update strategy.",
                        bp::init<>(bp::args("self")))
        .def_readwrite("prim_alpha", &BCLType::prim_alpha)
        .def_readwrite("prim_beta", &BCLType::prim_beta)
        .def_readwrite("dual_alpha", &BCLType::dual_alpha)
        .def_readwrite("dual_beta", &BCLType::dual_beta)
        .def_readwrite("mu_factor", &BCLType::mu_update_factor)
        .def_readwrite("rho_factor", &BCLType::rho_update_factor);
  }

  auto cl =
      bp::class_<SolverType, boost::noncopyable>(
          "SolverProxDDP",
          "A primal-dual augmented Lagrangian solver, based on DDP to compute "
          "search directions."
          " The solver instance initializes both a Workspace and Results which "
          "can "
          "be retrieved"
          " through the `getWorkspace` and `getResults` methods, respectively.",
          bp::init<Scalar, Scalar, Scalar, std::size_t, VerboseLevel>(
              (bp::arg("self"), bp::arg("tol"), bp::arg("mu_init") = 1e-2,
               bp::arg("rho_init") = 0., bp::arg("max_iters") = 1000,
               bp::arg("verbose") = VerboseLevel::QUIET)))
          .def_readwrite("bcl_params", &SolverType::bcl_params,
                         "BCL parameters.")
          .def_readwrite("multiplier_update_mode",
                         &SolverType::multiplier_update_mode)
          .def_readwrite("mu_init", &SolverType::mu_init,
                         "Initial dual regularization/ALM parameter.")
          .def_readwrite("rho_init", &SolverType::rho_init,
                         "Initial proximal regularization.")
          .def(SolverVisitor<SolverType>())
          .def("run", &SolverType::run,
               run_overloads((bp::arg("self"), bp::arg("problem"),
                              bp::arg("xs_init"), bp::arg("us_init"),
                              bp::arg("lams_init")),
                             "Run the algorithm. Can receive initial guess for "
                             "multiplier trajectory."));
  bp::scope().attr("ProxDDP") = cl;
}

void exposeSolvers() {
  exposeBase();
  exposeProxDDP();
  exposeFDDP();
}

} // namespace python
} // namespace proxddp
