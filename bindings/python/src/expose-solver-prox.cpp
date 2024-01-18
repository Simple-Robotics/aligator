#include "aligator/python/fwd.hpp"
#include "aligator/python/utils.hpp"

#include "aligator/solvers/proxddp/solver-proxddp.hpp"

namespace aligator {
namespace python {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(prox_run_overloads, run, 1, 4)

void exposeProxDDP() {
  using context::ConstVectorRef;
  using context::Results;
  using context::Scalar;
  using context::TrajOptProblem;
  using context::VectorRef;
  using context::Workspace;

  register_enum_symlink<proxsuite::nlp::LDLTChoice>(true);
  eigenpy::register_symbolic_link_to_registered_type<
      Linesearch<Scalar>::Options>();
  eigenpy::register_symbolic_link_to_registered_type<LinesearchStrategy>();
  eigenpy::register_symbolic_link_to_registered_type<
      proxsuite::nlp::LSInterpolation>();
  eigenpy::register_symbolic_link_to_registered_type<context::BCLParams>();

  using ProxScaler = ConstraintProximalScalerTpl<Scalar>;
  bp::class_<ProxScaler, boost::noncopyable>("ProxScaler", bp::no_init)
      .def(
          "set_weight",
          +[](ProxScaler &s, Scalar v, std::size_t j) {
            if (j >= s.size()) {
              PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
              bp::throw_error_already_set();
            }
            s.setWeight(v, j);
          },
          bp::args("self", "value", "j"))
      .add_property("size", &ProxScaler::size,
                    "Get the number of constraint blocks.")
      .add_property(
          "weights", +[](ProxScaler &s) -> VectorRef { return s.getWeights(); },
          +[](ProxScaler &s, const ConstVectorRef &w) {
            if (s.getWeights().size() != w.size()) {
              PyErr_SetString(PyExc_ValueError, "Input has wrong dimension.");
            }
            s.setWeights(w);
          },
          "Vector of weights for each constraint in the stack.")
      .add_property(
          "matrix",
          +[](ProxScaler &sc) -> ConstVectorRef { return sc.diagMatrix(); });

  bp::def("applyDefaultScalingStrategy", applyDefaultScalingStrategy<Scalar>,
          "scaler"_a, "Apply the default strategy for scaling constraints.");

  bp::class_<Workspace, bp::bases<WorkspaceBaseTpl<Scalar>>,
             boost::noncopyable>(
      "Workspace", "Workspace for ProxDDP.",
      bp::init<const TrajOptProblem &, bp::optional<LDLTChoice>>(
          bp::args("self", "problem", "ldlt_choice")))
      .def(
          "getConstraintScaler",
          +[](const Workspace &ws, std::size_t j) -> const ProxScaler & {
            if (j >= ws.cstr_scalers.size()) {
              PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
              bp::throw_error_already_set();
            }
            return ws.cstr_scalers[j];
          },
          bp::args("self", "j"), bp::return_internal_reference<>(),
          "Scalers of the constraints in the proximal algorithm.")
      .def_readwrite("lqrData", &Workspace::lqrData,
                     "Data buffer for the underlying LQ problem.")
      .def_readonly("kkt_mat", &Workspace::kkt_mats_)
      .def_readonly("kkt_rhs", &Workspace::kkt_rhs_)
      .def_readonly("kkt_residuals", &Workspace::kkt_resdls_,
                    "KKT system residuals.")
      .def_readonly("Lxs", &Workspace::Lxs_)
      .def_readonly("Lus", &Workspace::Lus_)
      .def_readonly("Lds", &Workspace::Lds_)
      .def_readonly("dxs", &Workspace::dxs)
      .def_readonly("dus", &Workspace::dus)
      .def_readonly("dlams", &Workspace::dlams)
      .def_readonly("trial_lams", &Workspace::trial_lams)
      .def_readonly("lams_plus", &Workspace::lams_plus)
      .def_readonly("lams_pdal", &Workspace::lams_pdal)
      .def_readonly("shifted_constraints", &Workspace::shifted_constraints)
      .def_readonly("proj_jacobians", &Workspace::proj_jacobians)
      .def_readonly("inner_crit", &Workspace::inner_criterion)
      .def_readonly("active_constraints", &Workspace::active_constraints)
      // .def(
      //     "get_ldlt",
      //     +[](const Workspace &ws,
      //         std::size_t i) -> proxsuite::nlp::linalg::ldlt_base<Scalar>
      //         const & {
      //       if (i >= ws.ldlts_.size()) {
      //         PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
      //         bp::throw_error_already_set();
      //       }
      //       return ws.ldlts_[i];
      //     },
      //     bp::return_internal_reference<>(), bp::args("self", "i"),
      //     "Get the LDLT algorithm for the i-th linear problem.")
      .def_readonly("prev_xs", &Workspace::prev_xs)
      .def_readonly("prev_us", &Workspace::prev_us)
      .def_readonly("prev_lams", &Workspace::prev_lams)
      .def_readonly("stage_prim_infeas", &Workspace::stage_prim_infeas)
      .def_readonly("stage_dual_infeas", &Workspace::stage_dual_infeas)
      .def_readonly("stage_inner_crits", &Workspace::stage_inner_crits)
      .def(PrintableVisitor<Workspace>());

  bp::class_<Results, bp::bases<ResultsBaseTpl<Scalar>>>(
      "Results", "Results struct for proxDDP.",
      bp::init<const TrajOptProblem &>())
      .def_readonly("al_iter", &Results::al_iter)
      .def(PrintableVisitor<Results>());

  using SolverType = SolverProxDDP<Scalar>;

  bp::class_<SolverType, boost::noncopyable>(
      "SolverProxDDP",
      "A proximal, augmented Lagrangian solver, using a DDP-type scheme to "
      "compute "
      "search directions and feedforward, feedback gains."
      " The solver instance initializes both a Workspace and a Results struct.",
      bp::init<Scalar, Scalar, Scalar, std::size_t, VerboseLevel,
               HessianApprox>(("self"_a, "tol", "mu_init"_a = 1e-2,
                               "rho_init"_a = 0., "max_iters"_a = 1000,
                               "verbose"_a = VerboseLevel::QUIET,
                               "hess_approx"_a = HessianApprox::GAUSS_NEWTON)))
      .def_readwrite("bcl_params", &SolverType::bcl_params, "BCL parameters.")
      .def_readwrite("max_refinement_steps", &SolverType::max_refinement_steps_)
      .def_readwrite("refinement_threshold", &SolverType::refinement_threshold_)
      .def_readwrite("ldlt_algo_choice", &SolverType::ldlt_algo_choice_,
                     "Choice of LDLT algorithm.")
      .def_readwrite("multiplier_update_mode",
                     &SolverType::multiplier_update_mode)
      .def_readwrite("mu_init", &SolverType::mu_init,
                     "Initial AL penalty parameter.")
      .def_readwrite("rho_init", &SolverType::rho_init,
                     "Initial proximal regularization.")
      .def_readwrite("mu_min", &SolverType::MU_MIN,
                     "Lower bound on the AL penalty parameter.")
      .def_readwrite(
          "rollout_max_iters", &SolverType::rollout_max_iters,
          "Maximum number of iterations when solving the forward dynamics.")
      .def_readwrite("max_al_iters", &SolverType::max_al_iters,
                     "Maximum number of AL iterations.")
      .def_readwrite("ls_mode", &SolverType::ls_mode, "Linesearch mode.")
      .def_readwrite("rollout_type", &SolverType::rollout_type_,
                     "Rollout type.")
      .def_readwrite("dual_weight", &SolverType::dual_weight,
                     "Dual penalty weight.")
      .def_readwrite("reg_min", &SolverType::reg_min,
                     "Minimum regularization value.")
      .def_readwrite("reg_max", &SolverType::reg_max,
                     "Maximum regularization value.")
      .def("updateLqrSubproblem", &SolverType::updateLqrSubproblem,
           bp::args("self"))
      .def("getLinesearchMu", &SolverType::getLinesearchMu, bp::args("self"))
      .def("computeCriterion", &SolverType::computeCriterion,
           bp::args("self", "problem"), "Compute problem stationarity.")
      .def("computeInfeasibilities", &SolverType::computeInfeasibilities,
           bp::args("self", "problem"), "Compute problem infeasibilities.")
      .def(SolverVisitor<SolverType>())
      .def("run", &SolverType::run,
           prox_run_overloads(
               (bp::arg("self"), bp::arg("problem"), bp::arg("xs_init"),
                bp::arg("us_init"), bp::arg("lams_init")),
               "Run the algorithm. Can receive initial guess for "
               "multiplier trajectory."));
}

} // namespace python
} // namespace aligator
