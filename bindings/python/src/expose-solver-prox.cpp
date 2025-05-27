/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/python/visitors.hpp"
#include "aligator/python/solvers.hpp"

#include "aligator/solvers/proxddp/solver-proxddp.hpp"

#include <eigenpy/std-unique-ptr.hpp>
#include <eigenpy/variant.hpp>

namespace aligator {
namespace python {

using context::Scalar;
using Linesearch = Linesearch<Scalar>;
using LinesearchOptions = Linesearch::Options;

static void exposeLinesearch() {

  bp::enum_<LSInterpolation>("LSInterpolation",
                             "Linesearch interpolation scheme.")
      .value("BISECTION", LSInterpolation::BISECTION)
      .value("QUADRATIC", LSInterpolation::QUADRATIC)
      .value("CUBIC", LSInterpolation::CUBIC);
  bp::class_<Linesearch>("Linesearch", bp::no_init)
      .def(bp::init<const LinesearchOptions &>(("self"_a, "options")))
      .def_readwrite("options", &Linesearch::options_);
  bp::class_<ArmijoLinesearch<Scalar>, bp::bases<Linesearch>>(
      "ArmijoLinesearch", bp::no_init)
      .def(bp::init<const LinesearchOptions &>(("self"_a, "options")));
  bp::class_<LinesearchOptions>("LinesearchOptions", "Linesearch options.",
                                bp::init<>(("self"_a), "Default constructor."))
      .def_readwrite("armijo_c1", &LinesearchOptions::armijo_c1)
      .def_readwrite("wolfe_c2", &LinesearchOptions::wolfe_c2)
      .def_readwrite(
          "dphi_thresh", &LinesearchOptions::dphi_thresh,
          "Threshold on the derivative at the initial point; the linesearch "
          "will be early-terminated if the derivative is below this threshold.")
      .def_readwrite("alpha_min", &LinesearchOptions::alpha_min,
                     "Minimum step size.")
      .def_readwrite("max_num_steps", &LinesearchOptions::max_num_steps)
      .def_readwrite("interp_type", &LinesearchOptions::interp_type,
                     "Interpolation type: bisection, quadratic or cubic.")
      .def_readwrite("contraction_min", &LinesearchOptions::contraction_min,
                     "Minimum step contraction.")
      .def_readwrite("contraction_max", &LinesearchOptions::contraction_max,
                     "Maximum step contraction.")
      .def(bp::self_ns::str(bp::self));

  bp::class_<NonmonotoneLinesearch<Scalar>, bp::bases<Linesearch>>(
      "NonmonotoneLinesearch", bp::no_init)
      .def(bp::init<LinesearchOptions>(("self"_a, "options")))
      .def_readwrite("avg_eta", &NonmonotoneLinesearch<Scalar>::avg_eta)
      .def_readwrite("beta_dec", &NonmonotoneLinesearch<Scalar>::beta_dec);
}

void exposeProxDDP() {
  using context::ConstVectorRef;
  using context::Results;
  using context::TrajOptProblem;
  using context::VectorRef;
  using context::Workspace;

  exposeLinesearch();

  bp::enum_<LQSolverChoice>("LQSolverChoice")
      .value("LQ_SOLVER_SERIAL", LQSolverChoice::SERIAL)
      .value("LQ_SOLVER_PARALLEL", LQSolverChoice::PARALLEL)
      .value("LQ_SOLVER_STAGEDENSE", LQSolverChoice::STAGEDENSE)
      .export_values();

  bp::class_<Workspace, bp::bases<WorkspaceBaseTpl<Scalar>>,
             boost::noncopyable>(
      "Workspace", "Workspace for ProxDDP.",
      bp::init<const TrajOptProblem &>(("self"_a, "problem")))
      .def_readonly("lqr_problem", &Workspace::lqr_problem,
                    "Buffers for the LQ subproblem.")
      .def_readonly("Lxs", &Workspace::Lxs)
      .def_readonly("Lus", &Workspace::Lus)
      .def_readonly("Lds", &Workspace::Lds)
      .def_readonly("Lvs", &Workspace::Lvs)
      .def_readonly("dxs", &Workspace::dxs)
      .def_readonly("dus", &Workspace::dus)
      .def_readonly("dvs", &Workspace::dvs)
      .def_readonly("dlams", &Workspace::dlams)
      .def_readonly("trial_vs", &Workspace::trial_vs)
      .def_readonly("trial_lams", &Workspace::trial_lams)
      .def_readonly("lams_plus", &Workspace::lams_plus)
      .def_readonly("lams_pdal", &Workspace::lams_pdal)
      .def_readonly("vs_plus", &Workspace::vs_plus)
      .def_readonly("vs_pdal", &Workspace::vs_pdal)
      .def_readonly("shifted_constraints", &Workspace::shifted_constraints)
      .def_readonly("cstr_proj_jacs", &Workspace::cstr_proj_jacs)
      .def_readonly("inner_crit", &Workspace::inner_criterion)
      .def_readonly("active_constraints", &Workspace::active_constraints)
      .def_readonly("prev_xs", &Workspace::prev_xs)
      .def_readonly("prev_us", &Workspace::prev_us)
      .def_readonly("prev_lams", &Workspace::prev_lams)
      .def_readonly("prev_vs", &Workspace::prev_vs)
      .def_readonly("stage_infeasibilities", &Workspace::stage_infeasibilities)
      .def_readonly("state_dual_infeas", &Workspace::state_dual_infeas)
      .def_readonly("control_dual_infeas", &Workspace::control_dual_infeas)
      .def(PrintableVisitor<Workspace>());

  bp::class_<Results, bp::bases<ResultsBaseTpl<Scalar>>>(
      "Results", "Results struct for proxDDP.",
      bp::init<const TrajOptProblem &>(("self"_a, "problem")))
      .def("cycleAppend", &Results::cycleAppend, ("self"_a, "problem", "x0"),
           "Cycle the results.")
      .def_readonly("al_iter", &Results::al_iter)
      .def_readonly("lams", &Results::lams)
      .def_readonly("vs", &Results::vs)
      .def(PrintableVisitor<Results>())
      .def(PrintAddressVisitor<Results>());

  using SolverType = SolverProxDDPTpl<Scalar>;
  using ls_variant_t = SolverType::LinesearchVariant::variant_t;

  auto cls =
      bp::class_<SolverType, boost::noncopyable>(
          "SolverProxDDP",
          "A proximal, augmented Lagrangian solver, using a DDP-type scheme to "
          "compute "
          "search directions and feedforward, feedback gains."
          " The solver instance initializes both a Workspace and a Results "
          "struct.",
          bp::init<const Scalar, const Scalar, std::size_t, VerboseLevel,
                   StepAcceptanceStrategy, HessianApprox>(
              ("self"_a, "tol", "mu_init"_a = 1e-2, "max_iters"_a = 1000,
               "verbose"_a = VerboseLevel::QUIET,
               "sa_strategy"_a = StepAcceptanceStrategy::LINESEARCH_NONMONOTONE,
               "hess_approx"_a = HessianApprox::GAUSS_NEWTON)))
          .def("cycleProblem", &SolverType::cycleProblem,
               ("self"_a, "problem", "data"),
               "Cycle the problem data (for MPC applications).")
          .def_readwrite("bcl_params", &SolverType::bcl_params,
                         "BCL parameters.")
          .def_readwrite("max_refinement_steps",
                         &SolverType::max_refinement_steps_)
          .def_readwrite("refinement_threshold",
                         &SolverType::refinement_threshold_)
          .def_readwrite("linear_solver_choice",
                         &SolverType::linear_solver_choice)
          .def_readwrite("multiplier_update_mode",
                         &SolverType::multiplier_update_mode)
          .def_readwrite("mu_init", &SolverType::mu_init,
                         "Initial AL penalty parameter.")
          .add_property("mu", &SolverType::mu)
          .def_readwrite(
              "rollout_max_iters", &SolverType::rollout_max_iters,
              "Maximum number of iterations when solving the forward dynamics.")
          .def_readwrite("max_al_iters", &SolverType::max_al_iters,
                         "Maximum number of AL iterations.")
          .def_readwrite("ls_mode", &SolverType::ls_mode, "Linesearch mode.")
          .def_readwrite("sa_strategy", &SolverType::sa_strategy_,
                         "StepAcceptance strategy.")
          .def_readwrite("rollout_type", &SolverType::rollout_type_,
                         "Rollout type.")
          .def_readwrite("dual_weight", &SolverType::dual_weight,
                         "Dual penalty weight.")
          .def_readwrite("reg_min", &SolverType::reg_min,
                         "Minimum regularization value.")
          .def_readwrite("reg_max", &SolverType::reg_max,
                         "Maximum regularization value.")
          .def("updateLQSubproblem", &SolverType::updateLQSubproblem, "self"_a)
          .def("computeCriterion", &SolverType::computeCriterion, "self"_a,
               "Compute problem stationarity.")
          .add_property("linearSolver",
                        bp::make_getter(&SolverType::linearSolver_,
                                        eigenpy::ReturnInternalStdUniquePtr{}),
                        "Linear solver for the semismooth Newton method.")
          .def_readwrite("filter", &SolverType::filter_,
                         "Pair filter used to accept a step.")
          .def("computeInfeasibilities", &SolverType::computeInfeasibilities,
               ("self"_a, "problem"), "Compute problem infeasibilities.")
          .add_property("num_threads", &SolverType::getNumThreads)
          .def("setNumThreads", &SolverType::setNumThreads,
               ("self"_a, "num_threads"))
          .add_property("target_dual_tol", &SolverType::getDualTolerance)
          .def("setDualTolerance", &SolverType::setDualTolerance,
               ("self"_a, "tol"),
               "Manually set the solver's dual infeasibility tolerance. Once "
               "this method is called, the dual tolerance and primal tolerance "
               "(target_tol) will not be synced when the latter changes and "
               "`solver.run()` is called.")
          .def(SolverVisitor<SolverType>())
          .add_property("linesearch",
                        bp::make_function(
                            +[](const SolverType &s) -> const ls_variant_t & {
                              return s.linesearch_;
                            },
                            eigenpy::ReturnInternalVariant<ls_variant_t>{}))
          .def("run", &SolverType::run,
               ("self"_a, "problem", "xs_init"_a = bp::list(),
                "us_init"_a = bp::list(), "vs_init"_a = bp::list(),
                "lams_init"_a = bp::list()),
               "Run the algorithm. Can receive initial guess for "
               "multiplier trajectory.");

  {
    using AlmParams = SolverType::AlmParams;
    bp::scope scope{cls};
#define _c(name) def_readwrite(#name, &AlmParams::name)
    bp::class_<AlmParams>("AlmParams", "Parameters for the ALM algorithm",
                          bp::init<>("self"_a))
        ._c(prim_alpha)
        ._c(prim_beta)
        ._c(dual_alpha)
        ._c(dual_beta)
        ._c(mu_update_factor)
        ._c(dyn_al_scale)
        ._c(mu_lower_bound);
#undef _c
  }
}

} // namespace python
} // namespace aligator
