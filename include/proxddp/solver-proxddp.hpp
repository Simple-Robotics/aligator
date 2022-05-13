/// @file solver-proxddp.hpp
/// @brief  Definitions for the proximal trajectory optimization algorithm.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/solver-workspace.hpp"
#include "proxddp/core/solver-results.hpp"

#include <fmt/color.h>
#include <fmt/ostream.h>

#include <Eigen/Cholesky>

namespace proxddp
{

  /// @brief Solver.
  template<typename _Scalar>
  struct SolverTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

    using Workspace = WorkspaceTpl<Scalar>;
    using value_store_t = internal::value_storage<Scalar>;
    using q_store_t = internal::q_function_storage<Scalar>;

    const Scalar mu_init = 0.01;
    const Scalar rho_init = 0.;

    /// Dual proximal/constraint penalty parameter \f$\mu\f$
    Scalar mu_ = mu_init;
    /// Primal proximal parameter \f$\rho > 0\f$
    Scalar rho_ = rho_init;

    /// Solver tolerance \f$\epsilon > 0\f$.
    const Scalar tol_ = 1e-6;
    
    /// @brief Perform the Riccati forward pass.
    ///
    /// @warning This function assumes \f$\delta x_0\f$ has already been computed!
    /// @returns This computes the primal-dual step \f$(\delta \bfx,\delta \bfu,\delta\bmlam)\f$
    void forwardPass(const ShootingProblemTpl<Scalar>& problem,
                     Workspace& workspace) const;

    /// @brief    Try a step of size \f$\alpha\f$.
    /// @returns  A primal-dual trial point
    ///           \f$(\bfx \oplus\alpha\delta\bfx, \bfu+\alpha\delta\bfu, \bmlam+\alpha\delta\bmlam)\f$
    void tryStep(const ShootingProblemTpl<Scalar>& problem,
                 Workspace& workspace,
                 ResultsTpl<Scalar>& results,
                 const Scalar alpha) const;

    /// Compute the active sets at each node and multiplier estimates, and projector Jacobian matrices.
    void computeActiveSetsAndMultipliers(
      const ShootingProblemTpl<Scalar>& problem,
      Workspace& workspace,
      ResultsTpl<Scalar>& results) const;

    /// @brief    Perform the Riccati backward pass.
    ///
    /// @warning  Compute the derivatives first!
    void backwardPass(
      const ShootingProblemTpl<Scalar>& problem,
      Workspace& workspace,
      ResultsTpl<Scalar>& results) const;

    /** @brief    Perform the inner loop of the algorithm (augmented Lagrangian minimization).
     */
    void solverInnerLoop(
      const ShootingProblemTpl<Scalar>& problem,
      Workspace& workspace,
      ResultsTpl<Scalar>& results)
    {

      results.xs_ = workspace.prev_xs_;
      results.us_ = workspace.prev_us_;
      results.lams_ = workspace.prev_lams_;

      std::size_t k = 0;
      while (k < MAX_ITERS)
      {

        backwardPass(problem, workspace, results);

        // check stopping criterion

        forwardPass(problem, workspace);

        performLinesearch(problem, workspace, results);

        k++;
      }


    }

    /// @brief Run the linesearch procedure.
    /// @returns    The optimal step size \f$\alpha^*\f$.
    Scalar performLinesearch(const ShootingProblemTpl<Scalar>& problem,
                             Workspace& workspace,
                             const ResultsTpl<Scalar>& results) const
    {

    }

  protected:

    /// Maximum number \f$N_{\mathrm{max}}\f$ of Newton iterations.
    const std::size_t MAX_ITERS = 100;

    /// @brief  Put together the Q-function parameters and compute the Riccati gains.
    inline void computeGains(
      const ShootingProblemTpl<Scalar>& problem,
      Workspace& workspace,
      ResultsTpl<Scalar>& results,
      const std::size_t step)
    {
      using StageModel = StageModelTpl<Scalar>;
      using FunctionData = FunctionDataTpl<Scalar>;

      const StageModel& stage = problem.stages_[step];
      const std::size_t numc = stage.numConstraints();

      const value_store_t& vnext = workspace.value_params[step];
      q_store_t& qparams = workspace.q_params[step];

      const StageDataTpl<Scalar>& stage_data = workspace.problem_data->stage_data[step];
      const CostDataTpl<Scalar>& cdata = *stage_data.cost_data;
      const DynamicsDataTpl<Scalar>& dyn_data = *stage_data.dyn_data;
      using Constraint = StageConstraintTpl<Scalar>;

      int nprim = stage.numPrimal();
      int ndual = stage.numDual();
      int nu = stage.nu();
      int ndx1 = stage.ndx1();
      int ndx2 = stage.ndx2();

      assert(vnext.storage.rows() == ndx2 + 1);
      assert(vnext.storage.cols() == ndx2 + 1);

      // Use the contiguous full gradient/jacobian/hessian buffers
      // to fill in the Q-function derivatives
      auto qhess_autoadj = qparams.hess_.template selfadjointView<Eigen::Lower>();
      qparams.storage.setZero();

      qparams.q_2_ = 2 * cdata.value_;
      qparams.grad_.head(ndx1 + nu) = cdata.grad_;
      qparams.grad_.tail(ndx2) = vnext.Vx_;
      qhess_autoadj.topLeftCorner(ndx1 + nu, ndx1 + nu) = cdata.hess_;
      qhess_autoadj.bottomRightCorner(ndx2, ndx2) = vnext.Vxx_;

      // self-adjoint view to (nprim + ndual) sized block of kkt buffer
      auto kkt_mat = workspace.getKktView(nprim, ndual);
      MatrixRef kkt_rhs = workspace.kktRhsFull_.topLeftCorner(nprim + ndual, stage.ndx1() + 1);
      MatrixRef kkt_jac = kkt_mat.block(nprim, 0, ndual, nprim);

      VectorRef lambda_in = results.lams_[step];
      VectorRef lamprev = workspace.prev_lams_[step];
      VectorRef lamplus = workspace.lams_plus_[step];
      VectorRef lampdal = workspace.lams_pdal_[step];

      // Loop over constraints
      for (std::size_t i = 0; i < numc; i++)
      {

        const auto& cstr = stage.constraints_[i];
        const ConstraintSetBase<Scalar>& cstr_set = cstr->set_;
        const FunctionData& cstr_data = *stage_data.constraint_data[i];

        // Grab Lagrange multiplier segments

        VectorRef lam_i     = stage.constraints_.getSegmentByConstraint(lambda_in, i);
        VectorRef lamprev_i = stage.constraints_.getSegmentByConstraint(lamprev,   i);
        VectorRef lamplus_i = stage.constraints_.getSegmentByConstraint(lamplus,   i);
        VectorRef lampdal_i = stage.constraints_.getSegmentByConstraint(lampdal,   i);

        // Pre-projection multiplier estimate

        lamplus_i = lamprev_i + (1. / mu_) * cstr_data.value_;
        MatrixRef J_ = cstr_data.jac_buffer_;

        // compose Jacobian by projector and project multiplier

        cstr_set.applyNormalConeProjectionJacobian(lamplus_i, J_);
        lamplus_i.noalias() = cstr_set.normalConeProjection(lamplus_i);
        lampdal_i = 2 * lamplus_i - lambda_in;

        // update multiplier

        qparams.grad_.noalias() += J_.transpose() * lam_i;
        qhess_autoadj.noalias() += cstr_data.vhp_buffer_;
        stage.constraints_.getBlockByConstraint(kkt_jac, i) = J_;
        // kkt_jac.middleRows(cursor, nr) = J_;
      }

      // blocks: u, y, and dual
      kkt_rhs.col(0).head(nprim) = qparams.grad_.tail(nprim);
      kkt_rhs.col(0).tail(ndual) = mu_ * (workspace.lams_plus_[step] - results.lams_);

      // KKT matrix: (u, y)-block = bottom right of qhessadj
      kkt_mat.topLeftCorner(nprim, nprim) = qhess_autoadj.bottomRightCorner(nprim, nprim);
      // dual block
      kkt_mat.bottomRightCorner(ndual, ndual) = - mu_ * MatrixXs::Identity(ndual, ndual);

      /* Compute gains with LDLT */
      using kkt_mat_t = decltype(kkt_mat);
      Eigen::LDLT<kkt_mat_t> ldlt_ = kkt_mat.ldlt();
      workspace.gains_[step] = -kkt_rhs;
      ldlt_.solveInPlace(workspace.gains_[step]);


      /* Value function */
      value_store_t& v_current = workspace.value_params[step];

      v_current.storage = qparams.storage.topLeftCorner(ndx1 + 1, ndx1 + 1);
      // v_current.v_2_ = qparams.q_2_;
      // v_current.Vx_ = qparams.Qx_;
      // v_current.Vxx_ = qparams.Qxx_;
      v_current.storage = v_current.storage + kkt_rhs.transpose() * workspace.gains_[step];
    }

  };

  
} // namespace proxddp

#include "proxddp/solver-proxddp.hxx"
