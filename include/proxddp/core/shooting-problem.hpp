#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/stage-model.hpp"


#include "proxddp/modelling/state-error.hpp"


namespace proxddp
{
  /// @brief    Shooting problem, consisting in a succession of nodes.
  ///
  /// @details  The problem can be written as a nonlinear program:
  /// \f[
  ///   \begin{aligned}
  ///     \min_{\bfx,\bfu}~& \sum_{i=0}^{N-1} \ell_i(x_i, u_i) + \ell_N(x_N)  \\
  ///     \subjectto & \varphi(x_i, u_i, x_{i+1}) = 0, \ i \in [ 0, N-1 ] \\
  ///                & g(x_i, u_i) \in \calC_i
  ///   \end{aligned}
  /// \f]
  template<typename _Scalar>
  struct ShootingProblemTpl
  {
    using Scalar = _Scalar;
    using StageModel = StageModelTpl<Scalar>;
    using ProblemData = ShootingProblemDataTpl<Scalar>;
    using CostAbstract = CostAbstractTpl<Scalar>;

    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

    /// Initial condition
    VectorXs x0_init_;
    StateErrorResidual<Scalar> init_state_error;

    /// Stages of the control problem.
    std::vector<StageModel> stages_;
    shared_ptr<CostAbstract> term_cost_;

    ShootingProblemTpl(const VectorXs& x0, const std::vector<StageModel>& stages, const shared_ptr<CostAbstract>& term_cost)
      : x0_init_(x0)
      , init_state_error(stages[0].xspace1(), stages[0].nu(), x0_init_)
      , stages_(stages)
      , term_cost_(term_cost) {}

    ShootingProblemTpl(const VectorXs& x0, const int nu, const ManifoldAbstractTpl<Scalar>& space, const shared_ptr<CostAbstract>& term_cost)
      : x0_init_(x0)
      , init_state_error(space, nu, x0_init_)
      , term_cost_(term_cost) {}

    /// @brief Add a stage to the control problem.
    void addStage(const StageModel& new_stage);
    /// @copybrief addStage()
    void addStage(StageModel&& new_stage);

    inline std::size_t numSteps() const;

    /// @brief Rollout the problem costs, constraints, dynamics, stage per stage.
    void evaluate(const std::vector<VectorXs>& xs,
                  const std::vector<VectorXs>& us,
                  ProblemData& prob_data) const;

    /// @brief Rollout the problem derivatives, stage per stage.
    ///
    /// @param xs State sequence
    /// @param us Control sequence
    void computeDerivatives(const std::vector<VectorXs>& xs,
                            const std::vector<VectorXs>& us,
                            ProblemData& prob_data) const;

    shared_ptr<ProblemData> createData() const
    {
      return std::make_shared<ProblemData>(*this);
    }

  };

  /// @brief Problem data struct.
  template<typename _Scalar>
  struct ShootingProblemDataTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using StageDataPtr = shared_ptr<StageDataTpl<Scalar>>;

    shared_ptr<FunctionDataTpl<Scalar>> init_data;
    /// Data structs for each stage of the problem.
    std::vector<StageDataPtr> stage_data;
    /// Terminal cost data.
    shared_ptr<CostDataAbstract<Scalar>> term_cost_data;

    ShootingProblemDataTpl(const ShootingProblemTpl<Scalar>& problem);
  };
  
} // namespace proxddp

#include "proxddp/core/shooting-problem.hxx"
